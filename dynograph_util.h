#pragma once

#include <hooks.h>
#include <inttypes.h>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <assert.h>
#ifdef USE_MPI
#include <omp.h>
#endif

namespace DynoGraph {

struct Args
{
    // Number of epochs in the benchmark
    int64_t num_epochs;
    // File path for edge list to load
    std::string input_path;
    // Number of edges to insert in each batch of insertions
    int64_t batch_size;
    // Algorithms to run after each epoch
    std::vector<std::string> alg_names;
    // Batch sort mode:
    enum class SORT_MODE {
        // Do not pre-sort batches
        UNSORTED,
        // Sort and deduplicate each batch before returning it
        PRESORT,
        // Each batch is a cumulative snapshot of all edges in previous batches
        SNAPSHOT
    } sort_mode;
    // Percentage of the graph to hold in memory
    double window_size;
    // Number of times to repeat the benchmark
    int64_t num_trials;

    Args() = default;
    std::string validate() const;

    static Args parse(int argc, char **argv);
    static void print_help(std::string argv0);
};

std::ostream& operator <<(std::ostream& os, Args::SORT_MODE sort_mode);
std::ostream& operator <<(std::ostream& os, const Args& args);

struct Edge
{
    int64_t src;
    int64_t dst;
    int64_t weight;
    int64_t timestamp;
};

bool operator<(const Edge& a, const Edge& b);
bool operator==(const Edge& a, const Edge& b);

// Forward reference
class Dataset;

class Batch
{
public:
    typedef std::vector<Edge>::const_iterator iterator;
    iterator begin() const;
    iterator end() const;
    Batch(iterator begin, iterator end);
    virtual int64_t num_vertices_affected() const;
    const Edge& operator[] (size_t i) const;
    size_t size() const;
    bool is_directed() const;
protected:
    iterator begin_iter, end_iter;
};

class DeduplicatedBatch : public Batch
{
protected:
    std::vector<Edge> deduped_edges;
public:
    explicit DeduplicatedBatch(const Batch& batch);
    virtual int64_t num_vertices_affected() const;
};

class Dataset
{
private:
    void loadEdgesBinary(std::string path);
    void loadEdgesAscii(std::string path);

    Args args;
    bool directed;
    int64_t max_num_vertices;
    int64_t min_timestamp;
    int64_t max_timestamp;

public:

    std::vector<Edge> edges;
    std::vector<Batch> batches;

    Dataset(Args args);

    int64_t getTimestampForWindow(int64_t batchId) const;
    std::shared_ptr<Batch> getBatch(int64_t batchId) const;

    bool isDirected() const;
    int64_t getMaxNumVertices() const;

    std::vector<Batch>::const_iterator begin() const;
    std::vector<Batch>::const_iterator end() const;

    bool enableAlgsForBatch(int64_t i) const;
};

class DynamicGraph
{
protected:
    const Args& args;
public:
    // Initialize the graph - your constructor must match this signature
    DynamicGraph(const Args& args, int64_t max_nv);
    // Return list of supported algs - your class must implement this method
    static std::vector<std::string> get_supported_algs();
    // Prepare to insert the batch
    virtual void before_batch(const Batch& batch, int64_t threshold) = 0;
    // Delete edges in the graph with a timestamp older than <threshold>
    virtual void delete_edges_older_than(int64_t threshold) = 0;
    // Insert the batch of edges into the graph
    virtual void insert_batch(const Batch& batch) = 0;
    // Run the specified algorithm
    virtual void update_alg(const std::string &alg_name, const std::vector<int64_t> &sources) = 0;
    // Return the degree of the specified vertex
    virtual int64_t get_out_degree(int64_t vertex_id) const = 0;
    // Return the number of vertices in the graph
    virtual int64_t get_num_vertices() const = 0;
    // Return the number of unique edges in the graph
    virtual int64_t get_num_edges() const = 0;
};

// Holds a vertex id and its out degree
struct vertex_degree
{
    int64_t vertex_id;
    int64_t out_degree;
    vertex_degree();
    vertex_degree(int64_t vertex_id, int64_t out_degree);
};
bool operator < (const vertex_degree &a, const vertex_degree &b);

/**
 * Returns a list of the highest N vertices in the graph
 * @param top_n Number of vertex ID's to return
 * @param nv Total number of vertices in the graph
 * @param get_degree Function which returns the degree of the specified vertex
 *        int64_t get_degree(int64_t vertex_id);
 * @return list of the top_n vertices with highest degree
 */
template <typename degree_getter, typename vertex_t>
std::vector<vertex_t>
find_high_degree_vertices(vertex_t top_n, vertex_t nv, degree_getter get_degree)
{
    top_n = std::min(top_n, nv);

    std::vector<vertex_degree> degrees(nv);
    #pragma omp parallel for
    for (vertex_t i = 0; i < nv; ++i) {
        int64_t degree = get_degree(i);
        degrees[i] = vertex_degree(i, degree);
    }

    // order by degree descending, vertex_id ascending
    std::sort(degrees.begin(), degrees.end());

    degrees.erase(degrees.begin() + top_n, degrees.end());
    std::vector<vertex_t> ids(degrees.size());
    std::transform(degrees.begin(), degrees.end(), ids.begin(),
        [](const vertex_degree &d) { return d.vertex_id; });
    return ids;
}

template<typename graph_t>
std::vector<int64_t>
pick_sources_for_alg(std::string alg_name, graph_t &graph)
{
    int64_t num_sources;
    if (alg_name == "bfs" || alg_name == "sssp") { num_sources = 1; }
    else if (alg_name == "bc") { num_sources = 128; }
    else { num_sources = 0; }

    int64_t nv = graph.get_num_vertices();
    num_sources = std::min(num_sources, nv);

    auto get_degree = [&graph](int64_t i){ return graph.get_out_degree(i); };
    std::vector<int64_t> sources;
    if (num_sources > 0) {
        sources = find_high_degree_vertices(num_sources, nv, get_degree);

#ifdef USE_MPI
        // For now, Boost only has bfs and sssp, which require just one source vertex
        assert(num_sources == 1);
        // Register the vertex_degree type with MPI
        MPI_Datatype vertex_degree_type;
        MPI_Type_contiguous(2, MPI_INT64_T, &vertex_degree_type);
        MPI_Type_commit(&vertex_degree_type);
        // Set the local max-degree vertex
        vertex_degree local_vertex_degree(sources[0], get_degree(sources[0]));
        // Gather all in rank 0
        int comm_size;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        std::vector<vertex_degree> gathered_vertex_degrees(comm_size);
        MPI_Gather(
            &local_vertex_degree, 1, vertex_degree_type,
            gathered_vertex_degrees.data(), 1, vertex_degree_type,
            0, MPI_COMM_WORLD
        );
        // Find max-degree vertex of all ranks
        vertex_degree global_vertex_degree = *std::max_element(
            gathered_vertex_degrees.begin(), gathered_vertex_degrees.end());
        // Broadcast to all ranks
        MPI_Bcast(&global_vertex_degree, 1, vertex_degree_type, 0, MPI_COMM_WORLD);
        sources = {global_vertex_degree.vertex_id};
#endif
    }
    return sources;
}

class Logger
{
protected:
    const std::string msg = "[DynoGraph] ";
    std::ostream &out;
    std::ostringstream oss;
    Logger (std::ostream &out);
public:
    // Print to error stream with prefix
    template <class T>
    Logger& operator<<(T&& x) {
#ifdef USE_MPI
        // Only print logging statements in rank 0 to keep output clean
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0) { return *this; }
#endif
        oss << std::forward<T>(x);
        // Once we have a full line, print with prefix
        if (oss.str().back() == '\n') {
            out << msg << oss.str();
            oss.str("");
        }
        return *this;
    }
    // Handle IO manipulators
    Logger& operator<<(std::ostream& (*manip)(std::ostream&));
    // Singleton getter
    static Logger& get_instance();
    virtual ~Logger();
};

template<typename graph_t>
void
run(int argc, char **argv)
{
    //static_assert(std::is_base_of<graph_t, DynamicGraph>(), "graph_t must implement DynoGraph::DynamicGraph");

    // Process command line arguments
    DynoGraph::Args args = DynoGraph::Args::parse(argc, argv);
    // Load graph data in from the file in batches
    DynoGraph::Dataset dataset(args);
    DynoGraph::Logger &logger = DynoGraph::Logger::get_instance();
    Hooks& hooks = Hooks::getInstance();

    for (int64_t trial = 0; trial < args.num_trials; trial++)
    {
        hooks.set_attr("trial", trial);
        // Initialize the graph data structure
        graph_t graph(args, dataset.getMaxNumVertices());

        // Run the algorithm(s) after each inserted batch
        for (int64_t batch_id = 0; batch_id < dataset.batches.size(); ++batch_id)
        {
            hooks.set_attr("batch", batch_id);
            hooks.region_begin("preprocess");
            std::shared_ptr<DynoGraph::Batch> batch = dataset.getBatch(batch_id);
            hooks.region_end();

            int64_t threshold = dataset.getTimestampForWindow(batch_id);
            graph.before_batch(*batch, threshold);

            hooks.set_stat("num_vertices", graph.get_num_vertices());
            hooks.set_stat("num_edges", graph.get_num_edges());

            if (args.window_size != 1.0)
            {
                logger << "Deleting edges older than " << threshold << "\n";
                hooks.region_begin("deletions");
                graph.delete_edges_older_than(threshold);
                hooks.region_end();
            }

            hooks.set_stat("num_vertices", graph.get_num_vertices());
            hooks.set_stat("num_edges", graph.get_num_edges());

            logger << "Inserting batch " << batch_id << "\n";
            hooks.region_begin("insertions");
            graph.insert_batch(*batch);
            hooks.region_end();

            hooks.set_stat("num_vertices", graph.get_num_vertices());
            hooks.set_stat("num_edges", graph.get_num_edges());

            if (dataset.enableAlgsForBatch(batch_id)) {
                for (std::string alg_name : args.alg_names)
                {
                    std::vector<int64_t> sources = pick_sources_for_alg(alg_name, graph);
                    if (sources.size() == 1) {
                        hooks.set_stat("source_vertex", sources[0]);
                    }
                    logger << "Running " << alg_name << "\n";
                    hooks.region_begin(alg_name);
                    graph.update_alg(alg_name, sources);
                    hooks.region_end();
                }
            }

            // Clear out the graph between batches in snapshot mode
            if (args.sort_mode == DynoGraph::Args::SORT_MODE::SNAPSHOT)
            {
                logger << "Reinitializing graph\n";
                // graph = graph_t(dataset, args) is no good here,
                // because this would allocate a new graph before deallocating the old one.
                // We probably won't have enough memory for that.
                // Instead, use an explicit destructor call followed by placement new
                graph.~graph_t();
                new(&graph) graph_t(args, dataset.getMaxNumVertices());
            }
        }
    }
}

// Terminate the benchmark in the event of an error
void die();

}; // end namespace DynoGraph