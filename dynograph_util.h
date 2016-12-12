#pragma once

#include <hooks.h>
#include <inttypes.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>
#include <assert.h>
#ifdef USE_MPI
#include <omp.h>
#endif

namespace DynoGraph {

const std::string msg = "[DynoGraph] ";

struct Args
{
    std::vector<std::string> alg_names;
    std::string input_path;
    enum SORT_MODE {
        // Do not pre-sort batches
        UNSORTED,
        // Sort and deduplicate each batch before returning it
        PRESORT,
        // Each batch is a cumulative snapshot of all edges in previous batches
        SNAPSHOT
    } sort_mode;
    int64_t window_size;
    int64_t num_batches;
    int64_t num_trials;
    int64_t enable_deletions;

    Args(int argc, char **argv);
};

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
protected:
    typedef std::vector<Edge>::const_iterator iterator;
    iterator begin_iter, end_iter;
public:
    iterator begin() const;
    iterator end() const;
    const Dataset& dataset;
    Batch(iterator begin, iterator end, const Dataset &dataset);
    virtual int64_t num_vertices_affected() const;
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
    void initBatchIterators();

    Args args;
    bool directed;
    int64_t maxNumVertices;

public:

    std::vector<Edge> edges;
    std::vector<Batch> batches;

    Dataset(Args args);
    Dataset(std::vector<Edge> edges, Args& args, int64_t maxNumVertices);

    int64_t getTimestampForWindow(int64_t batchId) const;
    std::shared_ptr<Batch> getBatch(int64_t batchId) const;

    bool isDirected() const;
    int64_t getMaxNumVertices() const;

    std::vector<Batch>::const_iterator begin() const;
    std::vector<Batch>::const_iterator end() const;
};

class DynamicGraph
{
protected:
    const Dataset& dataset;
    const Args& args;
public:
    // Initialize the graph - your constructor must match this signature
    DynamicGraph(const Dataset& dataset, const Args& args) : dataset(dataset), args(args) {};
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
typedef std::pair<int64_t, int64_t> vertex_degree;

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
        degrees[i] = std::make_pair(i, degree);
    }

    // order by degree descending, vertex_id ascending
    std::sort(degrees.begin(), degrees.end(),
        [](const vertex_degree &a, const vertex_degree &b) {
            if (a.second != b.second) { return a.second > b.second; }
            return a.first < b.first;
        }
    );

    degrees.erase(degrees.begin() + top_n, degrees.end());
    std::vector<vertex_t> ids(degrees.size());
    std::transform(degrees.begin(), degrees.end(), ids.begin(),
        [](const vertex_degree &d) { return d.first; });
    return ids;
}

#ifdef USE_MPI
void register_vertex_degree_type(MPI_Datatype *type);
void vertex_degree_reducer(vertex_degree *a, vertex_degree *b, int *len, MPI_Datatype *datatype);
#endif

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
        // Register a type with MPI to hold a tuple of (vertex_id, degree)
        MPI_Datatype vertex_degree_type;
        register_vertex_degree_type(&vertex_degree_type);
        // Set the local max-degree vertex
        vertex_degree local_vertex_degree = std::make_pair(sources[0], graph.get_out_degree(sources[0]));
        // Create a custom MPI reducer function that matches the sort logic of find_high_degree_vertices
        MPI_Op compare_op;
        MPI_Op_create((MPI_User_function *)(vertex_degree_reducer), false, &compare_op);
        // Reduce, giving all ranks the same source vertex
        vertex_degree global_vertex_degree;
        MPI_Allreduce(&local_vertex_degree, &global_vertex_degree, 1, vertex_degree_type, compare_op, MPI_COMM_WORLD);
        MPI_Op_free(&compare_op);
        sources = {global_vertex_degree.first};
#endif
    }
    return sources;
}

template<typename graph_t>
void
run(int argc, char **argv)
{
    //static_assert(std::is_base_of<graph_t, DynamicGraph>(), "graph_t must implement DynoGraph::DynamicGraph");

    // Process command line arguments
    DynoGraph::Args args(argc, argv);
    // Load graph data in from the file in batches
    DynoGraph::Dataset dataset(args);
    Hooks& hooks = Hooks::getInstance();

    for (int64_t trial = 0; trial < args.num_trials; trial++)
    {
        hooks.set_attr("trial", trial);
        // Initialize the graph data structure
        graph_t graph(dataset, args);

        // Run the algorithm(s) after each inserted batch
        for (int64_t batch_id = 0; batch_id < dataset.batches.size(); ++batch_id)
        {
            hooks.set_attr("batch", batch_id);
            hooks.region_begin("preprocess");
            std::shared_ptr<DynoGraph::Batch> batch = dataset.getBatch(batch_id);
            hooks.region_end();

            int64_t threshold = dataset.getTimestampForWindow(batch_id);
            graph.before_batch(*batch, threshold);

            if (args.enable_deletions)
            {
                std::cerr << msg << "Deleting edges older than " << threshold << "\n";
                hooks.region_begin("deletions");
                graph.delete_edges_older_than(threshold);
                hooks.region_end();
            }

            std::cerr << msg << "Inserting batch " << batch_id << "\n";
            hooks.region_begin("insertions");
            graph.insert_batch(*batch);
            hooks.region_end();

            for (std::string alg_name : args.alg_names)
            {
                std::vector<int64_t> sources = pick_sources_for_alg(alg_name, graph);
                std::cerr << msg << "Running " << alg_name << "\n";
                hooks.region_begin(alg_name);
                graph.update_alg(alg_name, sources);
                hooks.region_end();
            }

            // Clear out the graph between batches in snapshot mode
            if (args.sort_mode == DynoGraph::Args::SNAPSHOT)
            {
                // graph = graph_t(dataset, args) is no good here,
                // because this would allocate a new graph before deallocating the old one.
                // We probably won't have enough memory for that.
                // Instead, use an explicit destructor call followed by placement new
                graph.~graph_t();
                new(&graph) graph_t(dataset, args);
            }
        }
    }
}

}; // end namespace DynoGraph