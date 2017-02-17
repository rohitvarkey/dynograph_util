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

#include "mpi_macros.h"

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
    // Number of times to repeat each epoch
    int64_t num_alg_trials;

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
std::ostream& operator <<(std::ostream& os, const Edge& e);


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

class FilteredBatch : public Batch
{
public:
    explicit FilteredBatch(const Batch& batch, int64_t threshold);
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
    int64_t max_vertex_id;
    int64_t min_timestamp;
    int64_t max_timestamp;

    std::vector<Edge> edges;
    std::vector<Batch> batches;

public:
    Dataset(Args args);

    int64_t getTimestampForWindow(int64_t batchId) const;
    std::shared_ptr<Batch> getBatch(int64_t batchId) const;
    int64_t getNumBatches() const;
    int64_t getNumEdges() const;

    bool isDirected() const;
    int64_t getMaxVertexId() const;

    bool enableAlgsForBatch(int64_t i) const;
};

class DynamicGraph
{
public:
    const Args args;
    // Initialize the graph - your constructor must match this signature
    DynamicGraph(Args args, int64_t max_vertex_id);
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
    // Return a list of the n vertices with the highest degrees
    virtual std::vector<int64_t> get_high_degree_vertices(int64_t n) const = 0;
};

// Holds a vertex id and its out degree
struct vertex_degree
{
    int64_t vertex_id;
    int64_t out_degree;
    vertex_degree();
    vertex_degree(int64_t vertex_id, int64_t out_degree);
};
inline bool
operator < (const vertex_degree &a, const vertex_degree &b) {
    if (a.out_degree != b.out_degree) { return a.out_degree < b.out_degree; }
    return a.vertex_id > b.vertex_id;
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
        MPI_RANK_0_ONLY {
        oss << std::forward<T>(x);
        // Once we have a full line, print with prefix
        if (oss.str().back() == '\n') {
            out << msg << oss.str();
            oss.str("");
        }
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
    typedef DynoGraph::Args::SORT_MODE SORT_MODE;

    for (int64_t trial = 0; trial < args.num_trials; trial++)
    {
        hooks.set_attr("trial", trial);
        // Initialize the graph data structure
        graph_t graph(args, dataset.getMaxVertexId());

        // Step through one batch at a time
        // Epoch will be incremented as necessary
        int64_t epoch = 0;
        int64_t num_batches = dataset.getNumBatches();
        for (int64_t batch_id = 0; batch_id < num_batches; ++batch_id)
        {
            hooks.set_attr("batch", batch_id);
            hooks.set_attr("epoch", epoch);

            // Normally, we construct the graph incrementally
            if (args.sort_mode != SORT_MODE::SNAPSHOT)
            {
                // Batch preprocessing (preprocess)
                hooks.region_begin("preprocess");
                std::shared_ptr<DynoGraph::Batch> batch = dataset.getBatch(batch_id);
                hooks.region_end();

                int64_t threshold = dataset.getTimestampForWindow(batch_id);
                graph.before_batch(*batch, threshold);

                // Edge deletion benchmark (deletions)
                if (args.window_size != 1.0)
                {
                    logger << "Deleting edges older than " << threshold << "\n";

                    hooks.set_stat("num_vertices", graph.get_num_vertices());
                    hooks.set_stat("num_edges", graph.get_num_edges());
                    hooks.region_begin("deletions");
                    graph.delete_edges_older_than(threshold);
                    hooks.region_end();
                }

                // Edge insertion benchmark (insertions)
                logger << "Inserting batch " << batch_id << "\n";
                hooks.set_stat("num_vertices", graph.get_num_vertices());
                hooks.set_stat("num_edges", graph.get_num_edges());
                hooks.region_begin("insertions");
                graph.insert_batch(*batch);
                hooks.region_end();

            // In snapshot mode, we construct a new graph before each epoch
            } else if (args.sort_mode == SORT_MODE::SNAPSHOT) {

                if (dataset.enableAlgsForBatch(batch_id))
                {
                    logger << "Reinitializing graph\n";
                    // graph = graph_t(dataset, args) is no good here,
                    // because this would allocate a new graph before deallocating the old one.
                    // We probably won't have enough memory for that.
                    // Instead, use an explicit destructor call followed by placement new
                    graph.~graph_t();
                    new(&graph) graph_t(args, dataset.getMaxVertexId());

                    // This batch will be a cumulative, filtered snapshot of all the edges in previous batches
                    hooks.region_begin("preprocess");
                    std::shared_ptr<DynoGraph::Batch> batch = dataset.getBatch(batch_id);
                    hooks.region_end();

                    // Graph construction benchmark (insertions)
                    logger << "Constructing graph for epoch " << epoch << "\n";
                    hooks.region_begin("insertions");
                    graph.insert_batch(*batch);
                    hooks.region_end();

                }
            }

            // Graph algorithm benchmarks
            if (dataset.enableAlgsForBatch(batch_id)) {
                hooks.set_stat("num_vertices", graph.get_num_vertices());
                hooks.set_stat("num_edges", graph.get_num_edges());

                for (int64_t alg_trial = 0; alg_trial < args.num_alg_trials; ++alg_trial)
                {
                    hooks.set_stat("alg_trial", alg_trial);
                    for (std::string alg_name : args.alg_names)
                    {
                        int64_t num_sources;
                        if (alg_name == "bfs" || alg_name == "sssp") { num_sources = 1; }
                        else if (alg_name == "bc") { num_sources = 128; }
                        else { num_sources = 0; }
                        std::vector<int64_t> sources = graph.get_high_degree_vertices(num_sources);
                        if (sources.size() == 1) {
                            hooks.set_stat("source_vertex", sources[0]);
                        }
                        logger << "Running " << alg_name << " for epoch " << epoch << "\n";
                        hooks.region_begin(alg_name);
                        graph.update_alg(alg_name, sources);
                        hooks.region_end();
                    }
                }
                epoch += 1;
                assert(epoch <= args.num_epochs);
            }
        }
        assert(epoch == args.num_epochs);
    }
}

// Terminate the benchmark in the event of an error
void die();

}; // end namespace DynoGraph

#ifdef USE_MPI
BOOST_IS_BITWISE_SERIALIZABLE(DynoGraph::vertex_degree);
BOOST_IS_BITWISE_SERIALIZABLE(DynoGraph::Edge)
#endif
