#pragma once

#include <hooks.h>
#include <inttypes.h>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <algorithm>

#include "range.h"
#include "pvector.h"
#include "mpi_macros.h"
#include "alg_data_manager.h"
#include "logger.h"
#include "args.h"
#include "batch.h"
#include "edge.h"

namespace DynoGraph {

class IDataset
{
public:
    virtual int64_t getTimestampForWindow(int64_t batchId) const = 0;
    virtual std::shared_ptr<Batch> getBatch(int64_t batchId) = 0;
    virtual std::shared_ptr<Batch> getBatchesUpTo(int64_t batchId) = 0;
    virtual int64_t getNumBatches() const = 0;
    virtual int64_t getNumEdges() const = 0;
    virtual bool isDirected() const = 0;
    virtual int64_t getMaxVertexId() const = 0;
    virtual int64_t getMinTimestamp() const = 0;
    virtual int64_t getMaxTimestamp() const = 0;
    virtual void reset() {};
    virtual ~IDataset() = default;
};

class DynamicGraph
{
public:
    const Args args;
    // Initialize the graph - your constructor must match this signature
    DynamicGraph(Args args, int64_t max_vertex_id) : args(args) {}
    // Return list of supported algs - your class must implement this method
    static std::vector<std::string> get_supported_algs();
    // Prepare to insert the batch
    virtual void before_batch(const Batch& batch, int64_t threshold) = 0;
    // Delete edges in the graph with a timestamp older than <threshold>
    virtual void delete_edges_older_than(int64_t threshold) = 0;
    // Insert the batch of edges into the graph
    virtual void insert_batch(const Batch& batch) = 0;
    // Run the specified algorithm
    virtual void update_alg(
        // Name of algorithm to run
        const std::string &alg_name,
        // Source vertex(s), (applies to bfs, sampled bc, sssp, etc.)
        const std::vector<int64_t> &sources,
        // Contains the results from the previous epoch (for incremental algorithms)
        // Results should be written to this array
        Range<int64_t> data
    ) = 0;
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
// May be useful for implementing DynamicGraph::get_high_degree_vertices
struct vertex_degree
{
    int64_t vertex_id;
    int64_t out_degree;
    vertex_degree() {}
    vertex_degree(int64_t vertex_id, int64_t out_degree)
    : vertex_id(vertex_id), out_degree(out_degree) {}
};
inline bool
operator < (const vertex_degree &a, const vertex_degree &b) {
    if (a.out_degree != b.out_degree) { return a.out_degree < b.out_degree; }
    return a.vertex_id > b.vertex_id;
}

std::shared_ptr<IDataset>
create_dataset(const Args &args);

std::shared_ptr<Batch>
get_preprocessed_batch(int64_t batchId, IDataset &dataset, Args::SORT_MODE sort_mode);

bool
enable_algs_for_batch(int64_t batch_id, int64_t num_batches, int64_t num_epochs);

template<typename graph_t>
void
run(int argc, char **argv)
{
    //static_assert(std::is_base_of<graph_t, DynamicGraph>(), "graph_t must implement DynoGraph::DynamicGraph");

    // Process command line arguments
    DynoGraph::Args args = DynoGraph::Args::parse(argc, argv);
    // Load graph data in from the file in batches
    std::shared_ptr<DynoGraph::IDataset> dataset = create_dataset(args);
    int64_t max_vertex_id = dataset->getMaxVertexId();
    // Initialize a buffer of data for each algorithm
    DynoGraph::AlgDataManager alg_data_manager(max_vertex_id+1, args.alg_names);
    DynoGraph::Logger &logger = DynoGraph::Logger::get_instance();
    Hooks& hooks = Hooks::getInstance();
    typedef DynoGraph::Args::SORT_MODE SORT_MODE;

    for (int64_t trial = 0; trial < args.num_trials; trial++)
    {
        hooks.set_attr("trial", trial);
        // Initialize the graph data structure
        graph_t graph(args, max_vertex_id);

        // Step through one batch at a time
        // Epoch will be incremented as necessary
        int64_t epoch = 0;
        int64_t num_batches = dataset->getNumBatches();
        for (int64_t batch_id = 0; batch_id < num_batches; ++batch_id)
        {
            hooks.set_attr("batch", batch_id);
            hooks.set_attr("epoch", epoch);

            // Normally, we construct the graph incrementally
            if (args.sort_mode != SORT_MODE::SNAPSHOT)
            {
                // Batch preprocessing (preprocess)
                hooks.region_begin("preprocess");
                std::shared_ptr<DynoGraph::Batch> batch = get_preprocessed_batch(batch_id, *dataset, args.sort_mode);
                hooks.region_end();

                int64_t threshold = dataset->getTimestampForWindow(batch_id);
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

                if (enable_algs_for_batch(batch_id, num_batches, args.num_epochs))
                {
                    logger << "Reinitializing graph\n";
                    // graph = graph_t(dataset, args) is no good here,
                    // because this would allocate a new graph before deallocating the old one.
                    // We probably won't have enough memory for that.
                    // Instead, use an explicit destructor call followed by placement new
                    hooks.region_begin("destroy");
                    graph.~graph_t();
                    new(&graph) graph_t(args, max_vertex_id);
                    hooks.region_end();

                    // This batch will be a cumulative, filtered snapshot of all the edges in previous batches
                    hooks.region_begin("preprocess");
                    std::shared_ptr<DynoGraph::Batch> batch = get_preprocessed_batch(batch_id, *dataset, args.sort_mode);
                    hooks.region_end();

                    // Graph construction benchmark (insertions)
                    logger << "Constructing graph for epoch " << epoch << "\n";
                    hooks.region_begin("insertions");
                    graph.insert_batch(*batch);
                    hooks.region_end();
                }
            }

            // Graph algorithm benchmarks
            if (enable_algs_for_batch(batch_id, num_batches, args.num_epochs)) {

                for (int64_t alg_trial = 0; alg_trial < args.num_alg_trials; ++alg_trial)
                {
                    // When we do multiple trials, algs should start with the same data each time
                    if (alg_trial != 0) { alg_data_manager.rollback(); }

                    // Run each alg
                    for (std::string alg_name : args.alg_names)
                    {
                        // Pick source vertex(s)
                        int64_t num_sources;
                        std::vector<int64_t> sources;
                        if (alg_name == "sssp") { num_sources = 1; }
                        else if (alg_name == "bc") { num_sources = 128; }
                        else { num_sources = 0; }
                        if (alg_name == "bfs") {
                            std::string path = args.source_path;
                            num_sources = 64;
                            sources.resize(num_sources);

                            FILE* fp = fopen(path.c_str(), "r");
                            int rc = 0;
                            printf("sources:\n");
                            for (int64_t* src = &sources[0]; rc != EOF; ++src)
                            {
                                rc = fscanf(fp, "%ld\n", src);
                                printf("%ld ", *src);
                            }
                            printf("\n");
                            fclose(fp);
                        }
                        else{
                            sources = graph.get_high_degree_vertices(num_sources);
                        }
                        if (sources.size() == 1) {
                            hooks.set_stat("source_vertex", sources[0]);
                        }

                        logger << "Running " << alg_name << " for epoch " << epoch << "\n";
                        hooks.set_stat("alg_trial", alg_trial);
                        hooks.set_stat("num_vertices", graph.get_num_vertices());
                        hooks.set_stat("num_edges", graph.get_num_edges());
                        hooks.region_begin(alg_name);
                        graph.update_alg(alg_name, sources, alg_data_manager.get_data_for_alg(alg_name));
                        hooks.region_end();
                    }
                }
                alg_data_manager.dump(epoch);
                alg_data_manager.next_epoch();
                epoch += 1;
                assert(epoch <= args.num_epochs);
            }
        }
        assert(epoch == args.num_epochs);
        // Reset dataset for next trial
        dataset->reset();
    }
}

// Terminate the benchmark in the event of an error
void die();

}; // end namespace DynoGraph

#ifdef USE_MPI
BOOST_IS_BITWISE_SERIALIZABLE(DynoGraph::vertex_degree);
BOOST_IS_BITWISE_SERIALIZABLE(DynoGraph::Edge)
#endif
