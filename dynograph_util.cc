#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <algorithm>
#include <string>

#include "dynograph_util.h"
#include "logger.h"
#include "edgelist_dataset.h"
#include "rmat_dataset.h"
#include "helpers.h"
#include "args.h"
#include "pvector.h"

using namespace DynoGraph;
using std::cerr;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::stringstream;

int64_t
Batch::num_vertices_affected() const
{
    // Get a list of just the vertex ID's in this batch
    pvector<int64_t> vertices(size() * 2);
    std::transform(begin_iter, end_iter, vertices.begin(),
        [](const Edge& e){ return e.src; });
    std::transform(begin_iter, end_iter, vertices.begin() + size(),
        [](const Edge& e){ return e.dst; });

    // Deduplicate
    pvector<int64_t> unique_vertices(vertices.size());
    std::sort(vertices.begin(), vertices.end());
    auto end = std::unique_copy(vertices.begin(), vertices.end(), unique_vertices.begin());
    return static_cast<int64_t>(end - unique_vertices.begin());
}

int64_t
Batch::max_vertex_id() const {
    auto max_edge = std::max_element(begin_iter, end_iter,
        [](const Edge& a, const Edge& b) {
            return std::max(a.src, a.dst) < std::max(b.src, b.dst);
        }
    );
    return std::max(max_edge->src, max_edge->dst);
}

void
Batch::filter(int64_t threshold)
{
    begin_iter = std::find_if(begin_iter, end_iter,
        [threshold](const Edge& e) { return e.timestamp >= threshold; });
}

void
Batch::dedup_and_sort_by_out_degree()
{
    // Sort to prepare for deduplication
    auto by_src_dest_time = [](const Edge& a, const Edge& b) {
        // Order by src ascending, then dest ascending, then timestamp descending
        // This way the edge with the most recent timestamp will be picked when deduplicating
        return (a.src != b.src) ? a.src < b.src
             : (a.dst != b.dst) ? a.dst < b.dst
             :  a.timestamp > b.timestamp;
    };
    std::sort(begin_iter, end_iter, by_src_dest_time);

    // Deduplicate the edge list
    {
        pvector<Edge> deduped_edges(size());
        // Using std::unique_copy since there is no parallel version of std::unique
        auto end = std::unique_copy(begin_iter, end_iter, deduped_edges.begin(),
                // We consider only source and dest when searching for duplicates
                // The input is sorted, so we'll only get the most recent timestamp
                // BUG: Does not combine weights
                [](const Edge& a, const Edge& b) { return a.src == b.src && a.dst == b.dst; });
        // Copy deduplicated edges back into this batch
        std::transform(deduped_edges.begin(), end, begin_iter,
            [](const Edge& e) { return e; });
        // Adjust size
        size_t num_deduped_edges = (end - deduped_edges.begin());
        end_iter = begin_iter + num_deduped_edges;
    }

    // Allocate an array with an entry for each vertex
    pvector<int64_t> degrees(this->max_vertex_id()+1);
    #pragma omp parallel for
    for (int64_t i = 0; i < degrees.size(); ++i) {
        degrees[i] = i;
    }

    // Count the degree of each vertex
    std::transform(degrees.begin(), degrees.end(), degrees.begin(),
        [this](int64_t src) {
            Edge key = {src, 0, 0, 0};
            auto range = std::equal_range(begin_iter, end_iter, key,
                [](const Edge& a, const Edge& b) {
                    return a.src < b.src;
                }
            );
            return range.second - range.first;
        }
    );

    // Sort by out degree descending, src then dst
    auto by_out_degree = [&degrees](const Edge& a, const Edge& b) {
        if (degrees[a.src] != degrees[b.src]) {
            return degrees[a.src] > degrees[b.src];
        } else {
            return degrees[a.dst] > degrees[b.dst];
        }
    };
    std::sort(begin_iter, end_iter, by_out_degree);
}

shared_ptr<IDataset>
DynoGraph::create_dataset(const Args &args)
{
    DynoGraph::Logger& logger = DynoGraph::Logger::get_instance();

    shared_ptr<IDataset> dataset(nullptr);
    MPI_RANK_0_ONLY {
        if (has_suffix(args.input_path, ".rmat")) {
            // The suffix ".rmat" means we interpret input_path as a list of params, not as a literal path
            RmatArgs rmat_args(RmatArgs::from_string(args.input_path));
            std::string msg = rmat_args.validate();
            if (!msg.empty()) {
                logger << msg;
                die();
            }
            dataset = make_shared<RmatDataset>(args, rmat_args);

        } else if (has_suffix(args.input_path, ".graph.bin")
                   || has_suffix(args.input_path, ".graph.el"))
        {
            dataset = make_shared<EdgeListDataset>(args);

        } else {
            logger << "Unrecognized file extension for " << args.input_path << "\n";
            die();
        }
    }
    MPI_BARRIER();
#ifdef USE_MPI
    dataset = make_shared<ProxyDataset>(dataset);
#endif
    return dataset;
}

shared_ptr<Batch>
DynoGraph::get_preprocessed_batch(int64_t batchId, IDataset &dataset, Args::SORT_MODE sort_mode)
{
    int64_t threshold = dataset.getTimestampForWindow(batchId);

    switch (sort_mode)
    {
        case Args::SORT_MODE::UNSORTED:
        {
            shared_ptr<Batch> batch = dataset.getBatch(batchId);
            batch->filter(threshold);
            return batch;
        }
        case Args::SORT_MODE::PRESORT:
        {
            shared_ptr<Batch> batch = make_shared<ConcreteBatch>(
                std::move(*dataset.getBatch(batchId))
            );
            batch->filter(threshold);
            batch->dedup_and_sort_by_out_degree();
            return batch;
        }
        case Args::SORT_MODE::SNAPSHOT:
        {
            shared_ptr<Batch> cumulative_snapshot = make_shared<ConcreteBatch>(
                std::move(*dataset.getBatchesUpTo(batchId))
            );
            cumulative_snapshot->filter(threshold);
            cumulative_snapshot->dedup_and_sort_by_out_degree();
            return cumulative_snapshot;
        }
        default: assert(0); return nullptr;
    }
}

bool
DynoGraph::enable_algs_for_batch(int64_t batch_id, int64_t num_batches, int64_t num_epochs) {
    bool enable;
    MPI_RANK_0_ONLY {
    // How many batches in each epoch, on average?
    double batches_per_epoch = true_div(num_batches, num_epochs);
    // How many algs run before this batch?
    int64_t batches_before = round_down(true_div(batch_id, batches_per_epoch));
    // How many algs should run after this batch?
    int64_t batches_after = round_down(true_div((batch_id + 1), batches_per_epoch));
    // If the count changes between this batch and the next, we should run an alg now
    enable = (batches_after - batches_before) > 0;
    }
    MPI_BROADCAST_RESULT(enable);
    return enable;
}

void
DynoGraph::die()
{
    exit(-1);
}