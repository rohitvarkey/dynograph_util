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

using namespace DynoGraph;
using std::cerr;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::stringstream;

// Produces a filtered batch, in which all edges have timestamps newer than the threshold value
class FilteredBatch : public Batch
{
public:
    explicit FilteredBatch(const Batch& batch, int64_t threshold)
    : Batch(batch)
    {
        // Skip past edges that are older than the threshold
        begin_iter = std::find_if(batch.begin(), batch.end(),
            [threshold](const Edge& e) { return e.timestamp >= threshold; });
    }
};

// Produces a deduplicated batch, in which there are no duplicate edges
class DeduplicatedBatch : public Batch
{
protected:
    std::vector<Edge> deduped_edges;
public:
    explicit DeduplicatedBatch(const Batch& batch)
    : Batch(batch), deduped_edges(std::distance(batch.begin(), batch.end()))
    {
        // Make a copy of the original batch
        std::vector<Edge> sorted_edges(batch.begin(), batch.end()); // TODO is this init done in parallel?
        // Sort the edge list
        std::sort(sorted_edges.begin(), sorted_edges.end());

        // Deduplicate the edge list
        // Using std::unique_copy since there is no parallel version of std::unique
        auto end = std::unique_copy(sorted_edges.begin(), sorted_edges.end(), deduped_edges.begin(),
                // We consider only source and dest when searching for duplicates
                // The input is sorted, so we'll only get the most recent timestamp
                // BUG: Does not combine weights
                [](const Edge& a, const Edge& b) { return a.src == b.src && a.dst == b.dst; });
        deduped_edges.erase(end, deduped_edges.end());

        // Reinitialize the batch pointers
        begin_iter = deduped_edges.begin();
        end_iter = deduped_edges.end();
    }

    virtual int64_t num_vertices_affected() const
    {
        // Get a list of just the vertex ID's in this batch
        vector<int64_t> src_vertices(deduped_edges.size());
        vector<int64_t> dst_vertices(deduped_edges.size());
        std::transform(deduped_edges.begin(), deduped_edges.end(), src_vertices.begin(),
                [](const Edge& e){ return e.src; });
        std::transform(deduped_edges.begin(), deduped_edges.end(), dst_vertices.begin(),
                [](const Edge& e){ return e.dst; });
        src_vertices.insert(src_vertices.end(), dst_vertices.begin(), dst_vertices.end());

        // Deduplicate
        vector<int64_t> unique_vertices(src_vertices.size());
        std::sort(src_vertices.begin(), src_vertices.end());
        auto end = std::unique_copy(src_vertices.begin(), src_vertices.end(), unique_vertices.begin());
        unique_vertices.erase(end, unique_vertices.end());

        return unique_vertices.size();
    }
};

int64_t Batch::num_vertices_affected() const
{
    // We need to sort and deduplicate anyways, just use the implementation in DeduplicatedBatch
    auto sorted = DeduplicatedBatch(*this);
    return sorted.num_vertices_affected();
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
            shared_ptr<Batch> batch(dataset.getBatch(batchId));
            return make_shared<FilteredBatch>(*batch, threshold);;
        }
        case Args::SORT_MODE::PRESORT:
        {
            shared_ptr<Batch> batch(dataset.getBatch(batchId));
            return make_shared<DeduplicatedBatch>(FilteredBatch(*batch, threshold));
        }
        case Args::SORT_MODE::SNAPSHOT:
        {
            shared_ptr<Batch> cumulative_snapshot(dataset.getBatchesUpTo(batchId));
            return make_shared<DeduplicatedBatch>(FilteredBatch(*cumulative_snapshot, threshold));
        }
        default: assert(0); return nullptr;
    }
}

void
DynoGraph::die()
{
    exit(-1);
}