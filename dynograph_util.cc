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