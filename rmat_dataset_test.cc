
// Provides unit tests for the RmatDataset class

#include "dynograph_util.h"
#include "reference_impl.h"
#include "rmat_dataset.h"
#include <gtest/gtest.h>

using namespace DynoGraph;

TEST(RmatArgsTest, ParseFromString)
{
    RmatArgs rmat_args = RmatArgs::from_string("0.55-0.20-0.10-0.15-12G-8M.rmat");
    EXPECT_EQ(rmat_args.a, 0.55);
    EXPECT_EQ(rmat_args.b, 0.20);
    EXPECT_EQ(rmat_args.c, 0.10);
    EXPECT_EQ(rmat_args.d, 0.15);
    EXPECT_EQ(rmat_args.num_edges, 12LL * 1024 * 1024 * 1024);
    EXPECT_EQ(rmat_args.num_vertices, 8LL * 1024 * 1024);
}

TEST(RmatArgsTest, Validation)
{
    RmatArgs rmat_args;
    rmat_args = RmatArgs::from_string("0.55-0.20-0.10-0.15-12G-8M.rmat");
    EXPECT_TRUE(rmat_args.validate().empty());
    // Values don't sum to one
    rmat_args = RmatArgs::from_string("0.35-0.20-0.10-0.15-12G-8M.rmat");
    EXPECT_FALSE(rmat_args.validate().empty());
}


TEST(RmatDatasetTest, DeterministicParallelGeneration)
{
    Args args = {1, "dummy", 1000, {}, DynoGraph::Args::SORT_MODE::SNAPSHOT, 1.0, 1, 1};
    RmatArgs rmat_args = RmatArgs::from_string("0.55-0.20-0.10-0.15-1M-10K.rmat");
    RmatDataset dataset(args, rmat_args);

    omp_set_num_threads(1);

    std::shared_ptr<Batch> serial_batch = dataset.getBatch(0);
    EXPECT_EQ(serial_batch->size(), args.batch_size);

    dataset.reset();

    omp_set_num_threads(omp_get_max_threads());
    std::shared_ptr<Batch> parallel_batch = dataset.getBatch(0);
    EXPECT_EQ(parallel_batch->size(), args.batch_size);

    EXPECT_EQ(*serial_batch, *parallel_batch);
}