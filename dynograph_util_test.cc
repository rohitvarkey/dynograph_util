
// Provides unit tests for classes and functionality in dynograph_util

#include "dynograph_util.h"
#include "reference_impl.h"
#include <gtest/gtest.h>

using namespace DynoGraph;

// Make sure empty args fail to validate
TEST(DynoGraphUtilTests, ArgValidationFail) {
    Args args;
    EXPECT_NE(args.validate(), "");
}

class DatasetTest: public ::testing::TestWithParam<Args> {
public:
    static std::vector<Args> all_args;
    static void init_arg_list()
    {
        Args args;
        args.input_path = "data/worldcup-10K.graph.bin";
        args.num_trials = 1;
        args.sort_mode = Args::SORT_MODE::UNSORTED;
        args.alg_names = {};

        for (int64_t batch_size : { 100, 500, 5000 }) {
            args.batch_size = batch_size;
            for (double window_size : { 0.1, 0.5, 1.0 }) {
                args.window_size = window_size;
                for (int64_t num_epochs : {3, 5, 7}) {
                    args.num_epochs = num_epochs;
                    all_args.push_back(args);
                }
            }
        }
    }
protected:
    const Dataset dataset;
    DatasetTest() : dataset(GetParam()) {}
};
std::vector<Args> DatasetTest::all_args;


TEST_P(DatasetTest, LoadDatasetCorrectly) {
    const Args &args = GetParam();

    // Check that args validate correctly
    EXPECT_EQ(args.validate(), "");

    // HACK somehow load in truth data about how many edges are actually in the dataset
    const int64_t actual_num_edges = 44500;

    // Check that the right number of edges were loaded
    EXPECT_EQ(dataset.getNumEdges(), actual_num_edges);
    // Check that the dataset was partitioned into the right number of batches
    EXPECT_EQ(dataset.getNumBatches(), actual_num_edges / args.batch_size);
    if (args.window_size == 1.0) {
        for (int64_t i = 0; i < dataset.getNumBatches(); ++i) {
            EXPECT_EQ(dataset.getBatch(i)->size(), args.batch_size);
        }
    }
}

TEST_P(DatasetTest, SetWindowThresholdCorrectly) {
    const Args &args = GetParam();

    double first_filtered_batch = args.window_size * dataset.getNumBatches();
    const Edge& first_edge = *dataset.getBatch(0)->begin();
    const Edge& last_edge = *(dataset.getBatch(dataset.getNumBatches()-1)->end()-1);
    int64_t min_ts = first_edge.timestamp;
    int64_t max_ts = last_edge.timestamp;
    EXPECT_GE(max_ts, min_ts);

    for (int64_t i = 0; i < dataset.getNumBatches(); ++i)
    {
        int64_t threshold = dataset.getTimestampForWindow(i);

        if (i < first_filtered_batch)
        {
            EXPECT_EQ(threshold, min_ts);
        } else {
            EXPECT_GE(threshold, min_ts);
            EXPECT_LE(threshold, max_ts);
        }
    }
}

TEST_P(DatasetTest, DontSkipAnyEpochs) {
    const Args &args = GetParam();

    int64_t actual_num_epochs = 0;
    for (int64_t i = 0; i < dataset.getNumBatches(); ++i)
    {
        bool run_epoch = dataset.enableAlgsForBatch(i);
        if (run_epoch) {
            actual_num_epochs += 1;
        }
        // There should always be an epoch after the last batch
        if (i == dataset.getNumBatches() - 1) {
            EXPECT_EQ(run_epoch, true);
        }
    }
    // Make sure the right number of epochs will be run
    EXPECT_EQ(args.num_epochs, actual_num_epochs);
}

INSTANTIATE_TEST_CASE_P(LoadDatasetCorrectlyForAllArgs, DatasetTest, ::testing::ValuesIn(DatasetTest::all_args));
INSTANTIATE_TEST_CASE_P(SetWindowThresholdCorrectlyForAllArgs, DatasetTest, ::testing::ValuesIn(DatasetTest::all_args));
INSTANTIATE_TEST_CASE_P(DontSkipAnyEpochsForAllArgs, DatasetTest, ::testing::ValuesIn(DatasetTest::all_args));

class SortModeTest: public ::testing::TestWithParam<Args> {
public:
    static std::vector<Args> all_args;
    static void init_arg_list()
    {
        Args args;
        args.input_path = "data/worldcup-10K.graph.bin";
        args.num_epochs = 1;
        args.num_trials = 1;
        args.alg_names = {};

        for (int64_t batch_size : { 100, 500, 5000 }) {
            args.batch_size = batch_size;
            for (double window_size : { 0.1, 0.5, 1.0 }) {
                args.window_size = window_size;
                all_args.push_back(args);
            }
        }

    }
protected:
    SortModeTest() {}
};
std::vector<Args> SortModeTest::all_args;

TEST_P(SortModeTest, SortModeDoesntAffectEdgeCount)
{
    DynoGraph::Args args = GetParam();

    // Load the same graph in three different sort modes
    args.sort_mode = DynoGraph::Args::SORT_MODE::UNSORTED;
    DynoGraph::Dataset unsorted_dataset(args);
    reference_impl unsorted_graph(args, unsorted_dataset.getMaxVertexId());

    args.sort_mode = DynoGraph::Args::SORT_MODE::PRESORT;
    DynoGraph::Dataset presort_dataset(args);
    reference_impl presort_graph(args, presort_dataset.getMaxVertexId());

    args.sort_mode = DynoGraph::Args::SORT_MODE::SNAPSHOT;
    DynoGraph::Dataset snapshot_dataset(args);
    reference_impl snapshot_graph(args, snapshot_dataset.getMaxVertexId());

    auto values_match = [](int64_t a, int64_t b, int64_t c) { return a == b && b == c; };

    // Make sure the resulting graphs are the same in each batch, regardless of sort mode
    for (int64_t batch = 0; batch < unsorted_dataset.getNumBatches(); ++batch)
    {
        // Do deletions and check the graphs against each other
        //    snapshot_graph is omitted from this comparison, because we don't do deletions in snapshot mode
        int64_t unsorted_threshold = unsorted_dataset.getTimestampForWindow(batch);
        int64_t presort_threshold = presort_dataset.getTimestampForWindow(batch);

        ASSERT_EQ(unsorted_threshold, presort_threshold);

        unsorted_graph.delete_edges_older_than(unsorted_threshold);
        presort_graph.delete_edges_older_than(presort_threshold);

        ASSERT_EQ(unsorted_graph.get_num_edges(), presort_graph.get_num_edges());
        ASSERT_EQ(unsorted_graph.get_num_vertices(), presort_graph.get_num_vertices());

        for (int64_t v = 0; v < unsorted_graph.get_num_vertices(); ++v)
        {
            ASSERT_EQ(unsorted_graph.get_out_degree(v), presort_graph.get_out_degree(v));
        }

        // Do insertions and check all three graphs against each other
        unsorted_graph.insert_batch(*unsorted_dataset.getBatch(batch));
        presort_graph.insert_batch(*presort_dataset.getBatch(batch));
        snapshot_graph.insert_batch(*snapshot_dataset.getBatch(batch));

        ASSERT_PRED3(values_match,
            unsorted_graph.get_num_edges(),
            presort_graph.get_num_edges(),
            snapshot_graph.get_num_edges()
        );
        ASSERT_PRED3(values_match,
            unsorted_graph.get_num_vertices(),
            presort_graph.get_num_vertices(),
            snapshot_graph.get_num_vertices()
        );

        for (int64_t v = 0; v < unsorted_graph.get_num_vertices(); ++v)
        {
            ASSERT_PRED3(values_match,
                unsorted_graph.get_out_degree(v),
                presort_graph.get_out_degree(v),
                snapshot_graph.get_out_degree(v)
            );
        }

        // Clear out snapshot graph before next batch
        snapshot_graph.~reference_impl();
        new(&snapshot_graph) reference_impl(args, snapshot_dataset.getMaxVertexId());
    }
}

INSTANTIATE_TEST_CASE_P(SortModeDoesntAffectEdgeCount, SortModeTest, ::testing::ValuesIn(SortModeTest::all_args));

int main(int argc, char **argv)
{
#ifdef USE_MPI
    // Initialize MPI
    boost::mpi::environment env(argc, argv);
#endif
    DatasetTest::init_arg_list();
    SortModeTest::init_arg_list();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

