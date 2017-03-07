
// Provides unit tests for classes and functionality in dynograph_util

#include "dynograph_util.h"
#include "reference_impl.h"
#include "edgelist_dataset.h"
#include <gtest/gtest.h>
#include "pvector.h"

using namespace DynoGraph;

// Make sure empty args fail to validate
TEST(DynoGraphUtilTests, ArgValidationFail) {
    Args args;
    EXPECT_NE(args.validate(), "");
}

// Make sure we can filter a batch by timestamp
TEST(BatchTest, FilterBatch) {
    std::vector<Edge> edges = {
        {1, 2, 1, 100},
        {2, 3, 2, 200},
        {3, 4, 1, 300},
        {4, 5, 1, 400},
        {5, 6, 1, 500},
    };
    Batch batch(edges);
    ASSERT_EQ(batch.size(), 5);
    batch.filter(300);
    ASSERT_EQ(batch.size(), 3);
    ASSERT_EQ(batch[0], edges[2]);
    ASSERT_EQ(batch[1], edges[3]);
    ASSERT_EQ(batch[2], edges[4]);
}

// Make sure we can sort and dedup a batch by degree
TEST(BatchTest, DedupAndSortBatch) {
    std::vector<Edge> edges_before = {
        {1, 2, 1, 100},
        {2, 3, 1, 200},
        {2, 3, 1, 300},
        {2, 3, 1, 400},
        {2, 3, 1, 400},
        {2, 3, 1, 400},
        {2, 3, 1, 400},
        {2, 4, 1, 400},
        {3, 4, 1, 500},
        {3, 5, 1, 400},
        {3, 6, 1, 500},
        {3, 7, 1, 500},
        {3, 8, 1, 500},
        {3, 9, 1, 500},
    };

    std::vector<Edge> edges_after = {
        {3, 4, 1, 500},
        {3, 5, 1, 400},
        {3, 6, 1, 500},
        {3, 7, 1, 500},
        {3, 8, 1, 500},
        {3, 9, 1, 500},
        {2, 3, 1, 400},
        {2, 4, 1, 400},
        {1, 2, 1, 100},
    };
    Batch test_batch(edges_before);
    test_batch.dedup_and_sort_by_out_degree();
    Batch golden_batch(edges_after);

    auto batches_equal = [](const Batch& a, const Batch& b){
        if (a.size() != b.size()) {
            return ::testing::AssertionFailure();
        } else {
            bool equal = std::equal(a.begin(), a.end(), b.begin(),
                [](const Edge& x, const Edge& y) {
                    return x.src == y.src && x.dst == y.dst;
                }
            );
            return equal ? ::testing::AssertionSuccess() : ::testing::AssertionFailure();
        }
    };
    ASSERT_PRED2(batches_equal, test_batch, golden_batch);
}

TEST(BatchTest, GetMaxVertexId) {
    std::vector<Edge> edges = {
        {1, 2, 1, 100},
        {2, 3, 1, 200},
        {2, 3, 1, 300},
        {2, 3, 1, 400},
        {2, 3, 1, 400},
        {2, 3, 1, 400},
        {2, 10, 1, 400},
        {2, 4, 1, 400},
    };
    Batch batch(edges);
    ASSERT_EQ(batch.max_vertex_id(), 10);

    batch[2].src = 11;
    ASSERT_EQ(batch.max_vertex_id(), 11);
}

TEST(BatchTest, NumVerticesAffected) {
    std::vector<Edge> edges = {
        {1, 2, 1, 100},
        {2, 3, 1, 200},
        {2, 3, 1, 300},
        {2, 3, 1, 400},
        {2, 3, 1, 400},
        {2, 3, 1, 400},
        {2, 10, 1, 400},
        {2, 4, 1, 400},
    };
    Batch batch(edges);
    ASSERT_EQ(batch.num_vertices_affected(), 5);

    batch[2].src = 11;
    ASSERT_EQ(batch.num_vertices_affected(), 6);
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

        for (std::string input_path : { "data/worldcup-10K.graph.bin", "0.55-0.20-0.10-0.15-44500-8K.rmat" }) {
            args.input_path = input_path;
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
    }
protected:
    std::shared_ptr<IDataset> dataset;
    DatasetTest() : dataset(create_dataset(GetParam())) {}
};
std::vector<Args> DatasetTest::all_args;


TEST_P(DatasetTest, LoadDatasetCorrectly) {
    const Args &args = GetParam();

    // Check that args validate correctly
    EXPECT_EQ(args.validate(), "");

    // HACK somehow load in truth data about how many edges are actually in the dataset
    const int64_t actual_num_edges = 44500;

    // Check that the right number of edges were loaded
    EXPECT_EQ(dataset->getNumEdges(), actual_num_edges);
    // Check that the dataset was partitioned into the right number of batches
    EXPECT_EQ(dataset->getNumBatches(), actual_num_edges / args.batch_size);
    if (args.window_size == 1.0) {
        for (int64_t i = 0; i < dataset->getNumBatches(); ++i) {
            EXPECT_EQ(dataset->getBatch(i)->size(), args.batch_size);
        }
    }
}

// TODO this isn't a dataset test anymore, now that enable_algs_for_batch is a free function
TEST_P(DatasetTest, DontSkipAnyEpochs) {
    const Args &args = GetParam();

    int64_t actual_num_epochs = 0;
    int64_t num_batches = dataset->getNumBatches();
    for (int64_t i = 0; i < num_batches; ++i)
    {
        bool run_epoch = enable_algs_for_batch(i, num_batches, args.num_epochs);
        if (run_epoch) {
            actual_num_epochs += 1;
        }
        // There should always be an epoch after the last batch
        if (i == num_batches - 1) {
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
    typedef DynoGraph::Args::SORT_MODE SORT_MODE;
    args.sort_mode = SORT_MODE::UNSORTED;
    DynoGraph::EdgeListDataset dataset(args);
    int64_t max_vertex_id = dataset.getMaxVertexId();

    // Load the same graph in three different sort modes
    args.sort_mode = SORT_MODE::UNSORTED;
    reference_impl unsorted_graph(args, max_vertex_id);

    args.sort_mode = SORT_MODE::PRESORT;
    reference_impl presort_graph(args, max_vertex_id);

    args.sort_mode = SORT_MODE::SNAPSHOT;
    reference_impl snapshot_graph(args, max_vertex_id);

    auto values_match = [](int64_t a, int64_t b, int64_t c) { return a == b && b == c; };

    // Make sure the resulting graphs are the same in each batch, regardless of sort mode
    for (int64_t batch = 0; batch < dataset.getNumBatches(); ++batch)
    {
        // Do deletions and check the graphs against each other
        //    snapshot_graph is omitted from this comparison, because we don't do deletions in snapshot mode
        int64_t unsorted_threshold = dataset.getTimestampForWindow(batch);
        int64_t presort_threshold = dataset.getTimestampForWindow(batch);

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
        unsorted_graph.insert_batch(*get_preprocessed_batch(batch, dataset, SORT_MODE::UNSORTED));
        presort_graph.insert_batch(*get_preprocessed_batch(batch, dataset, SORT_MODE::PRESORT));
        snapshot_graph.insert_batch(*get_preprocessed_batch(batch, dataset, SORT_MODE::SNAPSHOT));

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
        new(&snapshot_graph) reference_impl(args, max_vertex_id);
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

