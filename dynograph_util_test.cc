
// Provides unit tests for classes and functionality in dynograph_util

#include "dynograph_util.h"
#include <gtest/gtest.h>

using namespace DynoGraph;

// Make sure empty args fail to validate
TEST(DynoGraphUtilTests, ArgValidationFail) {
    Args args;
    EXPECT_NE(args.validate(), "");
}

class DatasetTest: public ::testing::TestWithParam<Args> {
protected:
    const Dataset dataset;
    DatasetTest() : dataset(GetParam()) {}
};

TEST_P(DatasetTest, LoadDatasetCorrectly) {
    const Args &args = GetParam();

    // Check that args validate correctly
    EXPECT_EQ(args.validate(), "");

    // HACK somehow load in truth data about how many edges are actually in the dataset
    const int64_t actual_num_edges = 50;

    // Check that the right number of edges were loaded
    EXPECT_EQ(dataset.edges.size(), actual_num_edges);
    // Check that the dataset was partitioned into the right number of batches
    EXPECT_EQ(dataset.batches.size(), actual_num_edges / args.batch_size);
    for (int i = 0; i < dataset.batches.size(); ++i) {
        EXPECT_EQ(dataset.batches[i].size(), args.batch_size);
    }
}

TEST_P(DatasetTest, SetWindowThresholdCorrectly) {
    const Args &args = GetParam();

    double first_filtered_batch = args.window_size * dataset.batches.size();
    int64_t min_ts = dataset.edges.front().timestamp;
    int64_t max_ts = dataset.edges.back().timestamp;
    EXPECT_GE(max_ts, min_ts);

    for (int i = 0; i < dataset.batches.size(); ++i)
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
    for (int i = 0; i < dataset.batches.size(); ++i)
    {
        bool run_epoch = dataset.enableAlgsForBatch(i);
        if (run_epoch) {
            actual_num_epochs += 1;
        }
        // There should always be an epoch after the last batch
        if (i == dataset.batches.size() - 1) {
            EXPECT_EQ(run_epoch, true);
        }
    }
    // Make sure the right number of epochs will be run
    EXPECT_EQ(args.num_epochs, actual_num_epochs);
}


std::vector<Args> all_args;
void init_arg_list(std::vector<Args> &all_args, std::string path)
{
    Args args;
    args.num_epochs = 5;
    args.input_path = path;
    args.batch_size = 10;
    args.alg_names = {"cc", "pagerank"};
    args.sort_mode = Args::SORT_MODE::UNSORTED;
    args.window_size = 1.0;
    args.num_trials = 1;

    all_args.push_back(args);

    args.window_size = 0.5;

    all_args.push_back(args);
}

INSTANTIATE_TEST_CASE_P(LoadDatasetCorrectlyForAllArgs, DatasetTest, ::testing::ValuesIn(all_args));
INSTANTIATE_TEST_CASE_P(SetWindowThresholdCorrectlyForAllArgs, DatasetTest, ::testing::ValuesIn(all_args));
INSTANTIATE_TEST_CASE_P(DontSkipAnyEpochsForAllArgs, DatasetTest, ::testing::ValuesIn(all_args));

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Missing argument, need path to test graph\n";
        return -1;
    }
    init_arg_list(all_args, argv[1]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

