#pragma once

// Provides a test template for implementations of DynoGraph::DynamicGraph

#include <dynograph_util.h>
#include <gtest/gtest.h>

using namespace DynoGraph;

// Test fixture, to be templated with a class that implements DynamicGraph
template <typename graph_t>
class ImplTest : public ::testing::Test {
private:
    graph_t graph;
protected:
    Args args;
    DynamicGraph &impl;
public:
    ImplTest()
    : graph({1, "dummy", 1, graph_t::get_supported_algs(), Args::SORT_MODE::UNSORTED, 1.0, 1}, 100)
    , impl(graph)
    {}
};
TYPED_TEST_CASE_P(ImplTest);

// Make sure it can run all the algs it advertises
TYPED_TEST_P(ImplTest, CheckAlgs)
{
    // Create a simple graph
    std::vector<Edge> edges = {
        {1, 2, 1, 100},
        {2, 3, 2, 200},
        {3, 4, 1, 300},
        {4, 5, 2, 400},
        {5, 1, 1, 500},
    };
    Batch batch(edges.begin(), edges.end());
    this->args.batch_size = edges.size();
    this->impl.insert_batch(batch);

    // Run all supported algs
    for (auto alg_name : this->args.alg_names)
    {
        this->impl.update_alg(alg_name, {1});
    }
};

// Insert an edge and make sure num_edges == 1
TYPED_TEST_P(ImplTest, SingleEdgeInsert)
{
    // Create a batch with one edge
    std::vector<Edge> edges = {
        {2, 3, 0, 0}
    };
    Batch batch(edges.begin(), edges.end());
    this->args.batch_size = edges.size();

    // Ask the impl to insert the edge
    this->impl.insert_batch(batch);

    // Make sure it reports one edge back
    EXPECT_EQ(this->impl.get_num_edges(), 1);
}

// Insert several duplicates of the same edge and make sure num_edges == 1
TYPED_TEST_P(ImplTest, DuplicateEdgeInsert)
{
    // Create a batch with some duplicate edges
    std::vector<Edge> edges = {
        {2, 3, 0, 0},
        {2, 3, 0, 0},
        {2, 3, 0, 0},
    };
    Batch batch(edges.begin(), edges.end());
    this->args.batch_size = edges.size();

    // Ask the impl to insert the edges
    this->impl.insert_batch(batch);

    // Make sure it reports the right number of edges back
    EXPECT_EQ(this->impl.get_num_edges(), 1);

    // Insert the duplicates again, just to make sure the deduplication happens no matter what
    this->impl.insert_batch(batch);
    EXPECT_EQ(this->impl.get_num_edges(), 1);

}

// Insert a bidirectional edge between two nodes and make sure it is counted as two edges
TYPED_TEST_P(ImplTest, BidirectionalEdgeInsert)
{
    // Create two edges going both directions
    std::vector<Edge> edges = {
        {2, 3, 0, 0},
        {3, 2, 0, 0},
    };
    Batch batch(edges.begin(), edges.end());
    this->args.batch_size = edges.size();

    // Ask the impl to insert the batch
    this->impl.insert_batch(batch);

    // Make sure it reports the right number of edges back
    EXPECT_EQ(this->impl.get_num_edges(), 2);
}

// Make sure we get the right value back from get_out_degree
TYPED_TEST_P(ImplTest, GetOutDegree)
{
    // Create several edges emanating from the same source vertex
    std::vector<Edge> edges = {
        {1, 3, 0, 0},
        {1, 4, 0, 0},
        {1, 5, 0, 0},
    };
    Batch batch(edges.begin(), edges.end());
    this->args.batch_size = edges.size();

    // Ask the impl to insert the edge
    this->impl.insert_batch(batch);

    // Make sure it reports the right number of edges back
    EXPECT_EQ(this->impl.get_out_degree(1), 3);
}

// Make sure the graph can actually delete edges
TYPED_TEST_P(ImplTest, DeleteOlderThan)
{
    // Create several edges with distinct timestamps
    std::vector<Edge> edges = {
        {1, 3, 0, 100},
        {1, 4, 0, 200},
        {1, 5, 0, 300},
    };
    Batch batch(edges.begin(), edges.end());
    this->args.batch_size = edges.size();
    this->impl.insert_batch(batch);
    EXPECT_EQ(this->impl.get_num_edges(), 3);
    EXPECT_EQ(this->impl.get_out_degree(1), 3);

    // Should delete 1->3, leaving two edges
    this->impl.delete_edges_older_than(200);
    EXPECT_EQ(this->impl.get_num_edges(), 2);
    EXPECT_EQ(this->impl.get_out_degree(1), 2);

    // Shouldn't delete anything
    this->impl.delete_edges_older_than(0);
    EXPECT_EQ(this->impl.get_num_edges(), 2);
    EXPECT_EQ(this->impl.get_out_degree(1), 2);

    // Should delete 1->4 and 1->5, leaving the graph empty
    this->impl.delete_edges_older_than(400);
    EXPECT_EQ(this->impl.get_num_edges(), 0);
    EXPECT_EQ(this->impl.get_out_degree(1), 0);
}

// Make sure timestamps are updated on insert
TYPED_TEST_P(ImplTest, TimestampUpdate)
{
    // Create several edges with distinct timestamps
    std::vector<Edge> edges1 = {
        {1, 3, 0, 100},
        {1, 4, 0, 200},
        {1, 5, 0, 300},
    };
    Batch batch1(edges1.begin(), edges1.end());
    this->args.batch_size = edges1.size();
    this->impl.insert_batch(batch1);
    EXPECT_EQ(this->impl.get_num_edges(), 3);
    EXPECT_EQ(this->impl.get_out_degree(1), 3);

    // Insert identical edges with updated timestamps
    std::vector<Edge> edges2 = {
        {1, 3, 0, 400},
        {1, 4, 0, 500},
        {1, 5, 0, 600},
    };
    Batch batch2(edges2.begin(), edges2.end());
    this->impl.insert_batch(batch2);

    // Should delete 1->3, leaving two edges
    this->impl.delete_edges_older_than(500);
    EXPECT_EQ(this->impl.get_num_edges(), 2);
    EXPECT_EQ(this->impl.get_out_degree(1), 2);
}

// All tests in this file must be registered here
REGISTER_TYPED_TEST_CASE_P( ImplTest
    ,CheckAlgs
    ,SingleEdgeInsert
    ,DuplicateEdgeInsert
    ,BidirectionalEdgeInsert
    ,GetOutDegree
    ,DeleteOlderThan
    ,TimestampUpdate
);

// To instantiate this test, create a source file that includes this header and this line:
//      INSTANTIATE_TYPED_TEST_CASE_P(TEST_NAME_HERE, ImplTest, ClassThatImplementsDynamicGraph);
// Then link with gtest_main
