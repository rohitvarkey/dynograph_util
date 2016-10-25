#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <assert.h>
#include <parallel/algorithm>

#include "dynograph_util.hh"

using namespace DynoGraph;
using std::cerr;
using std::string;
using std::vector;
using std::unique_ptr;

/*
 * HACK
 *   This must be set to larger than the largest vertex ID that DynoGraph will ever see
 * Rationale:
 *   For a given graph, VertexPicker should always return the same sequence of vertex ID's.
 *   This includes "static" experiments, where dynograph_util loads a pre-processed subset of the graph
 *   and would see a different max_nv than the dynamic version.
 *
 *   Long term, dynograph_util should always load the entire dataset and produce "static" versions at runtime.
 *   For now, we will use an oversized distribution range and iterate until we get a valid vertex ID.
 */

int64_t
get_vertex_picker_range_max()
{
    static int64_t rv = 0;
    if (rv != 0) return rv;

    if (char * s = getenv("VERTEX_PICKER_RANGE_MAX"))
    {
        rv = atoll(s);
    } else {
        rv = 1LL << 30;
        cerr << msg << "WARNING: VertexPicker max range unset, set VERTEX_PICKER_RANGE_MAX\n";
        cerr << msg << "Defaulting to " << rv << "\n";
    }
    return rv;
}

Args::Args(int argc, char **argv)
{
    if (argc != 7)
    {
        cerr << msg << "Usage: alg_name sort_mode input_path num_batches window_size num_trials \n";
        exit(-1);
    }

    alg_name = argv[1];
    std::string sort_mode_str = argv[2];
    input_path = argv[3];
    num_batches = atoll(argv[4]);
    window_size = atoll(argv[5]);
    num_trials = atoll(argv[6]);

    if (num_batches < 1 || window_size < 1 || num_trials < 1)
    {
        cerr << msg << "num_batches, window_size, and num_trials must be positive\n";
        exit(-1);
    }

    if      (sort_mode_str == "unsorted") { sort_mode = UNSORTED; }
    else if (sort_mode_str == "presort")  { sort_mode = PRESORT;  }
    else if (sort_mode_str == "snapshot") { sort_mode = SNAPSHOT; }
    else {
        cerr << msg << "sort_mode must be one of ['unsorted', 'presort', 'snapshot']\n";
        exit(-1);
    }

}

bool DynoGraph::operator<(const Edge& a, const Edge& b)
{
    // Custom sorting order to prepare for deduplication
    // Order by src ascending, then dest ascending, then timestamp descending
    // This way the edge with the most recent timestamp will be picked when deduplicating
    return (a.src != b.src) ? a.src < b.src
         : (a.dst != b.dst) ? a.dst < b.dst
         : (a.timestamp != b.timestamp) ? a.timestamp > b.timestamp
         : false;
}

bool DynoGraph::operator==(const Edge& a, const Edge& b)
{
    return a.src == b.src
        && a.dst == b.dst
        && a.weight == b.weight
        && a.timestamp == b.timestamp;
}

// Count the number of lines in a text file
int64_t
count_lines(string path)
{
    FILE* fp = fopen(path.c_str(), "r");
    if (fp == NULL)
    {
        cerr << msg << "Failed to open " << path << "\n";
        exit(-1);
    }
    int64_t lines = 0;
    while(!feof(fp))
    {
        int ch = fgetc(fp);
        if(ch == '\n')
        {
            lines++;
        }
    }
    fclose(fp);
    return lines;
}

// Implementation of DynoGraph::Batch

Batch::Batch(iterator begin, iterator end)
 : begin_iter(begin), end_iter(end) {}

Batch::iterator
Batch::begin() { return begin_iter; }

Batch::iterator
Batch::end() { return end_iter; }

// Implementation of DynoGraph::DeduplicatedBatch

DeduplicatedBatch::DeduplicatedBatch(Batch &batch)
: Batch(batch), deduped_edges(std::distance(batch.begin(), batch.end())) {
    // Make a copy of the original batch
    std::vector<Edge> sorted_edges(batch.begin(), batch.end()); // TODO is this init done in parallel?
    // Sort the edge list
    std::sort(sorted_edges.begin(), sorted_edges.end(), DynoGraph::operator<);

    // Deduplicate the edge list
    // Using std::unique_copy since there is no parallel version of std::unique
    Batch::iterator end = std::unique_copy(sorted_edges.begin(), sorted_edges.end(), deduped_edges.begin(),
            // We consider only source and dest when searching for duplicates
            // The input is sorted, so we'll only get the most recent timestamp
            // BUG: Does not combine weights
            [](const Edge& a, const Edge& b) { return a.src == b.src && a.dst == b.dst; });
    deduped_edges.erase(end, deduped_edges.end());

    // Reinitialize the batch pointers
    begin_iter = deduped_edges.begin();
    end_iter = deduped_edges.end();
}

// Implementation of DynoGraph::Dataset

// Helper function to test a string for a given suffix
// http://stackoverflow.com/questions/20446201
bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void Dataset::initBatchIterators()
{
    // Intentionally rounding down here
    // TODO variable number of edges per batch
    int64_t edgesPerBatch = edges.size() / args.num_batches;

    // Store iterators to the beginning and end of each batch
    for (int i = 0; i < args.num_batches; ++i)
    {
        size_t offset = i * edgesPerBatch;
        auto begin = edges.begin() + offset;
        auto end = edges.begin() + offset + edgesPerBatch;
        batches.push_back(Batch(begin, end));
    }
}

int64_t getMaxVertexId(std::vector<Edge> &edges)
{
    int64_t max_nv = 0;
    #pragma omp parallel for reduction (max : max_nv)
    for (size_t i = 0; i < edges.size(); ++i)
    {
        Edge &e = edges[i];
        if (e.src > max_nv) { max_nv = e.src; }
        if (e.dst > max_nv) { max_nv = e.dst; }
    }
    assert(max_nv < get_vertex_picker_range_max());
    return max_nv;
}

// This version is only used by Boost right now, to create a smaller dataset from a bigger one
Dataset::Dataset(std::vector<Edge> edges, Args& args, int64_t maxNumVertices)
: args(args), directed(true), maxNumVertices(maxNumVertices), edges(edges)
{
    initBatchIterators();
}

Dataset::Dataset(Args args)
: args(args), directed(true)
{
    // Load edges from the file
    if (has_suffix(args.input_path, ".graph.bin"))
    {
        loadEdgesBinary(args.input_path);
    } else if (has_suffix(args.input_path, ".graph.el")) {
        loadEdgesAscii(args.input_path);
    } else {
        cerr << msg << "Unrecognized file extension for " << args.input_path << "\n";
        exit(-1);
    }

    initBatchIterators();
    // Could save work by counting max vertex id while loading edges, but easier to just do it here
    maxNumVertices = getMaxVertexId(edges) + 1;
}

void
Dataset::loadEdgesBinary(string path)
{
    cerr << msg << "Checking file size of " << path << "...\n";
    FILE* fp = fopen(path.c_str(), "rb");
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
    {
        cerr << msg << "Failed to stat " << path << "\n";
        exit(-1);
    }
    int64_t numEdges = st.st_size / sizeof(Edge);

    string directedStr = directed ? "directed" : "undirected";
    cerr << msg << "Preloading " << numEdges << " "
         << directedStr
         << " edges from " << path << "...\n";

    edges.resize(numEdges);

    size_t rc = fread(&edges[0], sizeof(Edge), numEdges, fp);
    if (rc != static_cast<size_t>(numEdges))
    {
        cerr << msg << "Failed to load graph from " << path << "\n";
        exit(-1);
    }
    fclose(fp);
}

void
Dataset::loadEdgesAscii(string path)
{
    cerr << msg << "Counting lines in " << path << "...\n";
    int64_t numEdges = count_lines(path);

    string directedStr = directed ? "directed" : "undirected";
    cerr << msg << "Preloading " << numEdges << " "
         << directedStr
         << " edges from " << path << "...\n";

    edges.resize(numEdges);

    FILE* fp = fopen(path.c_str(), "r");
    int rc = 0;
    for (Edge* e = &edges[0]; rc != EOF; ++e)
    {
        rc = fscanf(fp, "%ld %ld %ld %ld\n", &e->src, &e->dst, &e->weight, &e->timestamp);
    }
    fclose(fp);
}

VertexPicker::VertexPicker(int64_t nv, int64_t seed)
: seed(seed), max_nv(nv), distribution(0, get_vertex_picker_range_max()), generator(seed) {}

int64_t
VertexPicker::next() {
    int64_t value;
    do { value = distribution(generator); }
    while (value >= max_nv);

#ifndef NDEBUG
    cerr << msg << "picking vertex " << value
         << " from range [" << distribution.a() << "," << distribution.b() << "]\n";
#endif
    return value;
}

void
VertexPicker::reset() { generator.seed(seed); }

int64_t
Dataset::getTimestampForWindow(int64_t batchId)
{
    int64_t modifiedAfter = INT64_MIN;
    if (batchId > args.window_size)
    {
        // Intentionally rounding down here
        // TODO variable number of edges per batch
        int64_t edgesPerBatch = edges.size() / batches.size();
        int64_t startEdge = (batchId - args.window_size) * edgesPerBatch;
        modifiedAfter = edges[startEdge].timestamp;
    }
    return modifiedAfter;
};


template<typename T, typename ...Args>
std::unique_ptr<T> make_unique( Args&& ...args )
{
    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}

unique_ptr<Batch>
Dataset::getBatch(int64_t batchId)
{
    Batch & b = batches[batchId];
    switch (args.sort_mode)
    {
        case Args::UNSORTED:
        {
            return make_unique<Batch>(b);
        }
        case Args::PRESORT:
        {
            return make_unique<DeduplicatedBatch>(b);
        }
        case Args::SNAPSHOT:
        {
            int64_t threshold = getTimestampForWindow(batchId);
            auto start = std::find_if(edges.begin(), b.end(),
                [threshold](const Edge& e){ return e.timestamp >= threshold; });
            Batch filtered(start, b.end());
            return make_unique<DeduplicatedBatch>(filtered);
        }
        default: assert(0);
    }
}

bool
Dataset::isDirected()
{
    return directed;
}

int64_t
Dataset::getMaxNumVertices()
{
    return maxNumVertices;
}

std::vector<Batch>::iterator
Dataset::begin() { return batches.begin(); }
std::vector<Batch>::iterator
Dataset::end() { return batches.end(); }
