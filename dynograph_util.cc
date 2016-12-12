#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <algorithm>

#if defined(USE_MPI)
#include <mpi.h>
#include <valarray>

#endif

#include "dynograph_util.h"

using namespace DynoGraph;
using std::cerr;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::stringstream;

// Helper functions to split strings
// http://stackoverflow.com/a/236803/1877086
void split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
}
vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

Args::Args(int argc, char **argv)
{
    if (argc != 7)
    {
        cerr << msg << "Usage: alg_name sort_mode input_path num_batches window_size num_trials \n";
        exit(-1);
    }

    std::string alg_str = argv[1];
    alg_names = split(alg_str, ' ');
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

    enable_deletions = (window_size != num_batches);

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

Batch::Batch(iterator begin, iterator end, const Dataset &dataset )
 : begin_iter(begin), end_iter(end), dataset(dataset) {}

Batch::iterator
Batch::begin() const { return begin_iter; }

Batch::iterator
Batch::end() const { return end_iter; }

int64_t Batch::num_vertices_affected() const
{
    // We need to sort and deduplicate anyways, just use the implementation in DeduplicatedBatch
    auto sorted = DeduplicatedBatch(*this);
    return sorted.num_vertices_affected();
}

// Implementation of DynoGraph::DeduplicatedBatch

DeduplicatedBatch::DeduplicatedBatch(const Batch &batch)
: Batch(batch), deduped_edges(std::distance(batch.begin(), batch.end())) {
    // Make a copy of the original batch
    std::vector<Edge> sorted_edges(batch.begin(), batch.end()); // TODO is this init done in parallel?
    // Sort the edge list
    std::sort(sorted_edges.begin(), sorted_edges.end(), DynoGraph::operator<);

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

int64_t
DeduplicatedBatch::num_vertices_affected() const
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
    int64_t edges_per_batch = edges.size() / args.num_batches;

    // Store iterators to the beginning and end of each batch
    for (int i = 0; i < args.num_batches; ++i)
    {
        size_t offset = i * edges_per_batch;
        auto begin = edges.begin() + offset;
        auto end = edges.begin() + offset + edges_per_batch;
        batches.push_back(Batch(begin, end, *this));
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
#if defined(USE_MPI)
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (rank == 0){
#endif
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

    // Could save work by counting max vertex id while loading edges, but easier to just do it here
    maxNumVertices = getMaxVertexId(edges) + 1;

#if defined(USE_MPI)
        cerr << msg << "Distributing dataset to " << comm_size << " ranks...\n";
    }

    // Move the edges out of rank 0's member variable
    vector<Edge> all_edges = std::move(edges);
    edges.clear();

    // Send max_nv to all ranks
    MPI_Bcast(&maxNumVertices, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    /*
     * Now we need to distribute the edges to each process in the group
     * Each process will have the same number of batches, but each batch
     * will be a slice of the corresponding batch in the parent
     */
    int64_t edges_per_batch = all_edges.size() / args.num_batches;
    MPI_Bcast(&edges_per_batch, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    vector<int> bytes_per_rank(comm_size);
    std::valarray<int> displacements(comm_size+1);
    if (rank == 0)
    {
        // Divide the batch evenly among the ranks
        int edges_per_rank = std::floor(edges_per_batch / comm_size);
        for (int i = 0; i < comm_size; ++i)
        {
            // Assign a portion of edges to this rank
            int num_edges = edges_per_rank;
            // Distribute remainder evenly
            if (i < edges_per_batch % comm_size) { num_edges += 1; }
            // Calculate how big this chunk is and where it starts in the buffer
            bytes_per_rank[i] = num_edges * sizeof(Edge);
        }
        std::partial_sum(bytes_per_rank.begin(), bytes_per_rank.end(), std::begin(displacements));
        // Prefix sum gives the end of each buffer, shift to get a list of beginnings
        displacements = displacements.shift(-1);
    }

    // Send the size of each slice to the ranks
    int local_num_bytes;
    MPI_Scatter(
        bytes_per_rank.data(), 1, MPI_INT,
        &local_num_bytes, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // Allocate local storage for the edges
    edges.resize((local_num_bytes / sizeof(Edge)) * args.num_batches);
    for (int batchId = 0; batchId < args.num_batches; ++batchId)
    {
        // Scatter the edges in this batch to the ranks
        MPI_Scatterv(
            all_edges.data(), bytes_per_rank.data(), &displacements[0], MPI_BYTE,
            edges.data(), local_num_bytes, MPI_BYTE,
            0, MPI_COMM_WORLD
        );

        // Move to the next batch
        displacements += edges_per_batch * sizeof(Edge);
    }
#endif
    
    initBatchIterators();
}


#ifdef USE_MPI

// Register a type with MPI to hold a tuple of (vertex_id, degree)
void
DynoGraph::register_vertex_degree_type(MPI_Datatype *type)
{
    MPI_Datatype vertex_degree_type;
    MPI_Type_contiguous(2, MPI_INT64_T, &vertex_degree_type);
//    int blocks[] = {1, 1};
//    MPI_Aint displacements[] = {offsetof(vertex_degree, first), offsetof(vertex_degree, second)};
//    MPI_Datatype types[] = {MPI_INT64_T, MPI_INT64_T};
//    MPI_Type_create_struct(2, blocks, displacements, types, &vertex_degree_type);
    MPI_Type_commit(&vertex_degree_type);
}

// Reduce to get the vertex id with the highest degree across all processes
// (if there is a tie, pick the smallest vertex id)
// MPI requires this awkward signature: a and b are arrays of length len which should be reduced into b
void
DynoGraph::vertex_degree_reducer(vertex_degree *a, vertex_degree *b, int *len, MPI_Datatype *datatype) {
    for (int i = 0; i < *len; ++i) {
        int64_t vid_a = a[i].first;
        int64_t vid_b = b[i].first;
        int64_t degree_a = a[i].second;
        int64_t degree_b = b[i].second;
        vertex_degree &out = b[i];
        if (degree_a != degree_b) { out = degree_a > degree_b ? a[i] : b[i]; }
        else { out = vid_a < vid_b ? a[i] : b[i]; }
    }
}
#endif

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

int64_t
Dataset::getTimestampForWindow(int64_t batchId) const
{
    int64_t modifiedAfter = INT64_MIN;
    if (batchId > args.window_size)
    {
        // Intentionally rounding down here
        // TODO variable number of edges per batch
        int64_t edges_per_batch = edges.size() / batches.size();
        int64_t startEdge = (batchId - args.window_size) * edges_per_batch;
        modifiedAfter = edges[startEdge].timestamp;
    }
    return modifiedAfter;
};

shared_ptr<Batch>
Dataset::getBatch(int64_t batchId) const
{
    const Batch & b = batches[batchId];
    switch (args.sort_mode)
    {
        case Args::UNSORTED:
        {
            return make_shared<Batch>(b);
        }
        case Args::PRESORT:
        {
            cerr << msg << "Presorting batch " << batchId << "...\n";
            return make_shared<DeduplicatedBatch>(b);
        }
        case Args::SNAPSHOT:
        {
            cerr << msg << "Generating snapshot for batch " << batchId << "...\n";
            int64_t threshold = getTimestampForWindow(batchId);
            auto start = std::find_if(edges.begin(), b.end(),
                [threshold](const Edge& e){ return e.timestamp >= threshold; });
            Batch filtered(start, b.end(), *this);
            return make_shared<DeduplicatedBatch>(filtered);
        }
        default: assert(0);
    }
}

bool
Dataset::isDirected() const
{
    return directed;
}

int64_t
Dataset::getMaxNumVertices() const
{
    return maxNumVertices;
}

std::vector<Batch>::const_iterator
Dataset::begin() const { return batches.cbegin(); }
std::vector<Batch>::const_iterator
Dataset::end() const { return batches.cend(); }
