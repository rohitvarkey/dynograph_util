#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include "dynograph_util.hh"

using namespace DynoGraph;
using std::cerr;
using std::string;

bool DynoGraph::operator<(const Edge& a, const Edge& b)
{
    // Sort by src, dst, timestamp, weight
    return (a.src != b.src) ? a.src < b.src
         : (a.dst != b.dst) ? a.dst < b.dst
         : (a.timestamp != b.timestamp) ? a.timestamp < b.timestamp
         : (a.weight != b.weight) ? a.weight < b.weight
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
    int64_t edgesPerBatch = edges.size() / numBatches;

    // Store iterators to the beginning and end of each batch
    for (int i = 0; i < numBatches; ++i)
    {
        size_t offset = i * edgesPerBatch;
        auto begin = edges.begin() + offset;
        auto end = edges.begin() + offset + edgesPerBatch;
        batches.push_back(Batch(begin, end));
    }
}

Dataset::Dataset(std::vector<Edge> edges, int64_t numBatches)
: numBatches(numBatches), directed(true), edges(edges)
{
    // Sanity check
    if (numBatches < 1)
    {
        cerr << msg << "Need at least one batch\n";
        exit(-1);
    }

    initBatchIterators();
}


Dataset::Dataset(string path, int64_t numBatches)
: numBatches(numBatches), directed(true)
{
    // Sanity check
    if (numBatches < 1)
    {
        cerr << msg << "Need at least one batch\n";
        exit(-1);
    }

    // Load edges from the file
    if (has_suffix(path, ".graph.bin"))
    {
        loadEdgesBinary(path);
    } else if (has_suffix(path, ".graph.el")) {
        loadEdgesAscii(path);
    } else {
        cerr << msg << "Unrecognized file extension for " << path << "\n";
        exit(-1);
    }

    initBatchIterators();
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
    if (rc != numEdges)
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
Dataset::getTimestampForWindow(int64_t batchId, int64_t windowSize)
{
    int64_t modifiedAfter = INT64_MIN;
    if (batchId > windowSize)
    {
        // Intentionally rounding down here
        // TODO variable number of edges per batch
        int64_t edgesPerBatch = edges.size() / batches.size();
        int64_t startEdge = (batchId - windowSize) * edgesPerBatch;
        modifiedAfter = edges[startEdge].timestamp;
    }
    return modifiedAfter;
};

Batch
Dataset::getBatch(int64_t batchId)
{
    return batches[batchId];
}

int64_t
Dataset::getNumBatches()
{
    return batches.size();
}

std::vector<Batch>::iterator
Dataset::begin() { return batches.begin(); }
std::vector<Batch>::iterator
Dataset::end() { return batches.end(); }
