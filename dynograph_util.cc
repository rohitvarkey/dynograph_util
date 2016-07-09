#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include "dynograph_util.hh"

using namespace DynoGraph;
using std::cerr;
using std::string;

const string msg = "[DynoGraph] ";

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

Dataset::Dataset(string _path, int64_t _numBatches)
: path(_path), numBatches(_numBatches), directed(true)
{
    // Sanity check
    if (numBatches < 1)
    {
        cerr << msg << "Need at least one batch\n";
        exit(-1);
    }

    bool file_is_binary = false;
    directed = true;
    if (file_is_binary)
    {
        loadEdgesBinary(path);
    } else {
        loadEdgesAscii(path);
    }

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

    edges.reserve(numEdges);

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

    edges.reserve(numEdges);

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
