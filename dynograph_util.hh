#pragma once

#include <inttypes.h>
#include <vector>
#include <string>
#include <random>

namespace DynoGraph {

const std::string msg = "[DynoGraph] ";

struct Edge
{
    int64_t src;
    int64_t dst;
    int64_t weight;
    int64_t timestamp;
};

bool operator<(const Edge& a, const Edge& b);
bool operator==(const Edge& a, const Edge& b);

class Batch
{
private:
    typedef std::vector<Edge>::iterator iterator;
    iterator begin_iter, end_iter;
public:
    iterator begin();
    iterator end();
    Batch(iterator begin, iterator end);
};

class VertexPicker
{
public:
    VertexPicker(int64_t nv, int64_t seed);
    int64_t next();
    void reset();
private:
    int64_t seed;
    std::uniform_int_distribution<int64_t> distribution;
    // Use a 64-bit Mersene Twister for random number generation
    typedef std::mt19937_64 random_number_generator;
    random_number_generator generator;
};

class Dataset
{
private:
    void loadEdgesBinary(std::string path);
    void loadEdgesAscii(std::string path);
    void initBatchIterators();

    int64_t numBatches;
    int64_t directed;
    int64_t maxNumVertices;

public:

    std::vector<Edge> edges;
    std::vector<Batch> batches;

    Dataset(std::string path, int64_t numBatches);
    Dataset(std::vector<Edge> edges, int64_t numBatches);
    int64_t getTimestampForWindow(int64_t batchId, int64_t windowSize);
    Batch getBatch(int64_t batchId);

    int64_t getNumBatches();
    bool isDirected();
    int64_t getMaxNumVertices();

    std::vector<Batch>::iterator begin();
    std::vector<Batch>::iterator end();
};

}