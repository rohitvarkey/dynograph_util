#pragma once

#include <inttypes.h>
#include <vector>
#include <string>

namespace DynoGraph {

const std::string msg = "[DynoGraph] ";

struct Edge
{
    int64_t src;
    int64_t dst;
    int64_t weight;
    int64_t timestamp;
};

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

class Dataset
{
private:
    void loadEdgesBinary(std::string path);
    void loadEdgesAscii(std::string path);
    void initBatchIterators();
public:
    const int64_t numBatches;
    const int64_t directed;
    std::vector<Edge> edges;
    std::vector<Batch> batches;

    Dataset(std::string path, int64_t numBatches);
    Dataset(std::vector<Edge> edges, int64_t numBatches);
    int64_t getTimestampForWindow(int64_t batchId, int64_t windowSize);
    Batch getBatch(int64_t batchId);
    int64_t getNumBatches();
    std::vector<Batch>::iterator begin();
    std::vector<Batch>::iterator end();
};

}