#pragma once

#include <inttypes.h>
#include <vector>
#include <string>

namespace DynoGraph {

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
    int64_t numBatches;
    int64_t directed;
    std::vector<Edge> edges;
    std::vector<Batch> batches;

    void loadEdgesBinary(std::string path);
    void loadEdgesAscii(std::string path);
public:
    Dataset(std::string path, int64_t num_batches);
    int64_t getTimestampForWindow(int64_t batchId, int64_t windowSize);
    Batch getBatch(int64_t batchId);
    int64_t getNumBatches();
    std::vector<Batch>::iterator begin();
    std::vector<Batch>::iterator end();
};

}