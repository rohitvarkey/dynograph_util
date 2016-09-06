#pragma once

#include <inttypes.h>
#include <vector>
#include <string>
#include <memory>

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

class Dataset
{
private:
    void loadEdgesBinary(std::string path);
    void loadEdgesAscii(std::string path);
    void initBatchIterators();

    int64_t numBatches;
    int64_t directed;
    int64_t maxNumVertices;

    class RandomStream;
    std::shared_ptr<RandomStream> vertexPicker;
public:

    std::vector<Edge> edges;
    std::vector<Batch> batches;

    Dataset(std::string path, int64_t numBatches);
    Dataset(std::vector<Edge> edges, int64_t numBatches);
    int64_t getTimestampForWindow(int64_t batchId, int64_t windowSize);
    int64_t getRandomVertex();
    Batch getBatch(int64_t batchId);

    int64_t getNumBatches();
    bool isDirected();
    int64_t getMaxNumVertices();

    std::vector<Batch>::iterator begin();
    std::vector<Batch>::iterator end();
};

}