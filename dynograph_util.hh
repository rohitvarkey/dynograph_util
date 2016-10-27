#pragma once

#include <inttypes.h>
#include <vector>
#include <string>
#include <random>
#include <memory>

namespace DynoGraph {

const std::string msg = "[DynoGraph] ";

struct Args
{
    std::string alg_name;
    std::string input_path;
    enum SORT_MODE {
        // Do not pre-sort batches
        UNSORTED,
        // Sort and deduplicate each batch before returning it
        PRESORT,
        // Each batch is a cumulative snapshot of all edges in previous batches
        SNAPSHOT
    } sort_mode;
    int64_t window_size;
    int64_t num_batches;
    int64_t num_trials;
    int64_t enable_deletions;

    Args(int argc, char **argv);
};

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
protected:
    typedef std::vector<Edge>::iterator iterator;
    iterator begin_iter, end_iter;
public:
    iterator begin();
    iterator end();
    Batch(iterator begin, iterator end);
};

class DeduplicatedBatch : public Batch
{
protected:
    std::vector<Edge> deduped_edges;
public:
    explicit DeduplicatedBatch(Batch& batch);
};

class VertexPicker
{
public:
    VertexPicker(int64_t nv, int64_t seed);
    int64_t next();
    void reset();
private:
    int64_t seed;
    int64_t max_nv;
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

    Args args;
    int64_t directed;
    int64_t maxNumVertices;

public:

    std::vector<Edge> edges;
    std::vector<Batch> batches;

    Dataset(Args args);
    Dataset(std::vector<Edge> edges, Args& args, int64_t maxNumVertices);

    int64_t getTimestampForWindow(int64_t batchId);
    std::shared_ptr<Batch> getBatch(int64_t batchId);

    bool isDirected();
    int64_t getMaxNumVertices();

    std::vector<Batch>::iterator begin();
    std::vector<Batch>::iterator end();
};

}