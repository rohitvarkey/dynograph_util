#pragma once

#include <inttypes.h>
#include <vector>
#include <string>
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

// Forward reference
class Dataset;

class Batch
{
protected:
    typedef std::vector<Edge>::iterator iterator;
    iterator begin_iter, end_iter;
public:
    iterator begin();
    iterator end();
    Dataset& dataset;
    Batch(iterator begin, iterator end, Dataset &dataset);
    virtual int64_t num_vertices_affected();
};

class DeduplicatedBatch : public Batch
{
protected:
    std::vector<Edge> deduped_edges;
public:
    explicit DeduplicatedBatch(Batch& batch);
    virtual int64_t num_vertices_affected();
};

/**
 * Returns a list of the highest N vertices in the graph
 * @param top_n Number of vertex ID's to return
 * @param nv Total number of vertices in the graph
 * @param get_degree Function which returns the degree of the specified vertex
 *        int64_t get_degree(int64_t vertex_id);
 * @return list of the top_n vertices with highest degree
 */
template <typename degree_getter>
std::vector<int64_t>
find_high_degree_vertices(int64_t top_n, int64_t nv, degree_getter get_degree)
{
    typedef std::pair<int64_t, int64_t> vertex_degree;
    std::vector<vertex_degree> degrees(nv);
    #pragma omp parallel for
    for (int i = 0; i < nv; ++i) {
        degrees[i] = std::make_pair(i, get_degree(i));
    }

    // order by degree descending, vertex_id ascending
    std::sort(degrees.begin(), degrees.end(),
        [](const vertex_degree &a, const vertex_degree &b) {
            if (a.second != b.second) { return a.second > b.second; }
            return a.first < b.first;
        }
    );

    degrees.erase(degrees.begin() + top_n, degrees.end());
    std::vector<int64_t> ids(degrees.size());
    std::transform(degrees.begin(), degrees.end(), ids.begin(),
        [](const vertex_degree &d) { return d.first; });
    return ids;
}

class Dataset
{
private:
    void loadEdgesBinary(std::string path);
    void loadEdgesAscii(std::string path);
    void initBatchIterators();

    Args args;
    bool directed;
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