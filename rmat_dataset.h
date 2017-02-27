//
// Created by ehein6 on 2/27/17.
//

#ifndef STINGER_DYNOGRAPH_RMAT_DATASET_H
#define STINGER_DYNOGRAPH_RMAT_DATASET_H

#include "dynograph_util.h"
#include "rmat.h"

namespace DynoGraph {

struct RmatArgs
{
    double a, b, c, d;
    int64_t num_edges, num_vertices;

    static RmatArgs from_string(std::string str);
    std::string validate() const;
};

class RmatBatch : public Batch
{
protected:
    std::vector<Edge> edges;
public:
    explicit RmatBatch(rmat_edge_generator& generator, int64_t size, int64_t first_timestamp);
};

class RmatDataset : public IDataset {
private:
    Args args;
    RmatArgs rmat_args;
    int64_t current_batch;
    int64_t num_edges;
    int64_t num_batches;
    int64_t num_vertices;
    int64_t next_timestamp;
    DynoGraph::rmat_edge_generator generator;
public:
    RmatDataset(Args args, RmatArgs rmat_args);

    int64_t getTimestampForWindow(int64_t batchId) const;
    std::shared_ptr<Batch> getBatch(int64_t batchId);
    int64_t getNumBatches() const;
    int64_t getNumEdges() const;

    bool isDirected() const;
    int64_t getMaxVertexId() const;

    bool enableAlgsForBatch(int64_t i) const;
    void reset();
};

} // end namespace DynoGraph

#endif //STINGER_DYNOGRAPH_RMAT_DATASET_H
