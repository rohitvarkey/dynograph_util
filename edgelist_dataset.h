#pragma once
#include "dynograph_util.h"

namespace DynoGraph {

class EdgeListDataset : public IDataset
{
private:
    void loadEdgesBinary(std::string path);
    void loadEdgesAscii(std::string path);

    Args args;
    bool directed;
    int64_t max_vertex_id;
    int64_t min_timestamp;
    int64_t max_timestamp;

    std::vector<Edge> edges;
    std::vector<Batch> batches;

public:
    EdgeListDataset(Args args);

    int64_t getTimestampForWindow(int64_t batchId) const;
    std::shared_ptr<Batch> getBatch(int64_t batchId);
    std::shared_ptr<Batch> getBatchesUpTo(int64_t batchId);
    int64_t getNumBatches() const;
    int64_t getNumEdges() const;

    bool isDirected() const;
    int64_t getMaxVertexId() const;

    bool enableAlgsForBatch(int64_t i) const;
};

} // end namespace DynoGraph
