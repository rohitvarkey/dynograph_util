#ifndef REFERENCE_IMPL_H
#define REFERENCE_IMPL_H

#include <dynograph_util.h>
#include <map>
#include <inttypes.h>

/**
 * Reference implementation of the DynoGraph::DynamicGraph interface
*/
class reference_impl : public DynoGraph::DynamicGraph {

protected:
    struct edge_prop {
        int64_t weight;
        int64_t timestamp;
        edge_prop() : weight(-1), timestamp(-1) {};
        edge_prop(int64_t weight, int64_t timestamp) : weight(weight), timestamp(timestamp) {};
    };
    typedef std::map<int64_t, edge_prop> edge_list;
    typedef std::map<int64_t, edge_list> adjacency_list;
    adjacency_list graph;

public:

    reference_impl(DynoGraph::Args args, int64_t max_vertex_id)
    : DynoGraph::DynamicGraph(args, max_vertex_id) {};

    static std::vector<std::string> get_supported_algs() { return {}; };
    // Prepare to insert the batch
    virtual void before_batch(const DynoGraph::Batch& batch, int64_t threshold) {};
    // Delete edges in the graph with a timestamp older than <threshold>
    virtual void delete_edges_older_than(int64_t threshold)
    {
        for (std::pair<const int64_t, edge_list>& vertex : graph)
        {
            edge_list& neighbors = vertex.second;
            for (edge_list::iterator neighbor = neighbors.begin(); neighbor != neighbors.end();)
            {
                int64_t timestamp = neighbor->second.timestamp;
                if (timestamp < threshold) {
                    neighbor = neighbors.erase(neighbor);
                } else {
                    ++neighbor;
                }
            }
        }
    }
    // Insert the batch of edges into the graph
    virtual void insert_batch(const DynoGraph::Batch& batch)
    {
        for (DynoGraph::Edge e : batch)
        {
            // Locate edge in graph (will create if does not exist)
            edge_prop& edge = graph[e.src][e.dst];
            if (edge.weight == -1) {
                // Insert new edge
                edge.weight = e.weight;
                edge.timestamp = e.timestamp;
            } else {
                // Update existing edge
                edge.weight += e.weight;
                edge.timestamp = std::max(edge.timestamp, e.timestamp);
            }
        }
    }
    // Run the specified algorithm
    virtual void update_alg(const std::string &alg_name, const std::vector<int64_t> &sources) {};
    // Return the degree of the specified vertex
    virtual int64_t get_out_degree(int64_t vertex_id) const
    {
        adjacency_list::const_iterator vertex = graph.find(vertex_id);
        return vertex != graph.end() ? static_cast<int64_t>(vertex->second.size()) : 0;
    }
    // Return the number of vertices in the graph
    virtual int64_t get_num_vertices() const
    {
        return static_cast<int64_t>(graph.size());
    }
    // Return the number of unique edges in the graph
    virtual int64_t get_num_edges() const
    {
        int64_t num_edges = 0;
        for (const std::pair<int64_t, edge_list>& vertex : graph)
        {
            const edge_list& neighborhood = vertex.second;
            num_edges += neighborhood.size();
        }
        return num_edges;
    }
};

#endif
