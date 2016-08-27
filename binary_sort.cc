#include <iostream>
#include <vector>
#include <parallel/algorithm>
#include "dynograph_util.hh"

using namespace DynoGraph;

bool compare(const Edge& a, const Edge& b)
{
    // Custom sorting order to prepare for deduplication
    // Order by src ascending, then dest ascending, then timestamp descending
    // This way the edge with the most recent timestamp will be picked when deduplicating
    return (a.src != b.src) ? a.src < b.src
         : (a.dst != b.dst) ? a.dst < b.dst
         : (a.timestamp != b.timestamp) ? a.timestamp > b.timestamp
         : false;
}

int main(int argc, char *argv[])
{
    // Using C-style IO, so no need to keep cout/cerr in sync
    std::ios_base::sync_with_stdio(false);

    // Reserve space for the edge list
    // We don't know how much space we'll need, so just guess and resize as necessary
    std::vector<Edge> edges;
    edges.reserve(1024 * 1024);
    Edge e;

    // Keep reading in edges until we run out
    while (!feof(stdin))
    {
        size_t rc = fread(&e, sizeof(Edge), 1, stdin);
        if (rc) { edges.push_back(e); }
    }

    // Sort the edge list
    std::sort(edges.begin(), edges.end(), compare);

    // Write to stdout
    fwrite(&edges[0], sizeof(Edge), edges.size(), stdout);

    return 0;
}