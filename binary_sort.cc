#include <iostream>
#include <vector>
#include <parallel/algorithm>
#include "dynograph_util.hh"

using namespace DynoGraph;

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
    std::sort(edges.begin(), edges.end());

    // Write to stdout
    fwrite(&edges[0], sizeof(Edge), edges.size(), stdout);

    return 0;
}