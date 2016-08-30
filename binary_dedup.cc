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

    // Allocate space for the deduplicated edge list
    std::vector<Edge> deduped_edges;
    deduped_edges.reserve(edges.size());

    // Deduplicate the edge list
    // Using std::unique_copy since there is no parallel version of std::unique
    std::unique_copy(edges.begin(), edges.end(), std::back_inserter(deduped_edges),
        // We consider only source and dest when searching for duplicates
        // The input is sorted, so we'll only get the most recent timestamp
        // BUG: Does not combine weights
        [](const Edge& a, const Edge& b) { return a.src == b.src && a.dst == b.dst; });

    // Write to stdout
    fwrite(&deduped_edges[0], sizeof(Edge), deduped_edges.size(), stdout);

    return 0;
}