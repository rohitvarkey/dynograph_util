/**
 * dynograph_edge_count.h
 *
 * Defines macros for counting the number of edges traversed by each thread in a graph processing benchmark.
 * For each file that does graph processing,
 * 1. Include this header ( and add this path to includes )
 * 2. Add DYNOGRAPH_EDGE_COUNT_TRAVERSE_EDGE whenever an edge is traversed
 */

#ifndef DYNOGRAPH_EDGE_COUNT_H
#define DYNOGRAPH_EDGE_COUNT_H

#ifdef __cplusplus
extern "C" {
#endif

// Stores edge counts for each thread
// Do not access directly, instead use macros below
#include <inttypes.h>
extern uint64_t* dynograph_edge_count_num_traversed_edges;

#if defined(_OPENMP)
#include <omp.h>
#define DYNOGRAPH_EDGE_COUNT_THREAD_ID omp_get_thread_num()
#define DYNOGRAPH_EDGE_COUNT_THREAD_COUNT omp_get_max_threads()
#else // single-threaded
#define DYNOGRAPH_EDGE_COUNT_THREAD_ID 0
#define DYNOGRAPH_EDGE_COUNT_THREAD_COUNT 1
#endif

#ifndef ENABLE_DYNOGRAPH_EDGE_COUNT

// Define empty macros when disabled for zero overhead
#define DYNOGRAPH_EDGE_COUNT_TRAVERSE_EDGE()
#define DYNOGRAPH_EDGE_COUNT_TRAVERSE_MULTIPLE_EDGES(X)

#else // defined(ENABLE_DYNOGRAPH_EDGE_COUNT)

// Call whenever this thread traverses an edge
#define DYNOGRAPH_EDGE_COUNT_TRAVERSE_EDGE() \
do {                                                                                \
    dynograph_edge_count_num_traversed_edges[DYNOGRAPH_EDGE_COUNT_THREAD_ID]+=1;    \
} while(0)

// Use where possible to avoid putting TRAVERSE_EDGE in a tight loop
#define DYNOGRAPH_EDGE_COUNT_TRAVERSE_MULTIPLE_EDGES(X) \
do {                                                                                \
    dynograph_edge_count_num_traversed_edges[DYNOGRAPH_EDGE_COUNT_THREAD_ID]+=X;    \
} while(0)

#endif // ENABLE_DYNOGRAPH_EDGE_COUNT
#ifdef __cplusplus
}
#endif
#endif // DYNOGRAPH_EDGE_COUNT_H
