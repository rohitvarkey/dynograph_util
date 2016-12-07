#include "hooks.h"
#include <chrono>
#include <vector>
#include <valarray>
#include <iostream>
#include <fstream>
#include "json.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(USE_MPI)
#include <mpi.h>
#endif

#if defined(ENABLE_SNIPER_HOOKS)
#include <hooks_base.h>
#elif defined(ENABLE_GEM5_HOOKS)
#include <util/m5/m5op.h>
#elif defined(ENABLE_PIN_HOOKS)
// Nothing to include here
#elif defined(ENABLE_PERF_HOOKS)
#include <perf.h>
#endif

using json = nlohmann::json;
using std::cerr;
using std::string;
using std::vector;

class Hooks::impl
{
    friend class Hooks;
    // Stream for writing out json results
    std::ofstream out;
    // Name of the region we are currently in, or "" outside of a region
    string region_name;
    // Start and end times of the last region
    std::chrono::time_point<std::chrono::steady_clock> t1, t2;
    // Number of edges traversed during the current region (per thread)
    vector<int64_t> num_traversed_edges;
    // Dict of custom attributes that should be printed after every region_end
    json attrs;
    // Dict of custom results that should be printed after the next region_end
    json stats;
#if defined(ENABLE_PERF_HOOKS)
    // Names of perf events to collect this run
    vector<string> perf_event_names;
    // Number of perf events to collect each trial
    int perf_group_size;
    gBenchPerf_event perf_events;
    gBenchPerf_multi perf;
    int trial;
#endif

#if defined(_OPENMP)
    static int get_num_threads() { return omp_get_max_threads(); }
    static int get_thread_id() { return omp_get_thread_num(); }
#else
    static int get_num_threads() { return 1; }
    static int get_thread_id() { return 0; }
#endif

    static string
    get_output_filename()
    {
        if (const char* filename = getenv("HOOKS_FILENAME"))
        {
            return string(filename);
        } else {
            cerr << "WARNING: HOOKS_FILENAME unset, defaulting to stdout.\n";
            return "/dev/stdout";
        }
    }

#if defined(ENABLE_PERF_HOOKS)

    static vector<string>
    get_perf_event_names()
    {
        vector<string> event_names = {"", "--perf-event"};
        if (const char* env_names = getenv("PERF_EVENT_NAMES"))
        {
            char * names = new char[strlen(env_names) + 1];
            strcpy(names, env_names);
            char * p = strtok(names, " ");
            while (p)
            {
                event_names.push_back(string(p));
                p = strtok(NULL, " ");
            }
            delete[] names;

        } else {
            cerr << "WARNING: No perf events found in environment; set PERF_EVENT_NAMES.\n";
        }
        return event_names;
    }

    static int
    get_perf_group_size()
    {
        if (const char* env_group_size = getenv("PERF_GROUP_SIZE"))
        {
            return atoi(env_group_size);
        } else {
            cerr << "WARNING: Perf group size unspecified, defaulting to 4\n";
            return 4;
        }
    }

#endif

    impl()
     : out(get_output_filename(), std::ofstream::app)
     , region_name("")
     , num_traversed_edges(get_num_threads())
#if defined(ENABLE_PERF_HOOKS)
     , perf_event_names(get_perf_event_names())
     , perf_group_size(get_perf_group_size())
     , perf_events(perf_event_names, false)
     , perf(get_num_threads(), perf_events)
#endif
    {

    }

    void __attribute__ ((noinline))
    region_begin(string name)
    {
        // Check for mismatched begin/end pairs
        if (region_name != "") {
            cerr << "ERROR: called region_begin inside region\n";
            exit(-1);
        } else if (name == "") {
            cerr << "ERROR: region name cannot be empty\n";
            exit(-1);
        } else {
            region_name = name;
        }

        // Start the ROI
#if defined(ENABLE_SNIPER_HOOKS)
        parmacs_roi_begin();
#elif defined(ENABLE_GEM5_HOOKS)
        m5_reset_stats(0,0);
#elif defined(ENABLE_PIN_HOOKS)
        __asm__("");
#elif defined(ENABLE_PERF_HOOKS)
        // We can only collect perf_group_size events at a time
        // Collecting more events is done via multiple trials
        trial = attrs.find("trial") != attrs.end() ? attrs["trial"].get<int>() : 0;
        // After all event groups have been collected, start over with the first one
        int trial_max = (perf_event_names.size() + perf_group_size - 1) / perf_group_size;
        trial = trial % trial_max;
        #pragma omp parallel
        {
            int tid = get_thread_id();
            num_traversed_edges[tid] = 0;
            perf.open(tid, trial, perf_group_size);
            perf.start(tid, trial, perf_group_size);
        }
#endif
        // Start the timer
        t1 = std::chrono::steady_clock::now();
    }

    void __attribute__ ((noinline))
    region_end()
    {
        // Stop the timer
        t2 = std::chrono::steady_clock::now();

        // End the ROI
#if defined(ENABLE_SNIPER_HOOKS)
        parmacs_roi_end();
#elif defined(ENABLE_GEM5_HOOKS)
        m5_dumpreset_stats(0,0);
#elif defined(ENABLE_PIN_HOOKS)
        __asm__("");
#elif defined(ENABLE_PERF_HOOKS)
        #pragma omp parallel
        {
            int tid = get_thread_id();
            perf.stop(tid, trial, perf_group_size);
        }
#endif

        // Populate the results object
        json results = attrs;

        // Check for mismatched begin/end pairs
        if (region_name == "") {
            cerr << "ERROR: called region_end before region_begin\n";
            exit(-1);
        } else {
            // Set region name in output and reset for next region
            results["region_name"] = region_name;
            region_name = "";
        }

        // Save # of traversed edges if the function was used
        int64_t total_edges_traversed = 0;
        for (int64_t n : num_traversed_edges) {
            total_edges_traversed += n;
        }
        if (total_edges_traversed > 0) {
            results["num_traversed_edges"] = num_traversed_edges;
        }

        // Record time elapsed
        results["time_ms"] = std::chrono::duration<double, std::milli>(t2-t1).count();

        // Copy stats to the results object
        for (json::iterator it = stats.begin(); it != stats.end(); ++it){
            results[it.key()] = it.value();
        }
        // Stats get cleared at the end of each region, attrs do not.
        stats.clear();

#if defined(ENABLE_PERF_HOOKS)
        // Copy recorded counters into the output
        json counters = json::parse(perf.toString(trial, perf_group_size));
        for (json::iterator it = counters.begin(); it != counters.end(); ++it) {
            results[it.key()] = it.value();
        }
#endif

#if defined(USE_MPI)
        // Combine results from each rank so only rank 0 prints to stdout
        int rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        // Serialize results to string
        string local_results_string = results.dump();
        // Get string lengths from each process
        int32_t local_string_length = local_results_string.size();
        vector<int32_t> string_lengths(comm_size);
        MPI_Gather(
            &local_string_length, 1, MPI_INT64_T,
            string_lengths.data(), 1, MPI_INT64_T,
            0, MPI_COMM_WORLD
        );

        // Allocate storage for final string
        int total_string_length = std::accumulate(string_lengths.begin(), string_lengths.end(), 0);
        vector<char> global_results_vec(total_string_length);
        // Figure out where to put each incoming string
        std::valarray<int32_t> displacements(comm_size + 1);
        std::partial_sum(string_lengths.begin(), string_lengths.end(), begin(displacements));
        // Prefix sum gave us the end of each string, but we need the beginnings, so shift the whole list to the right
        // We have one extra spot at the end of the string, so we aren't losing anything
        displacements.shift(-1);

        // Get strings from each process
        MPI_Gatherv(
            // Some old versions of mpi.h aren't const-correct
            const_cast<char*>(local_results_string.c_str()), local_results_string.size(), MPI_CHAR,
            global_results_vec.data(), string_lengths.data(), &displacements[0], MPI_CHAR,
            0, MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            // Parse out the results string from each process
            vector<json> results_by_pid(comm_size);
            for (int i = 0; i < comm_size; ++i)
            {
                // Find the string position in the data
                char * s = global_results_vec.data() + displacements[i];
                size_t len = displacements[i + 1] - displacements[i];
                // Parse back into json object
                results_by_pid[i] = json::parse(string(s,len));
            }

            // For each top-level key, build an array of values from each pid for that key
            for (json::iterator it = results.begin(); it != results.end(); ++it)
            {
                string key = it.key();
                results[key] = json::array();
                for (int i = 1; i < comm_size; ++i)
                {
                    results[key] += results_by_pid[i][key];
                }
            }
        }

#endif

#if defined(HOOKS_PRETTY_PRINT)
    // setw is overloaded to format json with indents
    int indent = 2;
#else
    int indent = 0;
#endif


#if defined(USE_MPI)
    if (rank == 0){
#endif

        // At this point we've accumulated all the data for this ROI into a json object (results)
        // Finally, send it to the output stream
        out << std::setw(indent) << results << std::endl;

#if defined(USE_MPI)
    }
#endif


    }

    void traverse_edges(int64_t n) {
        num_traversed_edges[get_thread_id()] += n;
    }
    template<typename T>
    void
    set_attr(std::string key, T value) {
        attrs[key] = value;
    }
    template<typename T>
    void
    set_stat(std::string key, T value) {
        stats[key] = value;
    }

};

// Implementation of Hooks
// This is just a Singleton that forwards all calls to the private Hooks::implementation (impl)
// This minimizes the number of code changes that need to be made in calling code

Hooks&
Hooks::getInstance()
{
    static Hooks instance;
    return instance;
}

Hooks::Hooks()                                              { pimpl = new Hooks::impl(); }
Hooks::~Hooks()                                             { delete pimpl; }
void Hooks::region_begin(string name)                       { pimpl->region_begin(name); }
void Hooks::region_end()                                    { pimpl->region_end(); }
void Hooks::set_attr(std::string key, uint64_t value)       { pimpl->set_attr(key, value); }
void Hooks::set_attr(std::string key, int64_t value)        { pimpl->set_attr(key, value); }
void Hooks::set_attr(std::string key, double value)         { pimpl->set_attr(key, value); }
void Hooks::set_attr(std::string key, std::string value)    { pimpl->set_attr(key, value); }
void Hooks::set_stat(std::string key, uint64_t value)       { pimpl->set_stat(key, value); }
void Hooks::set_stat(std::string key, int64_t value)        { pimpl->set_stat(key, value); }
void Hooks::set_stat(std::string key, double value)         { pimpl->set_stat(key, value); }
void Hooks::set_stat(std::string key, std::string value)    { pimpl->set_stat(key, value); }
void Hooks::traverse_edges(uint64_t n)                      { pimpl->traverse_edges(n); }

// Implementation of C interface
//

extern "C" void
hooks_region_begin(const char* name)
{
    Hooks::getInstance().region_begin(name);
}

extern "C" void
hooks_region_end()
{
    Hooks::getInstance().region_end();
}

extern "C" void
hooks_set_attr_i64(const char * key, int64_t value)
{
    Hooks::getInstance().set_attr(key, value);
}

extern "C" void
hooks_set_attr_u64(const char * key, uint64_t value)
{
    Hooks::getInstance().set_attr(key, value);
}

extern "C" void
hooks_set_attr_f64(const char * key, double value)
{
    Hooks::getInstance().set_attr(key, value);
}

extern "C" void
hooks_set_attr_str(const char * key, const char* value)
{
    Hooks::getInstance().set_attr(key, value);
}

extern "C" void
hooks_traverse_edges(uint64_t n)
{
    Hooks::getInstance().traverse_edges(n);
}
