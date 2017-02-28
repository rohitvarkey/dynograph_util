#include "rmat_dataset.h"
#include "helpers.h"

using namespace DynoGraph;

int64_t
parse_int_with_suffix(std::string token)
{
    int64_t n = static_cast<int64_t>(std::stoll(token));
    switch(token.back())
    {
        case 'K': n *= 1LL << 10; break;
        case 'M': n *= 1LL << 20; break;
        case 'G': n *= 1LL << 30; break;
        case 'T': n *= 1LL << 40; break;
        default: break;
    }
    return n;
}

RmatArgs
RmatArgs::from_string(std::string str)
{
    std::istringstream ss(str);

    RmatArgs args;
    std::string token;
    // Format is a-b-c-d-ne-nv.rmat
    // Example: 0.55-0.15-0.15-0.15-500M-1M.rmat
    std::getline(ss, token, '-'); args.a = std::stod(token);
    std::getline(ss, token, '-'); args.b = std::stod(token);
    std::getline(ss, token, '-'); args.c = std::stod(token);
    std::getline(ss, token, '-'); args.d = std::stod(token);
    std::getline(ss, token, '-'); args.num_edges = parse_int_with_suffix(token);
    std::getline(ss, token, '.'); args.num_vertices = parse_int_with_suffix(token);

    return args;
}

std::string
RmatArgs::validate() const {
    std::ostringstream oss;
    // Validate parameters
    if (a < 0 || b < 0 || c < 0 || d < 0
        ||  a > 1 || b > 1 || c > 1 || d > 1
        ||  a + b + c + d > 1.0)
    {
        oss << "Invalid arguments: RMAT parameters must be fall in the range [0, 1] and sum to 1\n";
    } else if (num_edges < 0 || num_vertices < 0) {
        oss << "Invalid arguments: RMAT graph must have a positive number of edges and vertices\n";
    }
    return oss.str();
}

RmatDataset::RmatDataset(Args args, RmatArgs rmat_args)
: args(args)
, rmat_args(rmat_args)
, current_batch(0)
, num_edges(rmat_args.num_edges)
, num_batches(num_edges / args.batch_size)
, num_vertices(rmat_args.num_vertices)
, next_timestamp(0)
, generator(rmat_args.num_vertices, rmat_args.a, rmat_args.b, rmat_args.c, rmat_args.d)
{
    Logger &logger = Logger::get_instance();

    // Sanity check on arguments
    if (static_cast<size_t>(args.batch_size) > num_edges)
    {
        logger << "Invalid arguments: batch size (" << args.batch_size << ") "
               << "cannot be larger than the total number of edges in the dataset "
               << " (" << num_edges << ")\n";
        die();
    }

    if (args.num_epochs > num_batches)
    {
        logger << "Invalid arguments: number of epochs (" << args.num_epochs << ") "
               << "cannot be greater than the number of batches in the dataset "
               << "(" << num_batches << ")\n";
        die();
    }
}

int64_t
RmatDataset::getTimestampForWindow(int64_t batchId) const
{
    int64_t timestamp;
    // Calculate width of timestamp window
    int64_t window_time = num_edges * args.window_size;
    // Get the timestamp of the last edge in the current batch
    int64_t latest_time = (batchId+1)*args.batch_size;

    timestamp = std::max((int64_t)0, latest_time - window_time);

    return timestamp;
};

std::shared_ptr<Batch>
RmatDataset::getBatch(int64_t batchId)
{
    // Since this is a graph generator, batches must be generated in order
    assert(batchId == current_batch);
    current_batch += 1;

    int64_t first_timestamp = next_timestamp;
    next_timestamp += args.batch_size;
    return std::make_shared<RmatBatch>(generator, args.batch_size, first_timestamp);
}

std::shared_ptr<Batch>
RmatDataset::getBatchesUpTo(int64_t batchId)
{
    // Since this is a graph generator, batches must be generated in order
    reset();
    current_batch = batchId + 1;

    int64_t first_timestamp = next_timestamp;
    next_timestamp += args.batch_size;
    return std::make_shared<RmatBatch>(generator, current_batch * args.batch_size, first_timestamp);
}

bool
RmatDataset::isDirected() const
{
    return true;
}

int64_t
RmatDataset::getMaxVertexId() const
{
    return num_vertices + 1;
}

int64_t
RmatDataset::getNumBatches() const {
    return num_batches;
}

int64_t
RmatDataset::getNumEdges() const {
    return num_edges;
};

int64_t
RmatDataset::getMinTimestamp() const {
    return 0;
}

int64_t
RmatDataset::getMaxTimestamp() const {
    return num_edges;
}

void
RmatDataset::reset() {
    current_batch = 0;
    next_timestamp = 0;
    generator = rmat_edge_generator(rmat_args.num_vertices, rmat_args.a, rmat_args.b, rmat_args.c, rmat_args.d);
}

// Implementation of RmatBatch
RmatBatch::RmatBatch(rmat_edge_generator &generator, int64_t size, int64_t first_timestamp)
    : Batch(edges.begin(), edges.end()), edges(size)
{
    for (int i = 0; i < size; ++i)
    {
        Edge& e = edges[i];
        do { generator.next_edge(&e.src, &e.dst); }
        while (e.src == e.dst); // Discard self-edges
        e.weight = 1;
        e.timestamp = first_timestamp++;
    }
    begin_iter = edges.begin();
    end_iter = edges.end();
}
