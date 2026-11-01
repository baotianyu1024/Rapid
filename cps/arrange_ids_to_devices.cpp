#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <filesystem>
#include <cstring>
#include <numeric>
#include <cmath>
#include <chrono>
#include <climits>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <omp.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// g++ -O3 -std=c++17 -fopenmp -pthread arrange_ids_to_devices_v3.cpp -o arrange_ids_to_devices_v3

// --- ID encoding ---
static inline uint64_t encode_id(int tid, int fid) {
    return ((uint64_t)(uint32_t)tid << 32) | (uint32_t)fid;
}
static inline int decode_tid(uint64_t id) { return (int)(id >> 32); }
static inline int decode_fid(uint64_t id) { return (int)(id & 0xFFFFFFFF); }

// --- Args ---
struct Args {
    std::string input_bin;
    std::string hot_ids_dir;
    std::string output_dir;
    std::string init_state, state_out, seed_assignment;
    uint64_t start_sample = 0;
    int n_batches = 200;
    int batch_size = 8192;
    int num_devices = 8;
    std::vector<int> excluded_tables;
    int dense_count = 13;
    std::vector<int> table_lengths = {
        3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12,
        100, 27, 10, 3, 1, 1
    };
    int strategy = 3;              // default: migration threshold (best strategy)
    float cap_penalty = 0.002f;
    float migrate_threshold = 2.0f; // best threshold from grid search
    float beta = 0.0001f;          // penalty for device_size in greedy assignment
    bool strict_export_capacity = false;

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + arg);
            if (arg == "--input_bin") input_bin = argv[++i];
            else if (arg == "--hot_ids_dir") hot_ids_dir = argv[++i];
            else if (arg == "--output_dir") output_dir = argv[++i];
            else if (arg == "--seed_assignment") seed_assignment = argv[++i];
            else if (arg == "--init_state") init_state = argv[++i];
            else if (arg == "--state_out") state_out = argv[++i];
            else if (arg == "--start_sample") {
                std::string value = argv[++i];
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::runtime_error("start_sample must be a nonnegative integer");
                start_sample = std::stoull(value);
            }
            else if (arg == "--n_batches") n_batches = std::stoi(argv[++i]);
            else if (arg == "--batch_size") batch_size = std::stoi(argv[++i]);
            else if (arg == "--num_devices") num_devices = std::stoi(argv[++i]);
            else if (arg == "--dense_count") dense_count = std::stoi(argv[++i]);
            else if (arg == "--strategy") strategy = std::stoi(argv[++i]);
            else if (arg == "--cap_penalty") cap_penalty = std::stof(argv[++i]);
            else if (arg == "--beta") beta = std::stof(argv[++i]);
            else if (arg == "--strict_export_capacity") {
                std::string value = argv[++i];
                if (value != "0" && value != "1") throw std::runtime_error("strict_export_capacity must be 0 or 1");
                strict_export_capacity = value == "1";
            }
            else if (arg == "--migrate_threshold") migrate_threshold = std::stof(argv[++i]);
            else if (arg == "--excluded_tables") {
                excluded_tables.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-')
                    excluded_tables.push_back(std::stoi(argv[++i]));
            }
            else if (arg == "--table_lengths") {
                table_lengths.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-')
                    table_lengths.push_back(std::stoi(argv[++i]));
            }
            else throw std::runtime_error("Unknown argument: " + arg);
        }
        if (input_bin.empty() || hot_ids_dir.empty() || output_dir.empty())
            throw std::runtime_error("--input_bin, --hot_ids_dir, --output_dir required");
        if (n_batches <= 0 || batch_size <= 0 || num_devices <= 0 || num_devices > 128 ||
            int64_t(batch_size) * num_devices > INT_MAX || dense_count < 0 ||
            strategy < 1 || strategy > 4 || !std::isfinite(beta) || beta < 0 ||
            !std::isfinite(cap_penalty) || cap_penalty < 0 ||
            !std::isfinite(migrate_threshold) || migrate_threshold < 0 || table_lengths.empty())
            throw std::runtime_error("Invalid placement parameters");
        for (int len : table_lengths)
            if (len <= 0) throw std::runtime_error("table_lengths must be positive");
        for (int tid : excluded_tables)
            if (tid < 0 || size_t(tid) >= table_lengths.size())
                throw std::runtime_error("Invalid excluded table");
        if (!fs::is_directory(hot_ids_dir)) throw std::runtime_error("Missing hot_ids_dir");
        if (!seed_assignment.empty() && (!init_state.empty() || !state_out.empty()))
            throw std::runtime_error("Seed assignment cannot be combined with checkpoints");
        if (!state_out.empty() && fs::exists(state_out))
            throw std::runtime_error("Refusing to overwrite checkpoint: " + state_out);
    }
};

// --- Hot ID loading ---
std::unordered_set<uint64_t> load_hot_ids(const std::string& dir,
                                           const std::vector<int>& excluded,
                                           const std::vector<int>& table_lengths) {
    std::unordered_set<uint64_t> hot;
    std::unordered_set<int> excl(excluded.begin(), excluded.end());
    for (size_t tid = 0; tid < table_lengths.size(); ++tid) {
        if (excl.count(tid)) continue;
        auto path = fs::path(dir) / (std::to_string(tid) + ".txt");
        if (!fs::exists(path)) continue;
        if (!fs::is_regular_file(path)) throw std::runtime_error("Invalid hot ID file: " + path.string());
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open hot ID file: " + path.string());
        std::string line;
        while (std::getline(f, line)) {
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            if (line.empty()) continue;
            std::istringstream row(line);
            int fid;
            char separator;
            int64_t count;
            std::string extra;
            if (!(row >> fid >> separator >> count) || separator != ':' || fid < 0 || count < 0 || (row >> extra))
                throw std::runtime_error("Invalid hot ID row: " + path.string());
            hot.insert(encode_id((int)tid, fid));
        }
        if (f.bad()) throw std::runtime_error("Cannot read hot ID file: " + path.string());
    }
    std::cout << "Hot IDs loaded: " << hot.size() << "\n";
    return hot;
}

// --- Index mapping ---
struct IndexEntry { int sparse_idx; int table_id; };

std::vector<IndexEntry> get_indices_to_check(const std::vector<int>& table_lengths,
                                              const std::vector<int>& excluded) {
    std::vector<IndexEntry> entries;
    std::unordered_set<int> excl(excluded.begin(), excluded.end());
    int offset = 0;
    for (size_t tid = 0; tid < table_lengths.size(); ++tid) {
        if (!excl.count(tid))
            for (int i = 0; i < table_lengths[tid]; ++i)
                entries.push_back({offset + i, (int)tid});
        offset += table_lengths[tid];
    }
    return entries;
}

// --- Flat ID registry: maps uint64_t IDs to sequential indices ---
// Thread-safe for concurrent reads, serial for writes (new ID registration)
struct IDRegistry {
    std::unordered_map<uint64_t, uint32_t> id_to_idx;
    std::vector<uint64_t> idx_to_id;
    uint32_t size = 0;

    void reserve(size_t n) {
        id_to_idx.reserve(n);
        idx_to_id.reserve(n);
    }

    // Returns index for existing ID, or UINT32_MAX if not found
    uint32_t find(uint64_t id) const {
        auto it = id_to_idx.find(id);
        return it != id_to_idx.end() ? it->second : UINT32_MAX;
    }

    // Register new ID, returns its index. NOT thread-safe.
    uint32_t insert(uint64_t id) {
        if (size >= INT_MAX - 1) throw std::runtime_error("Registry exceeds int32 capacity");
        auto [it, inserted] = id_to_idx.try_emplace(id, size);
        if (inserted) {
            idx_to_id.push_back(id);
            size++;
        }
        return it->second;
    }
};

// --- Simulation with flat arrays ---
struct SimResult {
    IDRegistry registry;
    std::vector<int32_t> counts_flat; // [idx * num_devices + d]
    std::vector<int8_t> id_device;    // [idx] -> device; hot IDs stay -1 internally
    std::vector<int> device_sizes;   // cold owners only, not exported hot placement
    uint64_t origin_sample = 0, next_sample = 0, batches_processed = 0;
    uint64_t migration_events = 0;   // cumulative events, NOT net owner changes
};

static uint64_t record_bytes(const Args& args) {
    return 4 * (1ULL + args.dense_count +
                std::accumulate(args.table_lengths.begin(), args.table_lengths.end(), 0ULL));
}

// Input identity is canonical path + size + mtime (not a full trace content hash).
// OMP thread count is included because registration order follows static thread order.
static std::string state_parameters(const Args& args) {
    std::ostringstream s;
    s << "rapid-globalbatch-v1\n" << fs::canonical(args.input_bin).string() << '\n'
      << fs::file_size(args.input_bin) << '\n'
      << fs::last_write_time(args.input_bin).time_since_epoch().count() << '\n'
      << args.num_devices << ' ' << args.batch_size << ' ' << args.dense_count << ' '
      << args.strategy << ' ' << std::hexfloat << args.beta << ' ' << args.cap_penalty
      << ' ' << args.migrate_threshold << ' ' << omp_get_max_threads() << '\n';
    for (int len : args.table_lengths) s << len << ' ';
    s << '\n';
    for (int tid : args.excluded_tables) s << tid << ' ';
    return s.str();
}

// Versioned native-endian format, with endian marker and a checksum of every payload byte.
// Array lengths are checked against the remaining file size BEFORE allocation.
struct StateIO {
    std::fstream file;
    uint64_t hash = 14695981039346656037ULL, remaining = 0;
    bool writing;
    StateIO(const std::string& path, bool write) : writing(write) {
        file.open(path, std::ios::binary | (write ? (std::ios::out | std::ios::trunc) : std::ios::in));
        if (!file) throw std::runtime_error("Cannot open checkpoint: " + path);
        if (!write) {
            auto size = fs::file_size(path);
            if (size < sizeof(uint64_t)) throw std::runtime_error("Truncated checkpoint");
            remaining = size - sizeof(uint64_t);
        }
    }
    void bytes(void* data, size_t size) {
        if (size > size_t(std::numeric_limits<std::streamsize>::max()) ||
            (!writing && size > remaining)) throw std::runtime_error("Invalid checkpoint length");
        if (writing) file.write(static_cast<char*>(data), size);
        else { file.read(static_cast<char*>(data), size); remaining -= size; }
        if (!file) throw std::runtime_error("Checkpoint IO failed");
        auto p = static_cast<unsigned char*>(data);
        for (size_t i = 0; i < size; ++i) { hash ^= p[i]; hash *= 1099511628211ULL; }
    }
    template<class T> void scalar(T& value) { bytes(&value, sizeof(T)); }
    template<class T> void array(std::vector<T>& values, uint64_t n) {
        if (n > SIZE_MAX / sizeof(T) || (!writing && n > remaining / sizeof(T)))
            throw std::runtime_error("Invalid checkpoint array size");
        if (!writing) values.resize(size_t(n));
        else if (n > values.size()) throw std::runtime_error("Short state array");
        bytes(values.data(), size_t(n) * sizeof(T));
    }
    void finish() {
        if (writing) { file.write(reinterpret_cast<char*>(&hash), sizeof(hash)); file.flush(); }
        else {
            if (remaining != 0) throw std::runtime_error("Unexpected trailing checkpoint payload");
            uint64_t expected = 0;
            file.read(reinterpret_cast<char*>(&expected), sizeof(expected));
            if (hash != expected) throw std::runtime_error("Checkpoint checksum mismatch");
        }
        if (!file) throw std::runtime_error("Checkpoint IO failed");
        file.close();
        if (file.fail()) throw std::runtime_error("Checkpoint close failed");
    }
};

static void checkpoint(const Args& args, const std::unordered_set<uint64_t>& hot,
                       SimResult& state, bool writing) {
    static_assert(sizeof(int) == 4, "Checkpoint requires 32-bit int");
    const std::string path = writing ? args.state_out : args.init_state;
    const fs::path temporary_dir = path + ".tmp";
    const std::string temporary = (temporary_dir / "state.bin").string();
    if (writing && (fs::exists(path) || !fs::create_directory(temporary_dir)))
        throw std::runtime_error("Checkpoint destination already exists or is being written");
    StateIO io(writing ? temporary : path, writing);
    uint64_t magic = 0x5241504944535431ULL;
    uint32_t endian = 0x01020304;
    io.scalar(magic); io.scalar(endian);
    if (magic != 0x5241504944535431ULL || endian != 0x01020304)
        throw std::runtime_error("Unsupported checkpoint version or byte order");
    std::string expected = state_parameters(args), parameters = expected;
    uint64_t parameter_size = parameters.size();
    io.scalar(parameter_size);
    if (parameter_size != expected.size()) throw std::runtime_error("Checkpoint parameter mismatch");
    io.bytes(parameters.data(), parameter_size);
    if (parameters != expected) throw std::runtime_error("Checkpoint parameter/input mismatch");
    std::vector<uint64_t> hot_ids(hot.begin(), hot.end());
    std::sort(hot_ids.begin(), hot_ids.end());
    uint64_t nhot = hot_ids.size();
    io.scalar(nhot);
    if (nhot != hot.size()) throw std::runtime_error("Checkpoint hot-set mismatch");
    auto expected_hot = hot_ids;
    io.array(hot_ids, nhot);
    if (hot_ids != expected_hot) throw std::runtime_error("Checkpoint hot-set mismatch");
    io.scalar(state.origin_sample); io.scalar(state.next_sample);
    io.scalar(state.batches_processed); io.scalar(state.migration_events);
    uint64_t gb = uint64_t(args.batch_size) * args.num_devices;
    uint64_t total_samples = fs::file_size(args.input_bin) / record_bytes(args);
    if (state.next_sample < state.origin_sample || state.next_sample > total_samples ||
        state.next_sample - state.origin_sample > INT_MAX ||
        (state.next_sample - state.origin_sample) % gb != 0 ||
        state.batches_processed != (state.next_sample - state.origin_sample) / gb ||
        (!writing && state.next_sample != args.start_sample))
        throw std::runtime_error("Checkpoint cursor/window mismatch");
    uint64_t n = state.registry.size;
    io.scalar(n);
    if (n >= INT_MAX) throw std::runtime_error("Checkpoint registry too large");
    uint64_t row_bytes = sizeof(uint64_t) + sizeof(int8_t) + sizeof(int32_t) * args.num_devices;
    if (!writing && (io.remaining < uint64_t(args.num_devices) * sizeof(int) ||
        n > (io.remaining - uint64_t(args.num_devices) * sizeof(int)) / row_bytes))
        throw std::runtime_error("Truncated checkpoint arrays");
    io.array(state.registry.idx_to_id, n);
    io.array(state.counts_flat, n * args.num_devices);
    io.array(state.id_device, n);
    io.array(state.device_sizes, args.num_devices);
    io.finish();
    if (!writing) {
        state.registry.reserve(n);
        std::vector<int> sizes(args.num_devices, 0);
        std::unordered_set<int> excluded(args.excluded_tables.begin(), args.excluded_tables.end());
        for (uint32_t i = 0; i < n; ++i) {
            uint64_t id = state.registry.idx_to_id[i];
            int tid = decode_tid(id), d = state.id_device[i];
            if (tid < 0 || size_t(tid) >= args.table_lengths.size() || excluded.count(tid) ||
                !state.registry.id_to_idx.emplace(id, i).second ||
                (hot.count(id) ? d != -1 : (d < 0 || d >= args.num_devices)))
                throw std::runtime_error("Invalid checkpoint registry/owner");
            uint64_t total = 0;
            for (int dev = 0; dev < args.num_devices; ++dev) {
                auto count = state.counts_flat[size_t(i) * args.num_devices + dev];
                if (count < 0) throw std::runtime_error("Negative checkpoint count");
                total += count;
            }
            if (total == 0 || total > state.next_sample - state.origin_sample)
                throw std::runtime_error("Invalid checkpoint count sum");
            if (d >= 0) ++sizes[d];
        }
        if (sizes != state.device_sizes) throw std::runtime_error("Checkpoint device_sizes mismatch");
        state.registry.size = uint32_t(n);
    } else {
        // Same-directory hard link publishes atomically and refuses an existing target.
        fs::create_hard_link(temporary, path);
        fs::remove(temporary);
        fs::remove(temporary_dir);
    }
}

void run_simulation(const Args& args,
                    const std::unordered_set<uint64_t>& hot_id_set,
                    const std::vector<IndexEntry>& indices_to_check,
                    SimResult& result) {

    int ND = args.num_devices;
    int GLOBAL_BATCH = ND * args.batch_size;
    int num_indices = (int)indices_to_check.size();

    long long total_sparse = std::accumulate(args.table_lengths.begin(), args.table_lengths.end(), 0LL);
    size_t record_size = 4 + args.dense_count * 4 + total_sparse * 4;
    int sparse_off = 4 + args.dense_count * 4;

    bool use_cap_penalty = (args.strategy == 2 || args.strategy == 4);
    bool use_threshold = (args.strategy == 3 || args.strategy == 4);

    // Small initial capacity, geometric growth; only live rows are checkpointed.
    auto& registry = result.registry;
    auto& counts_flat = result.counts_flat;
    auto& id_device = result.id_device;
    auto& device_sizes = result.device_sizes;
    size_t max_ids = std::max<size_t>(4096, registry.size);
    registry.reserve(max_ids);
    counts_flat.resize(max_ids * ND, 0);
    id_device.resize(max_ids, -1);
    device_sizes.resize(ND, 0);
    if (uint64_t(GLOBAL_BATCH) > SIZE_MAX / record_size)
        throw std::runtime_error("Batch buffer size overflow");
    std::vector<char> batch_buffer((size_t)GLOBAL_BATCH * record_size);

    std::unique_ptr<FILE, decltype(&fclose)> input(fopen(args.input_bin.c_str(), "rb"), fclose);
    FILE* f = input.get();
    if (!f || args.start_sample > uint64_t(std::numeric_limits<off_t>::max()) / record_size ||
        fseeko(f, off_t(args.start_sample * record_size), SEEK_SET) != 0)
        throw std::runtime_error("Cannot open/seek input");

    // Per-sample parsed data: use flat arrays instead of vectors of vectors
    // For each sample: store its IDs as indices into the registry
    struct SampleIDs {
        std::vector<uint32_t> all_idx;   // registry indices for all IDs
        std::vector<uint32_t> cold_idx;  // registry indices for cold IDs
    };

    auto total_start = std::chrono::high_resolution_clock::now();
    double t_parse = 0, t_hits = 0, t_assign = 0, t_stats = 0, t_newid = 0, t_migrate = 0;

    for (int batch = 0; batch < args.n_batches; ++batch) {
        size_t records_read = fread(batch_buffer.data(), record_size, GLOBAL_BATCH, f);
        if (ferror(f)) throw std::runtime_error("Input read failed");
        if ((!args.init_state.empty() || !args.state_out.empty()) && records_read != size_t(GLOBAL_BATCH))
            throw std::runtime_error("Checkpoint runs require complete global batches");
        if (records_read == 0) break;
        int N = (int)records_read;

        auto t0 = std::chrono::high_resolution_clock::now();

        // === Phase 1: Parse samples, register IDs, build index arrays ===
        // Step 1a: Parallel parse + dedup per sample (output: raw eids)
        std::vector<std::vector<uint64_t>> all_eids(N), cold_eids(N);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const char* rec = batch_buffer.data() + (size_t)i * record_size;
            const int32_t* sp = reinterpret_cast<const int32_t*>(rec + sparse_off);

            auto& ae = all_eids[i];
            ae.reserve(num_indices);
            for (int j = 0; j < num_indices; ++j)
                ae.push_back(encode_id(indices_to_check[j].table_id,
                                        sp[indices_to_check[j].sparse_idx]));
            std::sort(ae.begin(), ae.end());
            ae.erase(std::unique(ae.begin(), ae.end()), ae.end());

            auto& ce = cold_eids[i];
            ce.reserve(ae.size());
            for (auto eid : ae)
                if (hot_id_set.find(eid) == hot_id_set.end())
                    ce.push_back(eid);
        }

        // Step 1b: Collect batch-unique IDs via parallel merge into sorted set
        // Then register only new ones (serial but fast — just hash lookups for existing)
        {
            // Parallel: find IDs not yet in registry
            std::vector<std::vector<uint64_t>> new_ids_per_thread;
            int nt = omp_get_max_threads();
            new_ids_per_thread.resize(nt);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& local_new = new_ids_per_thread[tid];
                #pragma omp for schedule(static)
                for (int i = 0; i < N; ++i) {
                    for (auto eid : all_eids[i]) {
                        if (registry.find(eid) == UINT32_MAX)
                            local_new.push_back(eid);
                    }
                }
            }

            // Serial: register only truly new IDs
            for (auto& vec : new_ids_per_thread)
                for (auto eid : vec)
                    registry.insert(eid); // insert is idempotent
        }

        // Grow flat arrays if needed
        if (registry.size > max_ids) {
            size_t new_max = std::max<size_t>(registry.size, max_ids + max_ids / 2);
            counts_flat.resize(new_max * ND, 0);
            id_device.resize(new_max, -1);
            max_ids = new_max;
        }

        // Step 1c: Parallel convert to index-based
        std::vector<SampleIDs> samples(N);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            auto& s = samples[i];
            s.all_idx.resize(all_eids[i].size());
            for (size_t j = 0; j < all_eids[i].size(); ++j)
                s.all_idx[j] = registry.find(all_eids[i][j]);
            s.cold_idx.resize(cold_eids[i].size());
            for (size_t j = 0; j < cold_eids[i].size(); ++j)
                s.cold_idx[j] = registry.find(cold_eids[i][j]);
        }
        all_eids.clear(); cold_eids.clear();

        auto t1 = std::chrono::high_resolution_clock::now();

        // === Phase 2: Compute hit counts ===
        std::vector<int> hit_counts((size_t)N * ND, 0);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            int* sh = hit_counts.data() + (size_t)i * ND;
            for (auto idx : samples[i].cold_idx) {
                int8_t dev = id_device[idx];
                if (dev >= 0) sh[dev]++;
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        // === Phase 3: Greedy assignment ===
        int target = N / ND + (N % ND != 0);
        std::vector<int> dev_batch(ND, 0);
        std::vector<int> assignments(N);

        for (int i = 0; i < N; ++i) {
            float best_score = -1e9f;
            int best_d = -1;
            for (int d = 0; d < ND; ++d) {
                if (dev_batch[d] >= target) continue;
                float score = (float)hit_counts[(size_t)i * ND + d] - args.beta * device_sizes[d];
                if (score > best_score) { best_score = score; best_d = d; }
            }
            if (best_d == -1) {
                best_d = (int)(std::min_element(dev_batch.begin(), dev_batch.end()) - dev_batch.begin());
            }
            assignments[i] = best_d;
            dev_batch[best_d]++;
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        // === Phase 4: Update counts_flat with parallel atomic updates ===
        // ~26M updates spread across 80M×32 array — low contention expected
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            int d = assignments[i];
            for (auto idx : samples[i].all_idx) {
                #pragma omp atomic
                counts_flat[(size_t)idx * ND + d]++;
            }
        }

        auto t4 = std::chrono::high_resolution_clock::now();

        // === Phase 4b: Register new cold IDs + collect active cold ===
        std::vector<uint32_t> active_cold_idx;
        {
            // Collect all cold indices, then sort+dedup (faster than bitset scan over 80M)
            std::vector<uint32_t> raw_cold;
            raw_cold.reserve(size_t(N) * 50);

            for (int i = 0; i < N; ++i) {
                int d = assignments[i];
                for (auto idx : samples[i].cold_idx) {
                    raw_cold.push_back(idx);
                    if (id_device[idx] < 0) {
                        id_device[idx] = (int8_t)d;
                        device_sizes[d]++;
                    }
                }
            }

            // Sort + dedup
            std::sort(raw_cold.begin(), raw_cold.end());
            raw_cold.erase(std::unique(raw_cold.begin(), raw_cold.end()), raw_cold.end());
            active_cold_idx = std::move(raw_cold);
        }

        auto t5 = std::chrono::high_resolution_clock::now();

        // === Phase 5: Migration ===
        double avg_sz = 0;
        for (int d = 0; d < ND; ++d) avg_sz += device_sizes[d];
        avg_sz /= ND;

        std::vector<std::pair<int, int>> mig_result(active_cold_idx.size());

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < active_cold_idx.size(); ++i) {
            uint32_t idx = active_cold_idx[i];
            int cur = id_device[idx];
            if (cur < 0) { mig_result[i] = {-1, -1}; continue; }

            const int32_t* cnt = counts_flat.data() + (size_t)idx * ND;

            int best = 0;
            float best_score = -1e18f;
            for (int d = 0; d < ND; ++d) {
                float score = (float)cnt[d];
                if (use_cap_penalty)
                    score -= args.cap_penalty * (float)std::max(0.0, (double)device_sizes[d] - avg_sz);
                if (score > best_score) { best_score = score; best = d; }
            }

            bool should = false;
            if (best != cur) {
                if (use_threshold) {
                    float threshold = cnt[cur] * args.migrate_threshold;
                    should = threshold < float(INT_MAX) && (cnt[best] > (int)threshold);
                } else {
                    float cur_score = (float)cnt[cur];
                    if (use_cap_penalty)
                        cur_score -= args.cap_penalty * (float)std::max(0.0, (double)device_sizes[cur] - avg_sz);
                    should = (best_score > cur_score);
                }
            }
            mig_result[i] = should ? std::make_pair(cur, best) : std::make_pair(-1, -1);
        }

        // Execute migrations (serial, just updating flat arrays)
        int mig_count = 0;
        for (size_t i = 0; i < active_cold_idx.size(); ++i) {
            if (mig_result[i].first == -1) continue;
            uint32_t idx = active_cold_idx[i];
            int from = mig_result[i].first, to = mig_result[i].second;
            id_device[idx] = (int8_t)to;
            device_sizes[from]--;
            device_sizes[to]++;
            mig_count++;
        }
        result.next_sample += N;
        result.batches_processed++;
        result.migration_events += mig_count;

        auto t6 = std::chrono::high_resolution_clock::now();

        t_parse += std::chrono::duration<double>(t1 - t0).count();
        t_hits += std::chrono::duration<double>(t2 - t1).count();
        t_assign += std::chrono::duration<double>(t3 - t2).count();
        t_stats += std::chrono::duration<double>(t4 - t3).count();
        t_newid += std::chrono::duration<double>(t5 - t4).count();
        t_migrate += std::chrono::duration<double>(t6 - t5).count();

        if (batch % 10 == 0 || batch == args.n_batches - 1) {
            size_t total_ids = registry.size;
            int min_sz = *std::min_element(device_sizes.begin(), device_sizes.end());
            int max_sz = *std::max_element(device_sizes.begin(), device_sizes.end());
            double batch_ms = std::chrono::duration<double, std::milli>(t6 - t0).count();
            std::cout << "Batch " << batch + 1 << "/" << args.n_batches
                      << " (" << (int)batch_ms << " ms, IDs: " << total_ids
                      << ", min/max: " << min_sz << "/" << max_sz
                      << ", mig: " << mig_count << ")\n";
        }
    }

    input.reset();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n--- Timing breakdown ---\n"
              << "  Parse:   " << std::fixed << std::setprecision(1) << t_parse << "s\n"
              << "  Hits:    " << t_hits << "s\n"
              << "  Assign:  " << t_assign << "s\n"
              << "  Stats:   " << t_stats << "s\n"
              << "  NewID:   " << t_newid << "s\n"
              << "  Migrate: " << t_migrate << "s\n"
              << "  TOTAL:   " << total_s << "s\n";

}

// --- Write output ---
void write_output(const Args& args,
                  const std::unordered_set<uint64_t>& hot_id_set,
                  const SimResult& result) {
    int ND = args.num_devices;
    struct Bucket { std::vector<std::string> priority, normal; };
    std::vector<Bucket> buckets(ND);

    // Hot IDs: round-robin per table (same as baseline, ordered iteration)
    std::vector<uint64_t> hot_sorted(hot_id_set.begin(), hot_id_set.end());
    std::sort(hot_sorted.begin(), hot_sorted.end()); // ensure deterministic order
    std::unordered_map<int, int> table_rr;
    std::mt19937 rng(12345); // fixed seed for reproducible hot ID placement
    for (auto eid : hot_sorted) {
        int tid = decode_tid(eid), fid = decode_fid(eid);
        if (table_rr.find(tid) == table_rr.end())
            table_rr[tid] = std::uniform_int_distribution<int>(0, ND - 1)(rng);
        int d = table_rr[tid];
        table_rr[tid] = (d + 1) % ND;
        buckets[d].priority.push_back(std::to_string(tid) + ":" + std::to_string(fid));
    }

    // Cold IDs: sorted by total frequency descending
    struct ColdEntry { uint32_t idx; int device; long long total_freq; };
    std::vector<ColdEntry> cold_sorted;
    cold_sorted.reserve(result.registry.size);

    for (uint32_t i = 0; i < result.registry.size; ++i) {
        uint64_t eid = result.registry.idx_to_id[i];
        if (hot_id_set.count(eid)) continue;
        int dev = result.id_device[i];
        if (dev < 0) continue;
        long long total = 0;
        const int32_t* cnt = result.counts_flat.data() + (size_t)i * ND;
        for (int d = 0; d < ND; ++d) total += cnt[d];
        cold_sorted.push_back({i, dev, total});
    }

    std::sort(cold_sorted.begin(), cold_sorted.end(),
        [](const ColdEntry& a, const ColdEntry& b) {
            return a.total_freq != b.total_freq ? a.total_freq > b.total_freq : a.idx < b.idx;
        });

    if (args.strict_export_capacity) {
        // Paper Step 4: explicit final shard capacity, previously absent.
        // Retain high-frequency IDs first; relocate only overflow cold IDs.
        const size_t total = hot_sorted.size() + cold_sorted.size();
        const size_t capacity = (total + ND - 1) / ND;
        std::vector<size_t> load(ND, 0), overflow;
        for (int d = 0; d < ND; ++d) {
            load[d] = buckets[d].priority.size();
            if (load[d] > capacity) throw std::runtime_error("Hot IDs exceed shard capacity");
        }
        for (size_t i = 0; i < cold_sorted.size(); ++i) {
            int d = cold_sorted[i].device;
            if (load[d] < capacity) ++load[d];
            else overflow.push_back(i);
        }
        for (size_t i : overflow) {
            auto& entry = cold_sorted[i];
            const int32_t* counts = result.counts_flat.data() + size_t(entry.idx) * ND;
            int best = -1;
            for (int d = 0; d < ND; ++d) {
                if (load[d] >= capacity) continue;
                if (best < 0 || counts[d] > counts[best] ||
                    (counts[d] == counts[best] && load[d] < load[best])) best = d;
            }
            if (best < 0) throw std::runtime_error("No capacity for overflow ID");
            entry.device = best; ++load[best];
        }
        std::cout << "Strict export capacity: " << capacity << ", moved cold IDs: " << overflow.size() << "\n";
    }

    for (const auto& e : cold_sorted) {
        uint64_t eid = result.registry.idx_to_id[e.idx];
        int tid = decode_tid(eid), fid = decode_fid(eid);
        buckets[e.device].normal.push_back(std::to_string(tid) + ":" + std::to_string(fid));
    }

    fs::create_directories(args.output_dir);
    for (int d = 0; d < ND; ++d) {
        auto path = fs::path(args.output_dir) / ("table_" + std::to_string(d));
        std::ofstream out(path);
        for (const auto& s : buckets[d].priority) out << s << "\n";
        for (const auto& s : buckets[d].normal) out << s << "\n";
        out.close();
        if (!out) throw std::runtime_error("Cannot write allocation: " + path.string());
    }

    // Print stats
    long long total = 0;
    std::vector<long long> dc(ND);
    for (int d = 0; d < ND; ++d) {
        dc[d] = buckets[d].priority.size() + buckets[d].normal.size();
        total += dc[d];
    }
    double avg = (double)total / ND;
    long long mn = *std::min_element(dc.begin(), dc.end());
    long long mx = *std::max_element(dc.begin(), dc.end());
    std::cout << "Total IDs: " << total << ", avg: " << (long long)avg
              << ", min: " << mn << ", max: " << mx
              << ", imbalance: " << std::fixed << std::setprecision(2)
              << (avg ? 100.0 * (mx - mn) / avg : 0.0) << "%\n";
}

int main(int argc, char* argv[]) {
    try {
        auto process_start = Clock::now();
        Args args;
        args.parse(argc, argv);
        const uint64_t rec = record_bytes(args);
        const uint64_t bytes = fs::file_size(args.input_bin);
        const uint64_t requested = uint64_t(args.n_batches) * args.batch_size * args.num_devices;
        const bool exact = !args.init_state.empty() || !args.state_out.empty();
        if (rec > INT_MAX || args.start_sample > bytes / rec ||
            (exact && (bytes % rec != 0 || requested > bytes / rec - args.start_sample)))
            throw std::runtime_error("Invalid or out-of-range sample window");
        std::cout << "Strategy: " << args.strategy << ", beta: " << args.beta
                  << ", cap_penalty: " << args.cap_penalty
                  << ", migrate_threshold: " << args.migrate_threshold
                  << ", n_batches: " << args.n_batches
                  << ", OMP threads: " << omp_get_max_threads() << "\n";
        auto hot_ids = load_hot_ids(args.hot_ids_dir, args.excluded_tables, args.table_lengths);
        auto indices = get_indices_to_check(args.table_lengths, args.excluded_tables);
        SimResult result;
        result.origin_sample = result.next_sample = args.start_sample;
        if (!args.seed_assignment.empty()) {
            std::ifstream seed(args.seed_assignment);
            if (!seed) throw std::runtime_error("Cannot open seed assignment");
            result.device_sizes.assign(args.num_devices, 0);
            std::string line;
            while (std::getline(seed, line)) {
                std::istringstream row(line);
                int tid, fid, owner; std::string extra;
                if (!(row >> tid >> fid >> owner) || (row >> extra) || tid < 0 ||
                    tid >= int(args.table_lengths.size()) || fid < 0 || owner < 0 || owner >= args.num_devices)
                    throw std::runtime_error("Invalid seed assignment row");
                uint64_t id = encode_id(tid, fid);
                if (hot_ids.count(id) || result.registry.id_to_idx.count(id) ||
                    std::find(args.excluded_tables.begin(), args.excluded_tables.end(), tid) != args.excluded_tables.end())
                    throw std::runtime_error("Hot, duplicate or excluded seed ID");
                result.registry.insert(id);
                result.id_device.push_back(int8_t(owner));
                ++result.device_sizes[owner];
            }
            if (seed.bad()) throw std::runtime_error("Cannot read seed assignment");
            result.counts_flat.assign(size_t(result.registry.size) * args.num_devices, 0);
            std::cout << "METIS seed cold IDs: " << result.registry.size << "\n";
        }
        double read_s = 0, write_s = 0;
        if (!args.init_state.empty()) {
            auto t = Clock::now();
            checkpoint(args, hot_ids, result, false);
            read_s = seconds_since(t);
        }
        if (std::min(requested, bytes / rec - args.start_sample) >
            INT_MAX - (result.next_sample - result.origin_sample))
            throw std::runtime_error("Run exceeds int32 cumulative-count capacity");
        const uint64_t previous_rows = result.registry.size;
        const uint64_t previous_events = result.migration_events;
        auto previous_owners = result.id_device;
        auto t = Clock::now();
        run_simulation(args, hot_ids, indices, result);
        double simulation_s = seconds_since(t);
        uint64_t net_cold_changes = 0, new_cold_rows = 0;
        for (uint32_t i = 0; i < result.registry.size; ++i) {
            if (result.id_device[i] < 0) continue;
            if (i >= previous_rows) ++new_cold_rows;
            else if (previous_owners[i] != result.id_device[i]) ++net_cold_changes;
        }
        // Export is const: hot round-robin output does not bind hot IDs internally.
        t = Clock::now();
        write_output(args, hot_ids, result);
        double export_s = seconds_since(t);
        if (!args.state_out.empty()) {
            t = Clock::now();
            checkpoint(args, hot_ids, result, true);
            write_s = seconds_since(t);
        }
        std::cout << std::setprecision(9)
                  << "STATS simulation_s=" << simulation_s << " export_s=" << export_s
                  << " process_s=" << seconds_since(process_start)
                  << " checkpoint_read_s=" << read_s << " checkpoint_write_s=" << write_s
                  << " checkpoint_read_bytes=" << (args.init_state.empty() ? 0 : fs::file_size(args.init_state))
                  << " checkpoint_write_bytes=" << (args.state_out.empty() ? 0 : fs::file_size(args.state_out))
                  << " start_sample=" << args.start_sample << " next_sample=" << result.next_sample
                  << " origin_sample=" << result.origin_sample
                  << " batches_processed=" << result.batches_processed
                  << " new_registry_rows=" << result.registry.size - previous_rows
                  << " new_cold_rows=" << new_cold_rows
                  << " net_cold_owner_changes=" << net_cold_changes
                  << " migration_events=" << result.migration_events - previous_events
                  << " cumulative_migration_events=" << result.migration_events
                  << " total_registry_rows=" << result.registry.size << '\n';
        std::cout << "Done.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
