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
#include <omp.h>

namespace fs = std::filesystem;

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

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--input_bin") input_bin = argv[++i];
            else if (arg == "--hot_ids_dir") hot_ids_dir = argv[++i];
            else if (arg == "--output_dir") output_dir = argv[++i];
            else if (arg == "--n_batches") n_batches = std::stoi(argv[++i]);
            else if (arg == "--batch_size") batch_size = std::stoi(argv[++i]);
            else if (arg == "--num_devices") num_devices = std::stoi(argv[++i]);
            else if (arg == "--dense_count") dense_count = std::stoi(argv[++i]);
            else if (arg == "--strategy") strategy = std::stoi(argv[++i]);
            else if (arg == "--cap_penalty") cap_penalty = std::stof(argv[++i]);
            else if (arg == "--beta") beta = std::stof(argv[++i]);
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
        }
        if (input_bin.empty() || hot_ids_dir.empty() || output_dir.empty()) {
            std::cerr << "Error: --input_bin, --hot_ids_dir, --output_dir required.\n";
            exit(1);
        }
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
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            if (line.empty()) continue;
            auto pos = line.find(" : ");
            if (pos != std::string::npos)
                hot.insert(encode_id((int)tid, std::stoi(line.substr(0, pos))));
        }
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
    std::vector<int8_t> id_device;    // [idx] -> device
};

SimResult run_simulation(const Args& args,
                         const std::unordered_set<uint64_t>& hot_id_set,
                         const std::vector<IndexEntry>& indices_to_check) {

    int ND = args.num_devices;
    int GLOBAL_BATCH = ND * args.batch_size;
    int num_indices = (int)indices_to_check.size();

    long long total_sparse = std::accumulate(args.table_lengths.begin(), args.table_lengths.end(), 0LL);
    size_t record_size = 4 + args.dense_count * 4 + total_sparse * 4;
    int sparse_off = 4 + args.dense_count * 4;

    bool use_cap_penalty = (args.strategy == 2 || args.strategy == 4);
    bool use_threshold = (args.strategy == 3 || args.strategy == 4);

    // Flat data structures
    IDRegistry registry;
    registry.reserve(80000000); // ~78M expected IDs

    // Pre-allocate for 80M IDs × ND devices
    size_t max_ids = 80000000;
    std::vector<int32_t> counts_flat(max_ids * ND, 0);
    std::vector<int8_t> id_device(max_ids, -1);
    std::vector<int> device_sizes(ND, 0);

    std::vector<char> batch_buffer((size_t)GLOBAL_BATCH * record_size);

    FILE* f = fopen(args.input_bin.c_str(), "rb");
    if (!f) { std::cerr << "Cannot open: " << args.input_bin << "\n"; exit(1); }

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
            size_t new_max = registry.size * 1.2;
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
        int target = (N + ND - 1) / ND;
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
            raw_cold.reserve(N * 50);

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
                if (use_threshold)
                    should = (cnt[best] > (int)(cnt[cur] * args.migrate_threshold));
                else {
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

    fclose(f);

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

    // Build result
    SimResult result;
    result.registry = std::move(registry);
    result.counts_flat = std::move(counts_flat);
    result.id_device = std::move(id_device);
    return result;
}

// --- Write output ---
void write_output(const Args& args,
                  const std::unordered_set<uint64_t>& hot_id_set,
                  SimResult& result) {
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
        [](const ColdEntry& a, const ColdEntry& b) { return a.total_freq > b.total_freq; });

    for (const auto& e : cold_sorted) {
        uint64_t eid = result.registry.idx_to_id[e.idx];
        int tid = decode_tid(eid), fid = decode_fid(eid);
        buckets[e.device].normal.push_back(std::to_string(tid) + ":" + std::to_string(fid));
    }

    fs::create_directories(args.output_dir);
    #pragma omp parallel for schedule(static)
    for (int d = 0; d < ND; ++d) {
        auto path = fs::path(args.output_dir) / ("table_" + std::to_string(d));
        std::ofstream out(path);
        for (const auto& s : buckets[d].priority) out << s << "\n";
        for (const auto& s : buckets[d].normal) out << s << "\n";
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
              << 100.0 * (mx - mn) / avg << "%\n";
}

int main(int argc, char* argv[]) {
    Args args;
    args.parse(argc, argv);

    std::cout << "Strategy: " << args.strategy
              << ", beta: " << args.beta
              << ", cap_penalty: " << args.cap_penalty
              << ", migrate_threshold: " << args.migrate_threshold
              << ", n_batches: " << args.n_batches
              << ", OMP threads: " << omp_get_max_threads() << "\n";

    auto hot_ids = load_hot_ids(args.hot_ids_dir, args.excluded_tables, args.table_lengths);
    auto indices = get_indices_to_check(args.table_lengths, args.excluded_tables);

    auto result = run_simulation(args, hot_ids, indices);

    write_output(args, hot_ids, result);
    std::cout << "Done.\n";
    return 0;
}
