/**
 * Evaluate partition quality WITH optimal sample-to-device assignment.
 *
 * For each sample in a batch, assign it to the device that has the most of its IDs.
 * Then measure locality = fraction of (sample, ID) pairs where ID is on sample's device.
 * Also enforce balance: each device gets exactly batch_size samples per global batch.
 *
 * This simulates the real training scenario where sample placement is optimized.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

// g++ -O3 -std=c++17 -fopenmp -pthread evaluate_with_sample_assignment.cpp -o evaluate_with_sample_assignment

static inline uint64_t encode_id(int tid, uint32_t fid) {
    return ((uint64_t)(uint32_t)tid << 32) | (uint64_t)fid;
}

struct Args {
    std::string input_bin;
    std::string assignment_dir;
    int num_devices = 8;
    int batch_size = 8192;
    int num_batches = 200;
    int dense_count = 13;
    std::vector<int> table_lengths = {3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1};

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--input") input_bin = argv[++i];
            else if (arg == "--assignment_dir") assignment_dir = argv[++i];
            else if (arg == "--num_devices") num_devices = std::stoi(argv[++i]);
            else if (arg == "--batch_size") batch_size = std::stoi(argv[++i]);
            else if (arg == "--num_batches") num_batches = std::stoi(argv[++i]);
            else if (arg == "--dense_count") dense_count = std::stoi(argv[++i]);
            else if (arg == "--table_lengths") {
                table_lengths.clear();
                while (i + 1 < argc && argv[i+1][0] != '-')
                    table_lengths.push_back(std::stoi(argv[++i]));
            }
        }
    }
};

int main(int argc, char* argv[]) {
    Args args;
    args.parse(argc, argv);

    int total_sparse = std::accumulate(args.table_lengths.begin(), args.table_lengths.end(), 0);
    size_t record_size = 4 + args.dense_count * 4 + total_sparse * 4;
    int sparse_off = 4 + args.dense_count * 4;
    int ND = args.num_devices;
    int GLOBAL_BATCH = ND * args.batch_size;

    // col_to_table mapping
    std::vector<int> col_to_table(total_sparse);
    {
        int off = 0;
        for (size_t t = 0; t < args.table_lengths.size(); ++t) {
            for (int j = 0; j < args.table_lengths[t]; ++j)
                col_to_table[off++] = (int)t;
        }
    }

    // Load assignment
    std::cout << "Loading assignment from: " << args.assignment_dir << std::endl;
    std::unordered_map<uint64_t, int8_t> assignment;
    assignment.reserve(60000000);
    std::vector<long long> dev_id_counts(ND, 0);

    for (int d = 0; d < ND; ++d) {
        auto path = fs::path(args.assignment_dir) / ("table_" + std::to_string(d));
        if (!fs::exists(path)) continue;
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto pos = line.find(':');
            if (pos == std::string::npos) continue;
            int tid = std::stoi(line.substr(0, pos));
            uint32_t fid = (uint32_t)std::stoul(line.substr(pos + 1));
            assignment[encode_id(tid, fid)] = (int8_t)d;
            dev_id_counts[d]++;
        }
    }
    std::cout << "Assignment loaded: " << assignment.size() << " IDs" << std::endl;

    // Balance of ID assignment
    double avg_ids = (double)assignment.size() / ND;
    long long max_ids = *std::max_element(dev_id_counts.begin(), dev_id_counts.end());
    double id_balance = avg_ids > 0 ? (double)max_ids / avg_ids : 0;
    std::cout << "ID Balance: " << id_balance << std::endl;

    // Evaluate with sample assignment
    FILE* f = fopen(args.input_bin.c_str(), "rb");
    if (!f) { std::cerr << "Cannot open " << args.input_bin << std::endl; return 1; }

    std::vector<char> buffer((size_t)GLOBAL_BATCH * record_size);
    long long total_accesses = 0, local_hits = 0;
    long long total_unique_local = 0, total_unique_remote = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < args.num_batches; ++batch) {
        size_t n = fread(buffer.data(), record_size, GLOBAL_BATCH, f);
        if ((int)n < GLOBAL_BATCH) break;

        // Phase 1: For each sample, compute hit count per device
        // hit_counts[i * ND + d] = number of IDs in sample i that are assigned to device d
        std::vector<int> hit_counts((size_t)GLOBAL_BATCH * ND, 0);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < GLOBAL_BATCH; ++i) {
            const char* rec = buffer.data() + (size_t)i * record_size;
            const int32_t* sp = reinterpret_cast<const int32_t*>(rec + sparse_off);
            int* hc = hit_counts.data() + (size_t)i * ND;

            for (int col = 0; col < total_sparse; ++col) {
                int tid = col_to_table[col];
                uint32_t fid = (uint32_t)sp[col];
                uint64_t eid = encode_id(tid, fid);
                auto it = assignment.find(eid);
                if (it != assignment.end()) {
                    hc[it->second]++;
                }
            }
        }

        // Phase 2: Greedy balanced assignment
        // Each device gets exactly batch_size samples
        // Sort samples by their max hit count (descending), then greedily assign
        struct SampleScore {
            int idx;
            int best_dev;
            int best_hits;
        };
        std::vector<SampleScore> scores(GLOBAL_BATCH);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < GLOBAL_BATCH; ++i) {
            const int* hc = hit_counts.data() + (size_t)i * ND;
            int best_d = 0, best_h = hc[0];
            for (int d = 1; d < ND; ++d) {
                if (hc[d] > best_h) { best_h = hc[d]; best_d = d; }
            }
            scores[i] = {i, best_d, best_h};
        }

        // Sort by best_hits descending (samples with strongest preference first)
        std::sort(scores.begin(), scores.end(),
            [](const SampleScore& a, const SampleScore& b) { return a.best_hits > b.best_hits; });

        // Greedy assignment with capacity
        std::vector<int> dev_cap(ND, 0);
        std::vector<int> sample_device(GLOBAL_BATCH);
        int target = args.batch_size;

        for (auto& s : scores) {
            const int* hc = hit_counts.data() + (size_t)s.idx * ND;

            // Try best device first
            int assigned = -1;
            if (dev_cap[s.best_dev] < target) {
                assigned = s.best_dev;
            } else {
                // Find device with most hits that still has capacity
                int best_d = -1, best_h = -1;
                for (int d = 0; d < ND; ++d) {
                    if (dev_cap[d] < target && hc[d] > best_h) {
                        best_h = hc[d]; best_d = d;
                    }
                }
                assigned = best_d;
            }
            if (assigned < 0) {
                // All full? shouldn't happen with exact sizing, fallback
                for (int d = 0; d < ND; ++d)
                    if (dev_cap[d] < target) { assigned = d; break; }
            }
            sample_device[s.idx] = assigned;
            dev_cap[assigned]++;
        }

        // Phase 3: Compute locality with the assigned devices
        long long batch_local = 0, batch_total = 0;

        #pragma omp parallel for reduction(+:batch_local, batch_total) schedule(static)
        for (int i = 0; i < GLOBAL_BATCH; ++i) {
            int my_dev = sample_device[i];
            const char* rec = buffer.data() + (size_t)i * record_size;
            const int32_t* sp = reinterpret_cast<const int32_t*>(rec + sparse_off);

            for (int col = 0; col < total_sparse; ++col) {
                int tid = col_to_table[col];
                uint32_t fid = (uint32_t)sp[col];
                uint64_t eid = encode_id(tid, fid);
                auto it = assignment.find(eid);
                batch_total++;
                if (it != assignment.end() && it->second == my_dev) {
                    batch_local++;
                }
            }
        }

        total_accesses += batch_total;
        local_hits += batch_local;

        if ((batch + 1) % 20 == 0) {
            double loc = 100.0 * local_hits / total_accesses;
            std::cout << "  Batch " << batch + 1 << "/" << args.num_batches
                      << ", locality: " << loc << "%" << std::endl;
        }
    }
    fclose(f);

    auto t_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(t_end - t_start).count();

    double locality = 100.0 * local_hits / total_accesses;

    std::cout << "\n=== Results (with sample assignment) ===" << std::endl;
    std::cout << "Locality: " << locality << "%" << std::endl;
    std::cout << "Balance factor: " << id_balance << std::endl;
    std::cout << "Total IDs: " << assignment.size() << std::endl;
    std::cout << "Eval time: " << eval_time << "s" << std::endl;
    std::cout << "IDs per device:";
    for (int d = 0; d < ND; ++d) std::cout << " " << dev_id_counts[d];
    std::cout << std::endl;

    return 0;
}
