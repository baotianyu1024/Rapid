/**
 * Fast partition evaluation tool.
 * Reads a binary dataset and an assignment file, computes locality metrics.
 *
 * Assignment file format: binary file of int8_t[max_encoded_id]
 *   where assignment[encode(tid, fid)] = device_id (-1 if unassigned)
 *
 * Or text format: one line per ID "table_id:feature_id device_id"
 *
 * Usage:
 *   ./evaluate_partition --input <bin> --assignment_dir <dir> --num_devices 8
 *       --batch_size 8192 --num_batches 200 --dense_count 13
 *       --table_lengths 3 2 1 ...
 *
 * assignment_dir contains files table_0 ... table_{N-1}, each with lines "tid:fid"
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
#include <sstream>
#include <omp.h>

namespace fs = std::filesystem;

// g++ -O3 -std=c++17 -fopenmp -pthread evaluate_partition.cpp -o evaluate_partition

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

    // Build table index mapping: sparse_col_idx -> table_id
    std::vector<int> col_to_table(total_sparse);
    {
        int off = 0;
        for (size_t t = 0; t < args.table_lengths.size(); ++t) {
            for (int j = 0; j < args.table_lengths[t]; ++j)
                col_to_table[off++] = (int)t;
        }
    }

    // Load assignment: table_d files -> unordered_map
    std::cout << "Loading assignment from: " << args.assignment_dir << std::endl;
    std::unordered_map<uint64_t, int8_t> assignment;
    assignment.reserve(60000000);

    std::vector<long long> dev_counts(ND, 0);
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
            dev_counts[d]++;
        }
    }
    std::cout << "Assignment loaded: " << assignment.size() << " IDs" << std::endl;

    // Balance
    double avg = (double)assignment.size() / ND;
    long long mx = *std::max_element(dev_counts.begin(), dev_counts.end());
    long long mn = *std::min_element(dev_counts.begin(), dev_counts.end());
    double balance = avg > 0 ? (double)mx / avg : 0;
    std::cout << "Balance: " << balance << " (min=" << mn << ", max=" << mx << ", avg=" << (long long)avg << ")" << std::endl;

    // Evaluate locality
    FILE* f = fopen(args.input_bin.c_str(), "rb");
    if (!f) { std::cerr << "Cannot open " << args.input_bin << std::endl; return 1; }

    std::vector<char> buffer((size_t)GLOBAL_BATCH * record_size);
    long long total_accesses = 0, local_hits = 0;
    std::vector<long long> cross_per_batch;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < args.num_batches; ++batch) {
        size_t n = fread(buffer.data(), record_size, GLOBAL_BATCH, f);
        if ((int)n < GLOBAL_BATCH) break;

        long long batch_local = 0, batch_total = 0;

        #pragma omp parallel for reduction(+:batch_local, batch_total) schedule(static)
        for (int i = 0; i < GLOBAL_BATCH; ++i) {
            int device = i / args.batch_size;
            const char* rec = buffer.data() + (size_t)i * record_size;
            const int32_t* sp = reinterpret_cast<const int32_t*>(rec + sparse_off);

            for (int col = 0; col < total_sparse; ++col) {
                int tid = col_to_table[col];
                uint32_t fid = (uint32_t)sp[col];
                uint64_t eid = encode_id(tid, fid);

                auto it = assignment.find(eid);
                if (it != assignment.end() && it->second == device) {
                    batch_local++;
                }
                batch_total++;
            }
        }

        total_accesses += batch_total;
        local_hits += batch_local;
        cross_per_batch.push_back(batch_total - batch_local);

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
    double avg_cross = 0;
    for (auto c : cross_per_batch) avg_cross += c;
    avg_cross /= cross_per_batch.size();

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Locality: " << locality << "%" << std::endl;
    std::cout << "Balance factor: " << balance << std::endl;
    std::cout << "Avg cross-device accesses/batch: " << (long long)avg_cross << std::endl;
    std::cout << "Total IDs: " << assignment.size() << std::endl;
    std::cout << "Eval time: " << eval_time << "s" << std::endl;
    std::cout << "IDs per device:";
    for (int d = 0; d < ND; ++d) std::cout << " " << dev_counts[d];
    std::cout << std::endl;

    return 0;
}
