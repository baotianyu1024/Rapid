#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <omp.h>

namespace fs = std::filesystem;

/* uint64 编码: 高 32 位 = table_id, 低 32 位 = feature_id */
static inline uint64_t encode_id(int tid, uint32_t fid) {
    return ((uint64_t)(uint32_t)tid << 32) | (uint64_t)fid;
}
static inline uint32_t decode_fid(uint64_t id) { return (uint32_t)(id & 0xFFFFFFFF); }

/* 稀疏列索引 -> table id */
std::vector<int> get_sparse_index_mapping(const std::vector<int>& table_lengths) {
    std::vector<int> m;
    for (size_t t = 0; t < table_lengths.size(); ++t)
        for (int i = 0; i < table_lengths[t]; ++i)
            m.push_back(t);
    return m;
}

/*
 * 从 generate_local_dataset 输出的已转换数据集 (dataset_0.bin ~ dataset_N.bin)
 * 直接计算每个 batch 每张卡的本地命中数和远程访问数。
 *
 * 原理：转换后的 sparse ID 满足 new_id = index * num_devices + device_id，
 * 因此 new_id % num_devices == 当前卡号 → 本地命中，否则 → 远程访问。
 *
 * 用法：
 *   ./verify_hit_rate_from_transformed \
 *     --dataset_dir <已转换数据集目录> \
 *     --num_devices 32 --batch_size 8192 --num_batches 10 \
 *     --dense_count 13 \
 *     --table_lengths 3 2 1 2 6 1 1 1 1 7 3 8 1 6 9 5 1 1 1 12 100 27 10 3 1 1
 */

struct Args {
    std::string dataset_dir;
    int num_devices = 32;
    int batch_size = 8192;
    int num_batches = 10;
    int dense_count = 13;
    bool shuffle_chunk = true;
    std::vector<int> table_lengths = {
        3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
    };

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset_dir") dataset_dir = argv[++i];
            else if (arg == "--num_devices") num_devices = std::stoi(argv[++i]);
            else if (arg == "--batch_size") batch_size = std::stoi(argv[++i]);
            else if (arg == "--num_batches") num_batches = std::stoi(argv[++i]);
            else if (arg == "--dense_count") dense_count = std::stoi(argv[++i]);
            else if (arg == "--no_shuffle") shuffle_chunk = false;
            else if (arg == "--table_lengths") {
                table_lengths.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-')
                    table_lengths.push_back(std::stoi(argv[++i]));
            }
        }
        if (dataset_dir.empty()) {
            std::cerr << "Error: --dataset_dir is required.\n";
            std::exit(1);
        }
    }
};

int main(int argc, char* argv[]) {
    Args args;
    args.parse(argc, argv);

    int total_sparse = std::accumulate(args.table_lengths.begin(), args.table_lengths.end(), 0);
    size_t record_size = 4 + args.dense_count * 4 + total_sparse * 4;
    size_t sparse_offset = 4 + args.dense_count * 4;
    int num_devices = args.num_devices;
    auto sparse_idx_to_tid = get_sparse_index_mapping(args.table_lengths);

    // 每张卡需要读多少样本
    size_t samples_per_device = (size_t)args.batch_size * args.num_batches;
    size_t bytes_per_device = samples_per_device * record_size;

    std::cout << "Config: " << num_devices << " devices, batch_size=" << args.batch_size
              << ", num_batches=" << args.num_batches
              << ", shuffle=" << (args.shuffle_chunk ? "yes" : "no")
              << "\nRecord size: " << record_size << " bytes, total_sparse: " << total_sparse
              << "\nSamples per device: " << samples_per_device << "\n\n";

    // 读取每张卡的数据
    std::cout << "Loading device datasets...\n";
    auto load_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<char>> device_data(num_devices);
    std::vector<size_t> device_sample_counts(num_devices, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int d = 0; d < num_devices; ++d) {
        fs::path path = fs::path(args.dataset_dir) / ("dataset_" + std::to_string(d) + ".bin");
        if (!fs::exists(path)) {
            std::cerr << "Warning: " << path << " not found, skipping device " << d << "\n";
            continue;
        }

        FILE* f = fopen(path.c_str(), "rb");
        if (!f) continue;

        device_data[d].resize(bytes_per_device);
        size_t read_bytes = fread(device_data[d].data(), 1, bytes_per_device, f);
        fclose(f);

        size_t actual_samples = read_bytes / record_size;
        device_data[d].resize(actual_samples * record_size);
        device_sample_counts[d] = actual_samples;
    }

    auto load_end = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(load_end - load_start).count();

    size_t total_loaded = 0;
    for (int d = 0; d < num_devices; ++d) total_loaded += device_sample_counts[d];
    std::cout << "Loaded " << total_loaded << " samples total (" << load_sec << "s)\n\n";

    // Shuffle 每张卡的数据（打乱样本顺序，模拟真实训练）
    if (args.shuffle_chunk) {
        std::cout << "Shuffling device data...\n";
        #pragma omp parallel for schedule(static)
        for (int d = 0; d < num_devices; ++d) {
            size_t count = device_sample_counts[d];
            if (count <= 1) continue;

            // 按 record 粒度 shuffle
            std::vector<size_t> indices(count);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 rng(42 + d);
            std::shuffle(indices.begin(), indices.end(), rng);

            std::vector<char> shuffled(count * record_size);
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(shuffled.data() + i * record_size,
                            device_data[d].data() + indices[i] * record_size,
                            record_size);
            }
            device_data[d] = std::move(shuffled);
        }
        std::cout << "Shuffle done.\n\n";
    }

    // 逐 batch 统计（单次遍历同时收集 hit rate 和 max_remote 指标）
    std::cout << "Computing hit rates per batch...\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Batch   | Local Hits  | Remote Hits | Total Unique | Hit Rate\n";
    std::cout << "------------------------------------------------------------------------\n";

    long long grand_local = 0, grand_remote = 0, grand_total = 0;
    long long overall_max_remote = 0;
    double sum_max_remote = 0;
    int actual_batches = 0;

    for (int batch = 0; batch < args.num_batches; ++batch) {
        size_t sample_start = (size_t)batch * args.batch_size;

        // 并行统计每张卡
        std::vector<long long> dev_local(num_devices, 0);
        std::vector<long long> dev_remote(num_devices, 0);
        std::vector<long long> dev_total(num_devices, 0);

        int any_data = 0;

        #pragma omp parallel for schedule(static) reduction(|:any_data)
        for (int d = 0; d < num_devices; ++d) {
            size_t count = device_sample_counts[d];
            size_t batch_start = sample_start;
            size_t batch_end = std::min(sample_start + (size_t)args.batch_size, count);

            if (batch_start >= count) continue;
            any_data = 1;

            // 收集本 batch 这张卡的所有 unique (table_id, sparse_id)
            std::unordered_set<uint64_t> unique_ids;
            unique_ids.reserve(args.batch_size * total_sparse / 2);

            for (size_t s = batch_start; s < batch_end; ++s) {
                const char* rec = device_data[d].data() + s * record_size;
                const uint32_t* sp = reinterpret_cast<const uint32_t*>(rec + sparse_offset);

                for (int col = 0; col < total_sparse; ++col) {
                    unique_ids.insert(encode_id(sparse_idx_to_tid[col], sp[col]));
                }
            }

            // 统计本地 vs 远程
            long long local_count = 0;
            long long remote_count = 0;
            for (uint64_t eid : unique_ids) {
                uint32_t id_val = decode_fid(eid);
                if ((int)(id_val % (uint32_t)num_devices) == d) {
                    local_count++;
                } else {
                    remote_count++;
                }
            }

            dev_local[d] = local_count;
            dev_remote[d] = remote_count;
            dev_total[d] = local_count + remote_count;
        }

        if (!any_data) break;
        actual_batches++;

        long long batch_local = 0, batch_remote = 0, batch_total = 0;
        for (int d = 0; d < num_devices; ++d) {
            batch_local += dev_local[d];
            batch_remote += dev_remote[d];
            batch_total += dev_total[d];
        }

        grand_local += batch_local;
        grand_remote += batch_remote;
        grand_total += batch_total;

        // max_remote 指标（同时收集，避免重复遍历）
        long long max_r = *std::max_element(dev_remote.begin(), dev_remote.end());
        long long min_r = *std::min_element(dev_remote.begin(), dev_remote.end());
        double avg_r = (double)batch_remote / num_devices;
        if (max_r > overall_max_remote) overall_max_remote = max_r;
        sum_max_remote += max_r;

        double hit_rate = batch_total > 0 ? 100.0 * batch_local / batch_total : 0;
        printf("  %3d   | %10lld  | %10lld  | %11lld  | %6.2f%%\n",
               batch + 1, batch_local, batch_remote, batch_total, hit_rate);

        // Per-device detail for this batch
        for (int d = 0; d < num_devices; ++d) {
            double dev_hit_rate = dev_total[d] > 0 ? 100.0 * dev_local[d] / dev_total[d] : 0;
            printf("        Dev %2d: local=%7lld  remote=%7lld  unique=%7lld  hit=%.2f%%\n",
                   d, dev_local[d], dev_remote[d], dev_total[d], dev_hit_rate);
        }
    }

    std::cout << "------------------------------------------------------------------------\n\n";

    // 汇总
    std::cout << "============================================================\n";
    std::cout << "Summary (" << actual_batches << " batches, " << num_devices << " devices)\n";
    std::cout << "============================================================\n";
    if (grand_total > 0) {
        printf("  Total unique ID requests : %lld\n", grand_total);
        printf("  Local hits               : %lld (%.2f%%)\n",
               grand_local, 100.0 * grand_local / grand_total);
        printf("  Remote accesses          : %lld (%.2f%%)\n",
               grand_remote, 100.0 * grand_remote / grand_total);
        printf("  Avg local hit rate/batch : %.2f%%\n",
               100.0 * grand_local / grand_total);
    } else {
        std::cout << "  No data.\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "Key Metric: max(remote) per batch across devices\n";
    std::cout << "============================================================\n";
    if (actual_batches > 0) {
        printf("  Overall max_remote across all batches: %lld\n", overall_max_remote);
        printf("  Avg max_remote per batch: %.1f\n", sum_max_remote / actual_batches);
    }

    return 0;
}
