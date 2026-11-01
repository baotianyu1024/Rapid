#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <filesystem>
#include <cstring>
#include <thread>
#include <future>
#include <mutex>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <cstdint>
#include <limits>
#include <omp.h>
#include "swap_optimizer.h"

namespace fs = std::filesystem;
static std::mutex log_mutex;

// g++ -O3 -std=c++17 -fopenmp -pthread generate_local_dataset_v2.cpp -o generate_local_dataset

/* =========================
 * uint64 编码替代 IDPair
 * ========================= */

static inline uint64_t encode_id(int tid, uint32_t fid) {
    return ((uint64_t)(uint32_t)tid << 32) | (uint64_t)fid;
}
static inline int decode_tid(uint64_t id) { return (int)(id >> 32); }
static inline uint32_t decode_fid(uint64_t id) { return (uint32_t)(id & 0xFFFFFFFF); }

/* =========================
 * 参数
 * ========================= */

struct Args {
    std::string input_file;
    std::string table_files_dir;
    std::string output_dir;

    int num_devices = 8;
    int train_batch_size = 8192;
    int steps_per_chunk = 1;
    int num_chunks = 20;
    int num_threads = 1;
    long long start_sample_idx = 0;
    int dense_count = 13;
    float gamma = 1.0f;
    std::string scheduling = "ours";
    bool hybrid_swap = false;
    cps_swap::Config swap_config;

    std::vector<int> excluded_tables;
    std::vector<int> reduction_tables;
    std::vector<int> table_lengths = {
        3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,
        100,27,10,3,1,1
    };

    std::string read_swap_value(int& i, int argc, char* argv[]) {
        cps_swap::require(i + 1 < argc, "Missing swap parameter value");
        const std::string value = argv[++i];
        cps_swap::require(!value.empty() && value.rfind("--", 0) != 0, "Missing swap parameter value");
        return value;
    }

    uint64_t read_swap_budget(int& i, int argc, char* argv[]) {
        const std::string value = read_swap_value(i, argc, argv);
        uint64_t number = 0;
        for (const char ch : value) {
            cps_swap::require(ch >= '0' && ch <= '9', "Swap parameters require unsigned decimal integers");
            const uint64_t digit = static_cast<unsigned>(ch - '0');
            cps_swap::require(number <= (std::numeric_limits<uint64_t>::max() - digit) / 10,
                              "Swap parameter exceeds uint64");
            number = number * 10 + digit;
        }
        return number;
    }

    uint32_t read_swap_parameter(int& i, int argc, char* argv[]) {
        const uint64_t number = read_swap_budget(i, argc, argv);
        cps_swap::require(number <= UINT32_MAX, "Swap parameter exceeds uint32");
        return static_cast<uint32_t>(number);
    }

    void parse(int argc, char* argv[]) {
        std::unordered_set<std::string> swap_seen;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--peak_weight" || arg.rfind("--swap_", 0) == 0)
                cps_swap::require(swap_seen.insert(arg).second, "Repeated swap parameter");
            if (arg == "--input_file") input_file = argv[++i];
            else if (arg == "--table_files_dir") table_files_dir = argv[++i];
            else if (arg == "--output_dir") output_dir = argv[++i];
            else if (arg == "--num_devices") num_devices = std::stoi(argv[++i]);
            else if (arg == "--train_batch_size") train_batch_size = std::stoi(argv[++i]);
            else if (arg == "--steps_per_chunk") steps_per_chunk = std::stoi(argv[++i]);
            else if (arg == "--num_chunks") num_chunks = std::stoi(argv[++i]);
            else if (arg == "--num_threads") num_threads = std::stoi(argv[++i]);
            else if (arg == "--dense_count") dense_count = std::stoi(argv[++i]);
            else if (arg == "--start_sample_idx") start_sample_idx = std::stoll(argv[++i]);
            else if (arg == "--gamma") gamma = std::stof(argv[++i]);
            else if (arg == "--scheduling") scheduling = argv[++i];
            else if (arg == "--peak_weight") swap_config.weight = read_swap_parameter(i, argc, argv);
            else if (arg == "--swap_rounds") swap_config.rounds = read_swap_parameter(i, argc, argv);
            else if (arg == "--swap_source_samples") swap_config.source_samples = read_swap_parameter(i, argc, argv);
            else if (arg == "--swap_candidates") swap_config.candidates = read_swap_parameter(i, argc, argv);
            else if (arg == "--swap_targets") swap_config.targets = read_swap_parameter(i, argc, argv);
            else if (arg == "--swap_evaluations_per_gpu") swap_config.evaluations_per_gpu = read_swap_budget(i, argc, argv);
            else if (arg == "--swap_algorithm") {
                const std::string value = read_swap_value(i, argc, argv);
                cps_swap::require(value == "weighted" || value == "ratio" || value == "hybrid",
                                  "--swap_algorithm must be weighted, ratio, or hybrid");
                hybrid_swap = value == "hybrid";
                swap_config.algorithm = (value == "ratio" || hybrid_swap)
                    ? cps_swap::Algorithm::Ratio : cps_swap::Algorithm::Weighted;
            }
            else if (arg == "--swap_target_numerator") swap_config.target_numerator = read_swap_parameter(i, argc, argv);
            else if (arg == "--swap_target_denominator") swap_config.target_denominator = read_swap_parameter(i, argc, argv);
            else if (arg == "--swap_cycles") {
                const std::string value = read_swap_value(i, argc, argv);
                cps_swap::require(value == "0" || value == "1", "--swap_cycles must be 0 or 1");
                swap_config.cycles = value == "1";
            }
            else if (arg == "--swap_stop_at_target") {
                const std::string value = read_swap_value(i, argc, argv);
                cps_swap::require(value == "0" || value == "1", "--swap_stop_at_target must be 0 or 1");
                swap_config.stop_at_target = value == "1";
            }
            else if (arg == "--swap_screen_queries_per_gpu") swap_config.screen_queries_per_gpu = read_swap_budget(i, argc, argv);
            else if (arg.rfind("--swap_", 0) == 0)
                throw std::runtime_error("Unknown swap parameter: " + arg);
            else if (arg == "--reduction_tables") {
                reduction_tables.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-')
                    reduction_tables.push_back(std::stoi(argv[++i]));
            }
            else if (arg == "--excluded_tables") {
                excluded_tables.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-') {
                    excluded_tables.push_back(std::stoi(argv[++i]));
                }
            }
            else if (arg == "--table_lengths") {
                table_lengths.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-') {
                    table_lengths.push_back(std::stoi(argv[++i]));
                }
            }
        }

        if (swap_config.algorithm == cps_swap::Algorithm::Ratio) {
            cps_swap::require(hybrid_swap || !swap_seen.count("--peak_weight"),
                              "--peak_weight cannot be used with --swap_algorithm ratio");
            if (!swap_seen.count("--swap_source_samples")) swap_config.source_samples = 64;
            if (!swap_seen.count("--swap_candidates")) swap_config.candidates = 16;
            if (!swap_seen.count("--swap_targets")) swap_config.targets = 4;
            if (!swap_seen.count("--swap_rounds")) swap_config.rounds = 8;
            if (!swap_seen.count("--swap_evaluations_per_gpu")) swap_config.evaluations_per_gpu = 125000;
            cps_swap::require(swap_config.target_denominator >= 1 && swap_config.target_denominator <= 1000 &&
                              swap_config.target_numerator >= swap_config.target_denominator &&
                              swap_config.target_numerator <= 2 * swap_config.target_denominator,
                              "Ratio target must satisfy 1 <= denominator <= 1000 and denominator <= numerator <= 2*denominator");
        } else {
            for (const char* name : {"--swap_target_numerator", "--swap_target_denominator",
                                     "--swap_cycles", "--swap_screen_queries_per_gpu", "--swap_stop_at_target"})
                if (swap_seen.count(name))
                    throw std::runtime_error(std::string(name) + " requires --swap_algorithm ratio");
        }
        cps_swap::require(scheduling == "ours" || scheduling == "sequential" ||
                          scheduling == "greedy_repair", "Unknown scheduling mode");
        cps_swap::require(swap_config.source_samples && swap_config.candidates && swap_config.targets,
                          "Swap search sizes must be positive");
        if (input_file.empty() || table_files_dir.empty() || output_dir.empty()) {
            std::cerr << "Missing required arguments\n";
            std::exit(1);
        }
        cps_swap::require(num_devices > 0 && train_batch_size > 0 && num_chunks > 0,
                          "GPU, batch and chunk counts must be positive");
        cps_swap::require(swap_config.evaluations_per_gpu <=
                              std::numeric_limits<uint64_t>::max() / static_cast<uint32_t>(num_devices),
                          "Swap evaluation budget overflows uint64");
        if (swap_config.algorithm == cps_swap::Algorithm::Ratio)
            cps_swap::require(swap_config.screen_queries_per_gpu <=
                                  std::numeric_limits<uint64_t>::max() / static_cast<uint32_t>(num_devices),
                              "Swap screening budget overflows uint64");
        cps_swap::require(steps_per_chunk == 1 && num_threads > 0,
                          "Peak optimization requires one global batch per chunk and positive async workers");
        cps_swap::require(excluded_tables.empty(), "Peak optimization must include all tables");
        std::unordered_set<int> reduction_seen;
        for (int table : reduction_tables) {
            cps_swap::require(table >= 0 && size_t(table) < table_lengths.size(), "Invalid Reduction table");
            cps_swap::require(reduction_seen.insert(table).second, "Repeated Reduction table");
        }
        cps_swap::require(reduction_tables.size() < table_lengths.size(), "Scheduling requires at least one Unique table");
        cps_swap::require(start_sample_idx >= 0, "Negative input sample offset");
        cps_swap::require(dense_count >= 0 && !table_lengths.empty(), "Invalid record layout");
        uint64_t columns = 0;
        for (int length : table_lengths) {
            cps_swap::require(length > 0, "Table pooling lengths must be positive");
            columns += static_cast<uint32_t>(length);
        }
        cps_swap::require(columns <= INT32_MAX && uint64_t(num_devices) * train_batch_size <= INT32_MAX,
                          "Record or batch dimensions exceed supported indices");
    }
};

/* =========================
 * 映射结果（uint64 编码）
 * ========================= */

struct MappingResult {
    std::unordered_map<uint64_t, uint32_t> global_id_map;
    std::unordered_map<uint64_t, int> id_to_device;
};

/* =========================
 * 构建映射
 * ========================= */

MappingResult load_and_build_mapping(const Args& args) {
    MappingResult res;
    std::cout << "[Init] Loading ID distribution from " << args.table_files_dir << "...\n";

    long long loaded_count = 0;
    for (int d = 0; d < args.num_devices; ++d) {
        fs::path p = fs::path(args.table_files_dir) / ("table_" + std::to_string(d));
        if (!fs::exists(p)) continue;

        std::ifstream fin(p);
        std::string line;
        while (std::getline(fin, line)) {
            size_t pos = line.find(':');
            if (pos == std::string::npos) continue;
            try {
                int tid = std::stoi(line.substr(0, pos));
                uint32_t fid = static_cast<uint32_t>(std::stoul(line.substr(pos + 1)));
                uint64_t eid = encode_id(tid, fid);
                res.id_to_device[eid] = d;
                loaded_count++;
            } catch (...) {}
        }
    }
    std::cout << "[Init] Loaded " << loaded_count << " entries from table files.\n";

    // 按 table 分组，排序后构建交织映射
    std::map<int, std::vector<std::pair<uint64_t, int>>> table_group;
    for (auto& [eid, dev] : res.id_to_device) {
        table_group[decode_tid(eid)].push_back({eid, dev});
    }

    for (auto& [tid, vec] : table_group) {
        std::sort(vec.begin(), vec.end());
        // 按 device 分组计数，使用 per-device local index 避免 TABLE_SIZE 膨胀
        std::vector<int> device_counter(args.num_devices, 0);
        for (auto& [eid, dev] : vec) {
            int local_idx = device_counter[dev]++;
            res.global_id_map[eid] = static_cast<uint32_t>(local_idx * args.num_devices + dev);
        }
    }

    // 预 rehash 到合理容量，减少后续查找的碰撞
    res.global_id_map.reserve(res.global_id_map.size() * 2);
    res.id_to_device.reserve(res.id_to_device.size() * 2);

    std::cout << "[Init] Mapping ready. Total IDs: " << res.global_id_map.size() << "\n";

    // 输出每张表的最大重映射 ID（用于设置训练的 TABLE_SIZE_ARRAY）
    std::cout << "[Init] Max remapped ID per table (for TABLE_SIZE_ARRAY):\n";
    for (auto& [tid, vec] : table_group) {
        // 统计每个 device 上该表的 ID 数
        std::vector<int> dev_counts(args.num_devices, 0);
        for (auto& [eid, dev] : vec) dev_counts[dev]++;
        int max_per_dev = *std::max_element(dev_counts.begin(), dev_counts.end());
        uint32_t max_id = (uint32_t)((max_per_dev - 1) * args.num_devices + (args.num_devices - 1));
        std::cout << "  Table " << tid << ": " << vec.size()
                  << " IDs, max_per_device=" << max_per_dev
                  << ", max_remapped_id=" << max_id
                  << ", required table_size >= " << (max_id + 1) << "\n";
    }

    return res;
}

/* =========================
 * 稀疏 index -> table id（flat array，cache 友好）
 * ========================= */

std::vector<int> get_sparse_index_mapping(const std::vector<int>& table_lengths) {
    std::vector<int> mapping;
    for (size_t t = 0; t < table_lengths.size(); ++t)
        for (int i = 0; i < table_lengths[t]; ++i)
            mapping.push_back((int)t);
    return mapping;
}

/* =========================
 * 轻量样本视图（无 unordered_set）
 * ========================= */

struct SampleView {
    const char* data_ptr;
    // 存储去重后的 unique encoded IDs（仅 checked 列）
    std::vector<uint64_t> unique_ids;
    // hits[d] = 该样本在 device d 上的命中数
    std::vector<int> hits;
};

/* =========================
 * 解析一个 chunk 的所有样本（OpenMP 并行）
 * ========================= */

void parse_samples_parallel(
    const char* raw_data,
    size_t num_samples,
    size_t record_size,
    size_t sparse_offset,
    const std::vector<std::pair<int, int>>& checked_indices,
    std::vector<SampleView>& samples)
{
    samples.resize(num_samples);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_samples; ++i) {
        SampleView& s = samples[i];
        s.data_ptr = raw_data + i * record_size;

        const uint32_t* sp = reinterpret_cast<const uint32_t*>(s.data_ptr + sparse_offset);

        // 用小 set 做去重，然后导出为 vector
        std::unordered_set<uint64_t> tmp_set;
        tmp_set.reserve(checked_indices.size());
        for (auto& [col_idx, tid] : checked_indices) {
            tmp_set.insert(encode_id(tid, sp[col_idx]));
        }
        s.unique_ids.assign(tmp_set.begin(), tmp_set.end());
    }
}

/* =========================
 * 计算每个样本的 hits 向量（OpenMP 并行）
 * ========================= */

void compute_hits_parallel(
    std::vector<SampleView>& samples,
    const std::unordered_map<uint64_t, int>& id_to_device,
    int num_devices)
{
    size_t num_samples = samples.size();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_samples; ++i) {
        SampleView& s = samples[i];
        s.hits.assign(num_devices, 0);
        for (uint64_t eid : s.unique_ids) {
            auto it = id_to_device.find(eid);
            if (it != id_to_device.end()) {
                s.hits[it->second]++;
            }
        }
    }
}

static constexpr float BALANCE_ALPHA = 1.0f;

std::vector<std::vector<int>>
balance_chunk(std::vector<SampleView>& samples,
              int num_devices,
              int target,
              float gamma)
{
    size_t num_samples = samples.size();

    // 先打乱顺序
    std::mt19937 rng(42);
    std::vector<int> order(num_samples);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<std::vector<int>> buckets(num_devices);
    for (int d = 0; d < num_devices; ++d) buckets[d].reserve(target + 1000);
    std::vector<int> load(num_devices, 0);

    for (int idx : order) {
        const auto& hits = samples[idx].hits;

        float best_score = -1e18f;
        int best = -1;

        for (int d = 0; d < num_devices; ++d) {
            if (load[d] >= target) continue;
            float score = BALANCE_ALPHA * hits[d]
                        - gamma * (float)load[d];
            if (score > best_score) {
                best_score = score;
                best = d;
            }
        }
        if (best < 0)
            best = (int)(std::min_element(load.begin(), load.end()) - load.begin());

        buckets[best].push_back(idx);
        load[best]++;
    }
    return buckets;
}

/* =========================
 * ID 转换 + 写入磁盘（并行转换，按设备写入）
 * ========================= */

void transform_and_write(
    int chunk_id,
    const std::vector<SampleView>& samples,
    const std::vector<std::vector<int>>& buckets,
    const Args& args,
    const MappingResult& mapping,
    const std::vector<int>& sparse_idx_to_tid,
    size_t record_size,
    size_t sparse_offset,
    std::vector<FILE*>& output_files,
    std::vector<std::unique_ptr<std::mutex>>& mutexes,
    std::atomic<long long>& unmapped_count)
{
    int num_devices = args.num_devices;
    size_t total_sparse = sparse_idx_to_tid.size();
    std::atomic<bool> write_failed{false};

    // 并行处理每个设备的 bucket
    #pragma omp parallel for schedule(dynamic)
    for (int d = 0; d < num_devices; ++d) {
        const auto& bucket = buckets[d];
        if (bucket.empty()) continue;

        // 每个线程有自己的 buffer
        std::vector<char> buf(bucket.size() * record_size);
        char* write_ptr = buf.data();
        long long local_unmapped = 0;

        for (int sample_idx : bucket) {
            const SampleView& s = samples[sample_idx];
            std::memcpy(write_ptr, s.data_ptr, record_size);

            uint32_t* sp = reinterpret_cast<uint32_t*>(write_ptr + sparse_offset);

            for (size_t col = 0; col < total_sparse; ++col) {
                uint64_t eid = encode_id(sparse_idx_to_tid[col], sp[col]);
                auto it = mapping.global_id_map.find(eid);
                if (it != mapping.global_id_map.end()) {
                    sp[col] = it->second;
                } else {
                    local_unmapped++;
                }
            }
            write_ptr += record_size;
        }

        unmapped_count.fetch_add(local_unmapped);

        // Completion order may differ from input order. The mutex protects the
        // FILE position; the explicit offset preserves the global-batch order.
        {
            std::lock_guard<std::mutex> lk(*mutexes[d]);
            const uint64_t stride = uint64_t(args.train_batch_size) * record_size;
            if (uint64_t(chunk_id) > uint64_t(std::numeric_limits<off_t>::max()) / stride ||
                fseeko(output_files[d], off_t(uint64_t(chunk_id) * stride), SEEK_SET) != 0 ||
                fwrite(buf.data(), 1, buf.size(), output_files[d]) != buf.size())
                write_failed.store(true);
        }
    }
    cps_swap::require(!write_failed.load(), "Cannot write complete output shard");
}

/* =========================
 * 单 chunk 处理（整合上述优化）
 * ========================= */

void process_chunk(
    int chunk_id,
    std::shared_ptr<std::vector<char>> raw,
    const Args& args,
    const MappingResult& mapping,
    const std::vector<std::pair<int, int>>& checked_indices,
    const std::vector<int>& sparse_idx_to_tid,
    size_t record_size,
    size_t sparse_offset,
    std::vector<FILE*>& output_files,
    std::vector<std::unique_ptr<std::mutex>>& mutexes,
    std::atomic<long long>& counter,
    std::atomic<long long>& unmapped_total)
{
    // Divide the OpenMP budget among independent batch workers. The search
    // within each batch stays serial, with unchanged seeds and decisions.
    omp_set_num_threads(std::max(1, omp_get_max_threads() / args.num_threads));
    size_t num_samples = raw->size() / record_size;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 阶段 1：并行解析样本
    std::vector<SampleView> samples;
    parse_samples_parallel(raw->data(), num_samples, record_size,
                           sparse_offset, checked_indices, samples);

    auto t1 = std::chrono::high_resolution_clock::now();

    // 阶段 2：并行计算 hits
    compute_hits_parallel(samples, mapping.id_to_device, args.num_devices);

    auto t2 = std::chrono::high_resolution_clock::now();

    cps_swap::require(num_samples == size_t(args.train_batch_size) * args.num_devices,
                      "Peak optimization requires a complete global batch");
    std::vector<std::vector<int>> buckets(args.num_devices);
    if (args.scheduling == "ours") {
        buckets = balance_chunk(samples, args.num_devices, args.train_batch_size, args.gamma);
    } else if (args.scheduling == "sequential") {
        for (size_t i = 0; i < num_samples; ++i)
            buckets[i / args.train_batch_size].push_back(static_cast<int>(i));
    } else {
        // Paper Table 3: unconstrained Max-Hit, then repair overfull ranks.
        // Deterministic ties: lowest rank; overflow: original sample order.
        for (size_t i = 0; i < num_samples; ++i) {
            const auto& h = samples[i].hits;
            int d = static_cast<int>(std::max_element(h.begin(), h.end()) - h.begin());
            buckets[d].push_back(static_cast<int>(i));
        }
        std::vector<int> overflow;
        for (auto& bucket : buckets) {
            if (bucket.size() > static_cast<size_t>(args.train_batch_size)) {
                overflow.insert(overflow.end(), bucket.begin() + args.train_batch_size, bucket.end());
                bucket.resize(args.train_batch_size);
            }
        }
        std::sort(overflow.begin(), overflow.end());
        for (int i : overflow) {
            int best = -1;
            for (int d = 0; d < args.num_devices; ++d) {
                if (buckets[d].size() >= static_cast<size_t>(args.train_batch_size)) continue;
                if (best < 0 || samples[i].hits[d] > samples[i].hits[best]) best = d;
            }
            cps_swap::require(best >= 0, "Repair could not find free capacity");
            buckets[best].push_back(i);
        }
    }
    if (args.scheduling == "ours" && args.swap_config.rounds > 0) {
        cps_swap::Samples swap_samples(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            auto& ids = swap_samples[i];
            ids.reserve(samples[i].unique_ids.size());
            for (uint64_t key : samples[i].unique_ids) {
                const auto owner = mapping.id_to_device.find(key);
                cps_swap::require(owner != mapping.id_to_device.end(), "Unmapped ID in peak optimization");
                ids.push_back({key, static_cast<uint32_t>(owner->second)});
            }
            cps_swap::canonicalize(ids);
        }
        cps_swap::Buckets initial(args.num_devices);
        for (int d = 0; d < args.num_devices; ++d)
            initial[d].assign(buckets[d].begin(), buckets[d].end());
        cps_swap::Result optimized;
        if (args.hybrid_swap) {
            auto weighted_config = args.swap_config;
            weighted_config.algorithm = cps_swap::Algorithm::Weighted;
            const auto weighted = cps_swap::optimize(swap_samples, initial, weighted_config);
            auto ratio_config = args.swap_config;
            ratio_config.algorithm = cps_swap::Algorithm::Ratio;
            optimized = cps_swap::optimize(swap_samples, weighted.buckets, ratio_config);
            optimized.before = weighted.before;
            optimized.accepted += weighted.accepted;
            optimized.evaluations += weighted.evaluations;
        } else {
            optimized = cps_swap::optimize(swap_samples, initial, args.swap_config);
        }
        for (int d = 0; d < args.num_devices; ++d)
            buckets[d].assign(optimized.buckets[d].begin(), optimized.buckets[d].end());
        const auto before_sum = std::accumulate(optimized.before.begin(), optimized.before.end(), uint64_t(0));
        const auto after_sum = std::accumulate(optimized.after.begin(), optimized.after.end(), uint64_t(0));
        const uint64_t before_peak = *std::max_element(optimized.before.begin(), optimized.before.end());
        const uint64_t after_peak = *std::max_element(optimized.after.begin(), optimized.after.end());
        std::lock_guard<std::mutex> log_lock(log_mutex);
        std::cout << "  Swap chunk " << chunk_id << " total=" << before_sum << "->" << after_sum
                  << " peak=" << before_peak << "->" << after_peak
                  << " accepted=" << optimized.accepted << " evaluations=" << optimized.evaluations;
        if (args.swap_config.algorithm == cps_swap::Algorithm::Ratio) {
            const auto& stats = optimized.stats;
            std::cout << " algorithm=" << (args.hybrid_swap ? "hybrid" : "ratio")
                      << " target=" << args.swap_config.target_numerator << '/'
                      << args.swap_config.target_denominator
                      << " ratio=" << (before_sum ? double(before_peak) * args.num_devices / before_sum : 0.0)
                      << "->" << (after_sum ? double(after_peak) * args.num_devices / after_sum : 0.0)
                      << " target_met=" << (stats.target_met ? "true" : "false")
                      << " stop_reason=" << stats.stop_reason
                      << " screened_id_queries=" << stats.screened_id_queries
                      << " pair_evaluations=" << stats.pair_evaluations
                      << " cycle_evaluations=" << stats.cycle_evaluations
                      << " accepted_pairs=" << stats.accepted_pairs
                      << " accepted_cycles=" << stats.accepted_cycles
                      << " rejected_total=" << stats.rejected_total
                      << " rejected_peak=" << stats.rejected_peak
                      << " rejected_objective=" << stats.rejected_objective
                      << " epochs=" << stats.epochs;
        }
        std::cout << "\n";
    }

    // Verify the paper's per-global-batch conservation rule on every run,
    // including sequential/repair/no-swap schedules.
    std::vector<uint8_t> seen(num_samples, 0);
    for (const auto& bucket : buckets) {
        cps_swap::require(bucket.size() == size_t(args.train_batch_size), "Unequal rank capacity");
        for (int index : bucket) {
            cps_swap::require(index >= 0 && size_t(index) < num_samples && !seen[index],
                              "Sample lost or duplicated within global batch");
            seen[index] = 1;
        }
    }
    cps_swap::require(std::all_of(seen.begin(), seen.end(), [](uint8_t x) { return x == 1; }),
                      "Missing sample in global batch");
    auto t3 = std::chrono::high_resolution_clock::now();

    // 阶段 4：ID 转换 + 写磁盘
    std::atomic<long long> chunk_unmapped{0};
    transform_and_write(chunk_id, samples, buckets, args, mapping,
                        sparse_idx_to_tid, record_size, sparse_offset,
                        output_files, mutexes, chunk_unmapped);
    unmapped_total.fetch_add(chunk_unmapped.load());

    auto t4 = std::chrono::high_resolution_clock::now();

    counter.fetch_add((long long)num_samples);

    std::lock_guard<std::mutex> log_lock(log_mutex);
    std::cout << "  Chunk " << chunk_id << " (" << num_samples << " samples): "
              << "parse=" << std::chrono::duration<double>(t1 - t0).count() << "s "
              << "hits=" << std::chrono::duration<double>(t2 - t1).count() << "s "
              << "balance=" << std::chrono::duration<double>(t3 - t2).count() << "s "
              << "write=" << std::chrono::duration<double>(t4 - t3).count() << "s "
              << "total=" << std::chrono::duration<double>(t4 - t0).count() << "s"
              << " unmapped=" << chunk_unmapped.load() << "\n";
}

/* =========================
 * 主流程
 * ========================= */

int main(int argc, char* argv[]) try {
    auto program_start = std::chrono::high_resolution_clock::now();

    Args args;
    args.parse(argc, argv);
    cps_swap::require(!fs::exists(args.output_dir), "Output directory already exists; refusing to overwrite");

    std::cout << "[Init] OMP threads: " << omp_get_max_threads()
              << ", async threads: " << args.num_threads << "\n";

    auto mapping = load_and_build_mapping(args);
    auto mapping_end = std::chrono::high_resolution_clock::now();

    fs::create_directories(args.output_dir);

    auto sparse_idx_to_tid = get_sparse_index_mapping(args.table_lengths);
    size_t total_sparse = sparse_idx_to_tid.size();
    size_t sparse_offset = 4 + size_t(args.dense_count) * 4;
    size_t record_size = sparse_offset + total_sparse * 4;

    // 构建 checked_indices（vector<pair> 替代 map，cache 友好）
    // Dense Reduction contributes B*G vectors per rank regardless of sample
    // assignment. Optimize only the remaining variable Unique costs. This
    // changes the objective, never the record layout or the output ID mapping.
    std::unordered_set<int> excluded(args.reduction_tables.begin(), args.reduction_tables.end());
    if (!excluded.empty())
        std::cout << "[Objective] Fixed Reduction tables=" << excluded.size()
                  << ", constant vectors/rank=" << uint64_t(args.train_batch_size) * args.num_devices * excluded.size()
                  << "; swap statistics below cover variable Unique traffic only.\n";
    std::vector<std::pair<int, int>> checked_indices;
    for (size_t i = 0; i < sparse_idx_to_tid.size(); ++i) {
        if (!excluded.count(sparse_idx_to_tid[i])) {
            checked_indices.push_back({(int)i, sparse_idx_to_tid[i]});
        }
    }

    // 使用 FILE* 替代 ifstream，提升 I/O 性能
    FILE* fin = fopen(args.input_file.c_str(), "rb");
    if (!fin) {
        std::cerr << "Error: Cannot open input file " << args.input_file << "\n";
        return 1;
    }

    if (args.start_sample_idx > 0) {
        cps_swap::require(uint64_t(args.start_sample_idx) <=
                              uint64_t(std::numeric_limits<off_t>::max()) / record_size,
                          "Input sample offset exceeds off_t");
        const off_t offset = static_cast<off_t>(uint64_t(args.start_sample_idx) * record_size);
        std::cout << "[Init] Seeking to sample index " << args.start_sample_idx
                  << " (Offset: " << offset << " bytes)...\n";
        cps_swap::require(fseeko(fin, offset, SEEK_SET) == 0, "Cannot seek input dataset");
    }

    // 输出文件也使用 FILE* + setvbuf
    std::vector<FILE*> output_files(args.num_devices);
    std::vector<std::unique_ptr<std::mutex>> mutexes;

    // output_dir must be new. An input offset selects a source slice, never
    // enables append mode (which would override seek offsets on writes).
    const char* mode = "wb";
    for (int d = 0; d < args.num_devices; ++d) {
        fs::path out_path = fs::path(args.output_dir) / ("dataset_" + std::to_string(d) + ".bin");
        output_files[d] = fopen(out_path.c_str(), mode);
        if (!output_files[d]) {
            std::cerr << "Error: Cannot open output file " << out_path << "\n";
            return 1;
        }
        // 64MB 写缓冲
        setvbuf(output_files[d], nullptr, _IOFBF, 64 * 1024 * 1024);
        mutexes.push_back(std::make_unique<std::mutex>());
    }
    std::cout << "[Init] Output files opened.\n";

    std::atomic<long long> counter{args.start_sample_idx};
    std::atomic<long long> unmapped_total{0};
    std::vector<std::future<void>> futures;

    size_t chunk_samples = (size_t)args.train_batch_size * args.steps_per_chunk * args.num_devices;
    size_t chunk_bytes = chunk_samples * record_size;

    std::cout << "[Init] Starting processing loop. Chunk size: " << chunk_samples
              << " samples (" << chunk_bytes / (1024 * 1024) << " MB).\n";
    auto processing_start = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < args.num_chunks; ++c) {
        auto raw = std::make_shared<std::vector<char>>(chunk_bytes);
        size_t read_bytes = fread(raw->data(), 1, chunk_bytes, fin);
        cps_swap::require(ferror(fin) == 0, "Cannot read input dataset");

        if (read_bytes == 0) {
            std::lock_guard<std::mutex> log_lock(log_mutex);
            std::cout << "[Main] End of file reached.\n";
            break;
        }

        if (read_bytes < chunk_bytes) {
            std::lock_guard<std::mutex> log_lock(log_mutex);
            std::cout << "[Main] Last chunk: " << read_bytes << " bytes.\n";
            // 对齐到完整 record
            read_bytes = (read_bytes / record_size) * record_size;
            raw->resize(read_bytes);
        }

        // 限制并发线程数
        while (futures.size() >= (size_t)args.num_threads) {
            futures.erase(
                std::remove_if(futures.begin(), futures.end(),
                    [](auto& f) {
                        if (f.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) return false;
                        f.get();
                        return true;
                    }),
                futures.end());

            if (futures.size() >= (size_t)args.num_threads) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        {
            std::lock_guard<std::mutex> log_lock(log_mutex);
            std::cout << "[Main] Submitting Chunk " << c << "...\n";
        }
        futures.emplace_back(std::async(std::launch::async, process_chunk,
            c, raw, std::cref(args), std::cref(mapping),
            std::cref(checked_indices), std::cref(sparse_idx_to_tid),
            record_size, sparse_offset,
            std::ref(output_files), std::ref(mutexes), std::ref(counter),
            std::ref(unmapped_total)));
    }

    {
        std::lock_guard<std::mutex> log_lock(log_mutex);
        std::cout << "\n[Main] Waiting for remaining threads...\n";
    }
    for (auto& f : futures) f.get();

    bool close_failed = fclose(fin) != 0;
    for (int d = 0; d < args.num_devices; ++d) {
        if (fclose(output_files[d]) != 0) close_failed = true;
    }
    cps_swap::require(!close_failed, "Cannot flush or close dataset files");

    auto program_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(program_end - program_start).count();

    std::cout << "\n[Done] Total samples processed: " << counter
              << ", total time: " << total_sec << "s\n";
    std::cout << "[Timing] mapping_wall="
              << std::chrono::duration<double>(mapping_end - program_start).count()
              << "s processing_wall="
              << std::chrono::duration<double>(program_end - processing_start).count()
              << "s workers=" << args.num_threads << "\n";
    if (unmapped_total > 0) {
        std::cerr << "[Warning] " << unmapped_total
                  << " sparse entries were not found in the ID mapping and kept original values.\n";
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "generate_local_dataset: " << error.what() << "\n";
    return 1;
}
