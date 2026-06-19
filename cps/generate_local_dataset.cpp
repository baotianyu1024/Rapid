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
#include <omp.h>

namespace fs = std::filesystem;

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
    int steps_per_chunk = 10;
    int num_chunks = 20;
    int num_threads = 5;
    long long start_sample_idx = 0;
    int dense_count = 13;
    float gamma = 1.0f;

    std::vector<int> excluded_tables;
    std::vector<int> table_lengths = {
        3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,
        100,27,10,3,1,1
    };

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
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

        if (input_file.empty() || table_files_dir.empty() || output_dir.empty()) {
            std::cerr << "Missing required arguments\n";
            std::exit(1);
        }
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

/* =========================
 * 贪心负载均衡分配（带 remote 惩罚，最小化 max(remote)）
 *
 * 原理：在选择最佳设备时，综合考虑本地命中数和该设备已累积的
 * unique remote ID 数量。通过 gamma 参数惩罚 remote 过高的设备，
 * 使 remote 访问在各设备间更均衡。
 *
 * 建议配合 steps_per_chunk=1 使用，使每个 batch 独立优化。
 * ========================= */

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

        // 写入磁盘（加锁保护同一文件的写入顺序）
        {
            std::lock_guard<std::mutex> lk(*mutexes[d]);
            fwrite(buf.data(), 1, buf.size(), output_files[d]);
        }
    }
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

    // 阶段 3：贪心分配
    int target = (int)((num_samples + args.num_devices - 1) / args.num_devices);
    auto buckets = balance_chunk(samples, args.num_devices, target, args.gamma);

    auto t3 = std::chrono::high_resolution_clock::now();

    // 阶段 4：ID 转换 + 写磁盘
    std::atomic<long long> chunk_unmapped{0};
    transform_and_write(samples, buckets, args, mapping,
                        sparse_idx_to_tid, record_size, sparse_offset,
                        output_files, mutexes, chunk_unmapped);
    unmapped_total.fetch_add(chunk_unmapped.load());

    auto t4 = std::chrono::high_resolution_clock::now();

    counter.fetch_add((long long)num_samples);

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

int main(int argc, char* argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    Args args;
    args.parse(argc, argv);

    std::cout << "[Init] OMP threads: " << omp_get_max_threads()
              << ", async threads: " << args.num_threads << "\n";

    auto mapping = load_and_build_mapping(args);

    fs::create_directories(args.output_dir);

    auto sparse_idx_to_tid = get_sparse_index_mapping(args.table_lengths);
    size_t total_sparse = sparse_idx_to_tid.size();
    size_t record_size = 4 + args.dense_count * 4 + total_sparse * 4;
    size_t sparse_offset = 4 + args.dense_count * 4;

    // 构建 checked_indices（vector<pair> 替代 map，cache 友好）
    std::unordered_set<int> excluded(args.excluded_tables.begin(), args.excluded_tables.end());
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
        long long offset = args.start_sample_idx * (long long)record_size;
        std::cout << "[Init] Seeking to sample index " << args.start_sample_idx
                  << " (Offset: " << offset << " bytes)...\n";
        fseeko(fin, offset, SEEK_SET);
    }

    // 输出文件也使用 FILE* + setvbuf
    std::vector<FILE*> output_files(args.num_devices);
    std::vector<std::unique_ptr<std::mutex>> mutexes;

    const char* mode = (args.start_sample_idx > 0) ? "ab" : "wb";
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

    for (int c = 0; c < args.num_chunks; ++c) {
        auto raw = std::make_shared<std::vector<char>>(chunk_bytes);
        size_t read_bytes = fread(raw->data(), 1, chunk_bytes, fin);

        if (read_bytes == 0) {
            std::cout << "[Main] End of file reached.\n";
            break;
        }

        if (read_bytes < chunk_bytes) {
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
                        return f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
                    }),
                futures.end());

            if (futures.size() >= (size_t)args.num_threads) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        std::cout << "[Main] Submitting Chunk " << c << "...\n";
        futures.emplace_back(std::async(std::launch::async, process_chunk,
            c, raw, std::cref(args), std::cref(mapping),
            std::cref(checked_indices), std::cref(sparse_idx_to_tid),
            record_size, sparse_offset,
            std::ref(output_files), std::ref(mutexes), std::ref(counter),
            std::ref(unmapped_total)));
    }

    std::cout << "\n[Main] Waiting for remaining threads...\n";
    for (auto& f : futures) f.wait();

    // 关闭所有文件
    fclose(fin);
    for (int d = 0; d < args.num_devices; ++d) {
        fclose(output_files[d]);
    }

    auto program_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(program_end - program_start).count();

    std::cout << "\n[Done] Total samples processed: " << counter
              << ", total time: " << total_sec << "s\n";
    if (unmapped_total > 0) {
        std::cerr << "[Warning] " << unmapped_total
                  << " sparse entries were not found in the ID mapping and kept original values.\n";
    }
    return 0;
}
