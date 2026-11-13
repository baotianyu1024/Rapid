#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace HugeCTR {
namespace MultiHot {

// Experimental single-host plan applied to the pinned input batch before upload.
class RapidHostPlan {
 public:
  RapidHostPlan(const std::string& directory, size_t ranks, size_t global_batch,
                size_t record_bytes, size_t dense_count, const std::string& lengths)
      : ranks_(ranks), global_batch_(global_batch), local_batch_(global_batch / ranks),
        record_bytes_(record_bytes), dense_count_(dense_count) {
    profiling_ = std::getenv("RAPID_PROFILE_HOST") != nullptr;
    if (const char* value = std::getenv("RAPID_MAPPING_THREADS")) {
      mapping_threads_ = std::stoi(value);
      if (mapping_threads_ < 1 || mapping_threads_ > 128)
        throw std::runtime_error("RAPID_MAPPING_THREADS must be between 1 and 128");
    }
    std::istringstream in(lengths);
    std::string token;
    while (std::getline(in, token, ',')) {
      const auto n = std::stoul(token);
      if (!n) throw std::runtime_error("RAPID table hotness must be positive");
      for (size_t i = 0; i < n; ++i) columns_.push_back(lengths_.size());
      lengths_.push_back(n);
    }
    if ((1 + dense_count_ + columns_.size()) * sizeof(uint32_t) != record_bytes_)
      throw std::runtime_error("RAPID layout does not match input record size");
    std::ifstream map(directory + "/embedding_mapping.tsv");
    if (!map) throw std::runtime_error("Cannot open RAPID embedding mapping");
    uint32_t table, key, owner, physical;
    std::unordered_set<uint64_t> physical_ids;
    std::vector<uint32_t> max_key(lengths_.size(), 0);
    std::vector<bool> dense_safe(lengths_.size(), true);
    while (map >> table >> key >> owner >> physical) {
      if (table >= lengths_.size() || owner >= ranks_ || physical % ranks_ != owner ||
          !owners_.emplace((uint64_t(table) << 32) | key, physical).second ||
          !physical_ids.emplace((uint64_t(table) << 32) | physical).second)
        throw std::runtime_error("Invalid or duplicate RAPID embedding mapping");
      max_key[table] = std::max(max_key[table], key);
      if (physical == std::numeric_limits<uint32_t>::max()) dense_safe[table] = false;
    }
    if (!map.eof()) throw std::runtime_error("Malformed RAPID embedding mapping");
    // Bounded direct-index tables avoid pointer chasing in the per-sample hot path.
    dense_owners_.resize(lengths_.size());
    size_t dense_bytes = 0;
    constexpr size_t dense_limit = size_t{1} << 30;
    for (size_t t = 0; t < lengths_.size(); ++t) {
      const size_t bytes = (size_t(max_key[t]) + 1) * sizeof(uint32_t);
      if (dense_safe[t] && bytes <= dense_limit - dense_bytes) {
        dense_owners_[t].assign(size_t(max_key[t]) + 1, std::numeric_limits<uint32_t>::max());
        dense_bytes += bytes;
      }
    }
    for (const auto& item : owners_) {
      const size_t t = item.first >> 32;
      if (!dense_owners_[t].empty()) dense_owners_[t][uint32_t(item.first)] = item.second;
    }
    for (size_t r = 0; r < ranks_; ++r) {
      std::ifstream f(directory + "/sample_indices_" + std::to_string(r) + ".bin",
                      std::ios::binary | std::ios::ate);
      if (!f || f.tellg() < 0 || static_cast<size_t>(f.tellg()) % sizeof(uint64_t))
        throw std::runtime_error("Invalid RAPID sample index file");
      size_t n = static_cast<size_t>(f.tellg()) / sizeof(uint64_t);
      if (n % local_batch_ || (r && n != indices_[0].size()))
        throw std::runtime_error("RAPID ranks have inconsistent sample counts");
      indices_.emplace_back(n);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(indices_.back().data()), n * sizeof(uint64_t));
      if (!f) throw std::runtime_error("Cannot read RAPID sample indices");
    }
    if (const char* cache = std::getenv("RAPID_CACHE_MAPPED_BATCHES");
        cache && std::strcmp(cache, "1") == 0) {
      const size_t batches = batch_count();
      const size_t batch_bytes = global_batch_ * record_bytes_;
      if (batches > (size_t{8} << 30) / batch_bytes)
        throw std::runtime_error("RAPID mapped batch cache exceeds 8 GiB");
      for (size_t i = 0; i < batches; ++i)
        cached_batches_.emplace_back(std::make_unique<CachedBatch>());
    }
  }

  void apply(size_t batch_id, const std::vector<uint8_t*>& shards,
             const std::vector<size_t>& sizes) const {
    const auto start = std::chrono::steady_clock::now();
    if (shards.size() != ranks_ || sizes.size() != ranks_ ||
        (batch_id + 1) * local_batch_ > indices_[0].size())
      throw std::runtime_error("RAPID plan does not cover this batch");
    for (size_t size : sizes)
      if (size != local_batch_ * record_bytes_)
        throw std::runtime_error("RAPID requires complete global batches");
    std::unique_lock<std::mutex> cache_lock;
    CachedBatch* cached = nullptr;
    if (!cached_batches_.empty()) {
      cached = cached_batches_[batch_id].get();
      cache_lock = std::unique_lock<std::mutex>(cached->mutex);
      if (!cached->data.empty()) {
        #pragma omp parallel for num_threads(mapping_threads_)
        for (size_t r = 0; r < ranks_; ++r)
          std::memcpy(shards[r], cached->data.data() + r * local_batch_ * record_bytes_, sizes[r]);
        if (profiling_) {
          const auto done = std::chrono::steady_clock::now();
          std::cerr << "RAPID_HOST_CACHE_HIT batch=" << batch_id << " copy_us="
                    << std::chrono::duration_cast<std::chrono::microseconds>(done - start).count()
                    << '\n';
        }
        return;
      }
    }
    // Reuse the per-worker buffer: allocating and zeroing a full batch each time
    // otherwise dominates host preprocessing on cache-resident inputs.
    thread_local std::vector<uint8_t> output;
    output.resize(global_batch_ * record_bytes_);
    uint8_t* const output_data = output.data();
    std::vector<uint8_t> seen(global_batch_, 0);
    for (size_t rank = 0; rank < ranks_; ++rank) {
      for (size_t j = 0; j < local_batch_; ++j) {
        const uint64_t absolute = indices_[rank][batch_id * local_batch_ + j];
        const uint64_t first = batch_id * global_batch_;
        if (absolute < first || absolute >= first + global_batch_ || seen[absolute - first]++)
          throw std::runtime_error("RAPID sample indices are not a batch permutation");
      }
    }
    const auto validated = std::chrono::steady_clock::now();
    std::atomic<bool> missing{false};
    #pragma omp parallel for num_threads(mapping_threads_)
    for (size_t output_row = 0; output_row < global_batch_; ++output_row) {
        const size_t rank = output_row / local_batch_, j = output_row % local_batch_;
        const uint64_t absolute = indices_[rank][batch_id * local_batch_ + j];
        const uint64_t first = batch_id * global_batch_;
        const size_t source = absolute - first;
        uint8_t* dst = output_data + output_row * record_bytes_;
        std::memcpy(dst, shards[source / local_batch_] + (source % local_batch_) * record_bytes_,
                    record_bytes_);
        for (size_t c = 0; c < columns_.size(); ++c) {
          uint32_t value;
          uint8_t* field = dst + (1 + dense_count_ + c) * sizeof(uint32_t);
          std::memcpy(&value, field, sizeof(value));
          const size_t table = columns_[c];
          uint32_t physical = std::numeric_limits<uint32_t>::max();
          const auto& dense = dense_owners_[table];
          if (!dense.empty()) {
            if (value < dense.size()) physical = dense[value];
          } else {
            const auto it = owners_.find((uint64_t(table) << 32) | value);
            if (it != owners_.end()) physical = it->second;
          }
          if (physical == std::numeric_limits<uint32_t>::max()) {
            // A UINT32_MAX physical ID is valid only in the hash-backed path.
            if (dense.empty()) {
              const auto it = owners_.find((uint64_t(table) << 32) | value);
              if (it != owners_.end()) { std::memcpy(field, &it->second, sizeof(it->second)); continue; }
            }
            missing.store(true, std::memory_order_relaxed);
          } else {
            std::memcpy(field, &physical, sizeof(physical));
          }
        }
    }
    if (missing.load(std::memory_order_relaxed))
      throw std::runtime_error("RAPID mapping misses input ID");
    const auto mapped = std::chrono::steady_clock::now();
    if (cached) cached->data.assign(output_data, output_data + output.size());
    #pragma omp parallel for num_threads(mapping_threads_)
    for (size_t r = 0; r < ranks_; ++r)
      std::memcpy(shards[r], output_data + r * local_batch_ * record_bytes_, sizes[r]);
    if (profiling_) {
      const auto done = std::chrono::steady_clock::now();
      const auto us = [](auto d) { return std::chrono::duration_cast<std::chrono::microseconds>(d).count(); };
      std::ostringstream message;
      message << "RAPID_HOST_PROFILE batch=" << batch_id
              << " prepare_us=" << us(validated - start)
              << " map_us=" << us(mapped - validated)
              << " copyback_us=" << us(done - mapped) << '\n';
      std::cerr << message.str();
    }
  }

  size_t batch_count() const { return indices_[0].size() / local_batch_; }

  bool extends(const RapidHostPlan& old) const {
    if (ranks_ != old.ranks_ || global_batch_ != old.global_batch_ ||
        record_bytes_ != old.record_bytes_ || dense_count_ != old.dense_count_ ||
        lengths_ != old.lengths_ || batch_count() < old.batch_count()) return false;
    for (const auto& item : old.owners_) {
      auto it = owners_.find(item.first);
      if (it == owners_.end() || it->second != item.second) return false;
    }
    for (size_t r = 0; r < ranks_; ++r)
      if (!std::equal(old.indices_[r].begin(), old.indices_[r].end(), indices_[r].begin()))
        return false;
    return true;
  }

 private:
  struct CachedBatch {
    std::mutex mutex;
    std::vector<uint8_t> data;
  };
  size_t ranks_, global_batch_, local_batch_, record_bytes_, dense_count_;
  int mapping_threads_ = 8;
  bool profiling_ = false;
  std::vector<size_t> lengths_, columns_;
  std::unordered_map<uint64_t, uint32_t> owners_;
  std::vector<std::vector<uint32_t>> dense_owners_;
  std::vector<std::vector<uint64_t>> indices_;
  std::vector<std::unique_ptr<CachedBatch>> cached_batches_;
};
}  // namespace MultiHot
}  // namespace HugeCTR
