#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "flat_counts.h"

namespace cps_swap {

struct Id {
    uint64_t key;
    uint32_t owner;
};

using Sample = std::vector<Id>;
using Samples = std::vector<Sample>;
using Buckets = std::vector<std::vector<uint32_t>>;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

inline void canonicalize(Sample& sample) {
    std::sort(sample.begin(), sample.end(), [](const Id& a, const Id& b) {
        return a.key < b.key;
    });
    size_t out = 0;
    for (const Id id : sample) {
        if (out && sample[out - 1].key == id.key) {
            require(sample[out - 1].owner == id.owner, "Conflicting owners for one ID");
        } else {
            sample[out++] = id;
        }
    }
    sample.resize(out);
}

enum class Algorithm { Weighted, Ratio };

struct Config {
    uint32_t weight = 4;
    uint32_t rounds = 8;
    uint32_t source_samples = 256;
    uint32_t candidates = 16;
    uint32_t targets = 4;
    uint64_t evaluations_per_gpu = 12500;
    Algorithm algorithm = Algorithm::Weighted;
    uint32_t target_numerator = 59;
    uint32_t target_denominator = 50;
    bool cycles = true;
    bool stop_at_target = false;
    uint64_t screen_queries_per_gpu = 4000000;
};

struct RatioStats {
    uint64_t screened_id_queries = 0;
    uint64_t pair_evaluations = 0;
    uint64_t cycle_evaluations = 0;
    uint64_t accepted_pairs = 0;
    uint64_t accepted_cycles = 0;
    uint64_t rejected_total = 0;
    uint64_t rejected_peak = 0;
    uint64_t rejected_objective = 0;
    uint64_t epochs = 0;
    bool target_met = false;
    std::string stop_reason;
};

struct Result {
    Buckets buckets;
    std::vector<uint64_t> before;
    std::vector<uint64_t> after;
    uint64_t evaluations = 0;
    uint64_t accepted = 0;
    RatioStats stats;
};

inline void validate_buckets(const Samples& samples, const Buckets& buckets) {
    require(!buckets.empty(), "At least one GPU is required");
    require(samples.size() <= std::numeric_limits<uint32_t>::max(), "Too many samples");
    const size_t capacity = buckets.front().size();
    require(capacity * buckets.size() == samples.size(), "Batch dimensions disagree");
    std::vector<uint8_t> seen(samples.size(), 0);
    for (const auto& bucket : buckets) {
        require(bucket.size() == capacity, "GPU sample capacities differ");
        for (uint32_t sample : bucket) {
            require(sample < samples.size() && !seen[sample], "Invalid sample permutation");
            seen[sample] = 1;
        }
    }
}

inline std::vector<uint64_t> count_remote(const Samples& samples, const Buckets& buckets) {
    std::vector<uint64_t> volumes(buckets.size(), 0);
    for (size_t d = 0; d < buckets.size(); ++d) {
        std::unordered_set<uint64_t> ids;
        for (uint32_t i : buckets[d]) {
            for (const Id id : samples[i]) {
                if (id.owner != d) ids.insert(id.key);
            }
        }
        volumes[d] = ids.size();
    }
    return volumes;
}

class State {
public:
    State(const Samples& samples, Buckets buckets)
        : samples_(samples), buckets_(std::move(buckets)), counts_(buckets_.size()),
          volumes_(buckets_.size(), 0) {
        validate_buckets(samples_, buckets_);
        for (const Sample& sample : samples_) {
            for (size_t i = 0; i < sample.size(); ++i) {
                require(sample[i].owner < buckets_.size(), "ID owner is out of range");
                require(!i || sample[i - 1].key < sample[i].key,
                        "Samples must have sorted, unique ID keys");
            }
        }
        for (uint32_t d = 0; d < buckets_.size(); ++d) {
            for (uint32_t sample : buckets_[d]) add(d, samples_[sample]);
            volumes_[d] = counts_[d].size();
        }
    }

    const Buckets& buckets() const { return buckets_; }
    const std::vector<uint64_t>& volumes() const { return volumes_; }

    uint32_t removable(uint32_t rank, uint32_t slot, uint64_t* queries = nullptr) const {
        uint32_t total = 0;
        for (const Id id : samples_[buckets_[rank][slot]]) {
            if (id.owner != rank && count(rank, id.key, queries) == 1) ++total;
        }
        return total;
    }

    uint32_t insertion_cost(uint32_t rank, const Sample& sample,
                            uint64_t* queries = nullptr) const {
        require(rank < buckets_.size(), "Insertion rank out of range");
        require(sample.size() <= std::numeric_limits<uint32_t>::max(), "Sample too large");
        uint32_t total = 0;
        for (const Id id : sample) {
            if (id.owner != rank && count(rank, id.key, queries) == 0) ++total;
        }
        return total;
    }

    std::pair<int64_t, int64_t> swap_delta(uint32_t a, uint32_t x,
                                         uint32_t b, uint32_t y) const {
        require(a < buckets_.size() && b < buckets_.size() &&
                x < buckets_[a].size() && y < buckets_[b].size(), "Swap slot out of range");
        if (a == b) return {0, 0};
        const Sample& left = samples_[buckets_[a][x]];
        const Sample& right = samples_[buckets_[b][y]];
        return {replacement_delta(a, left, right), replacement_delta(b, right, left)};
    }

    void apply_swap(uint32_t a, uint32_t x, uint32_t b, uint32_t y) {
        const auto delta = swap_delta(a, x, b, y);
        if (a == b) {
            std::swap(buckets_[a][x], buckets_[b][y]);
            return;
        }
        const Sample& left = samples_[buckets_[a][x]];
        const Sample& right = samples_[buckets_[b][y]];
        remove(a, left);
        remove(b, right);
        add(a, right);
        add(b, left);
        volumes_[a] = uint64_t(int64_t(volumes_[a]) + delta.first);
        volumes_[b] = uint64_t(int64_t(volumes_[b]) + delta.second);
        require(volumes_[a] == counts_[a].size() && volumes_[b] == counts_[b].size(),
                "Swap reference counts disagree with volume");
        std::swap(buckets_[a][x], buckets_[b][y]);
    }

    std::array<int64_t, 3> cycle_delta(uint32_t a, uint32_t x, uint32_t b, uint32_t y,
                                        uint32_t c, uint32_t z, bool reverse = false) const {
        require(a < buckets_.size() && b < buckets_.size() && c < buckets_.size() &&
                a != b && a != c && b != c, "Cycle needs three distinct valid ranks");
        require(x < buckets_[a].size() && y < buckets_[b].size() && z < buckets_[c].size(),
                "Cycle slot out of range");
        const Sample& left = samples_[buckets_[a][x]];
        const Sample& middle = samples_[buckets_[b][y]];
        const Sample& right = samples_[buckets_[c][z]];
        return reverse ? std::array<int64_t, 3>{{replacement_delta(a, left, middle),
                                                replacement_delta(b, middle, right),
                                                replacement_delta(c, right, left)}}
                       : std::array<int64_t, 3>{{replacement_delta(a, left, right),
                                                replacement_delta(b, middle, left),
                                                replacement_delta(c, right, middle)}};
    }

    void apply_cycle(uint32_t a, uint32_t x, uint32_t b, uint32_t y,
                     uint32_t c, uint32_t z, bool reverse = false) {
        const auto delta = cycle_delta(a, x, b, y, c, z, reverse);
        const std::array<uint32_t, 3> ranks{{a, b, c}}, slots{{x, y, z}};
        const std::array<uint32_t, 3> outgoing{{buckets_[a][x], buckets_[b][y], buckets_[c][z]}};
        const std::array<uint32_t, 3> incoming = reverse
            ? std::array<uint32_t, 3>{{outgoing[1], outgoing[2], outgoing[0]}}
            : std::array<uint32_t, 3>{{outgoing[2], outgoing[0], outgoing[1]}};
        // All deltas refer to the same old state; never realize a cycle as two swaps.
        for (size_t i = 0; i < 3; ++i) remove(ranks[i], samples_[outgoing[i]]);
        for (size_t i = 0; i < 3; ++i) add(ranks[i], samples_[incoming[i]]);
        for (size_t i = 0; i < 3; ++i) {
            volumes_[ranks[i]] = uint64_t(int64_t(volumes_[ranks[i]]) + delta[i]);
            require(volumes_[ranks[i]] == counts_[ranks[i]].size(),
                    "Cycle reference counts disagree with volume");
            buckets_[ranks[i]][slots[i]] = incoming[i];
        }
    }

public:
    uint32_t count(uint32_t rank, uint64_t key, uint64_t* queries = nullptr) const {
        if (queries) {
            require(*queries < std::numeric_limits<uint64_t>::max(), "Query counter overflow");
            ++*queries;
        }
        const auto it = counts_[rank].find(key);
        return it == counts_[rank].end() ? 0 : it->second;
    }

    int64_t replacement_delta(uint32_t rank, const Sample& outgoing,
                              const Sample& incoming) const {
        size_t i = 0, j = 0;
        int64_t delta = 0;
        while (i < outgoing.size() || j < incoming.size()) {
            if (j == incoming.size() || (i < outgoing.size() && outgoing[i].key < incoming[j].key)) {
                const Id id = outgoing[i++];
                if (id.owner != rank && count(rank, id.key) == 1) --delta;
            } else if (i == outgoing.size() || incoming[j].key < outgoing[i].key) {
                const Id id = incoming[j++];
                if (id.owner != rank && count(rank, id.key) == 0) ++delta;
            } else {
                ++i;
                ++j;
            }
        }
        return delta;
    }

    void remove(uint32_t rank, const Sample& sample) {
        for (const Id id : sample) {
            if (id.owner == rank) continue;
            const auto it = counts_[rank].find(id.key);
            require(it != counts_[rank].end() && it->second, "Missing outgoing ID reference");
            if (--it->second == 0) counts_[rank].erase(it);
        }
    }

    void add(uint32_t rank, const Sample& sample) {
        for (const Id id : sample) {
            if (id.owner != rank) ++counts_[rank][id.key];
        }
    }

    const Samples& samples_;
    Buckets buckets_;
    std::vector<FlatCounts> counts_;
    std::vector<uint64_t> volumes_;
};

inline long double squared(uint64_t value) {
    return static_cast<long double>(value) * value;
}

inline uint64_t mixed(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

namespace ratio_detail {

using Wide = unsigned __int128;

inline Wide checked_add(Wide a, Wide b) {
    require(a <= ~Wide(0) - b, "Ratio wide addition overflow");
    return a + b;
}

inline Wide checked_mul(Wide a, Wide b) {
    require(!b || a <= ~Wide(0) / b, "Ratio wide multiplication overflow");
    return a * b;
}

inline uint64_t budget_product(uint64_t per_gpu, uint32_t gpus) {
    const Wide product = checked_mul(per_gpu, gpus);
    require(product <= std::numeric_limits<uint64_t>::max(), "Ratio budget overflow");
    return uint64_t(product);
}

struct Metrics {
    Wide excess = 0;
    Wide squares = 0;
    uint64_t total = 0;
    uint64_t peak = 0;
};

inline Metrics measure(const std::vector<uint64_t>& volumes, uint32_t p, uint32_t q) {
    Metrics value;
    Wide total = 0;
    for (uint64_t v : volumes) {
        total = checked_add(total, v);
        value.peak = std::max(value.peak, v);
        value.squares = checked_add(value.squares, checked_mul(v, v));
    }
    require(total <= uint64_t(std::numeric_limits<int64_t>::max()),
            "Ratio communication exceeds signed delta capacity");
    value.total = uint64_t(total);
    const Wide threshold = checked_mul(p, total);
    const Wide qg = checked_mul(q, volumes.size());
    // A changed S changes the threshold for EVERY rank, including untouched ranks.
    for (uint64_t v : volumes) {
        const Wide load = checked_mul(qg, v);
        if (load > threshold) {
            const Wide e = load - threshold;
            value.excess = checked_add(value.excess, checked_mul(e, e));
        }
    }
    return value;
}

inline bool same(const Metrics& a, const Metrics& b) {
    return std::tie(a.excess, a.peak, a.total, a.squares) ==
           std::tie(b.excess, b.peak, b.total, b.squares);
}

inline bool improves(const Metrics& candidate, const Metrics& current) {
    if (current.excess)
        return std::tie(candidate.excess, candidate.peak, candidate.total, candidate.squares) <
               std::tie(current.excess, current.peak, current.total, current.squares);
    return !candidate.excess &&
           std::tie(candidate.total, candidate.peak, candidate.squares) <
           std::tie(current.total, current.peak, current.squares);
}

inline bool better_return(const Metrics& candidate, const Metrics& best) {
    if ((candidate.excess == 0) != (best.excess == 0)) return candidate.excess == 0;
    if (candidate.excess) {
        // G cancels. Zero communication is feasible and never reaches this branch.
        const Wide left = checked_mul(candidate.peak, best.total);
        const Wide right = checked_mul(best.peak, candidate.total);
        if (left != right) return left < right;
    }
    return std::tie(candidate.total, candidate.peak) < std::tie(best.total, best.peak);
}

inline void validate(const Samples& samples, const Buckets& initial, const Config& config) {
    require(config.source_samples && config.candidates && config.targets,
            "Candidate search sizes must be positive");
    require(config.target_denominator >= 1 && config.target_denominator <= 1000 &&
            config.target_numerator >= config.target_denominator &&
            config.target_numerator <= 2 * config.target_denominator,
            "Ratio target must satisfy 1 <= denominator <= 1000 and denominator <= numerator <= 2*denominator");
    require(!initial.empty() && initial.size() <= std::numeric_limits<uint32_t>::max(),
            "Ratio rank count is out of range");
    require(samples.size() <= std::numeric_limits<uint32_t>::max() &&
            initial.front().size() <= std::numeric_limits<uint32_t>::max(),
            "Ratio sample count is out of range");
    const uint32_t gpus = uint32_t(initial.size());
    (void)budget_product(config.evaluations_per_gpu, gpus);
    (void)budget_product(config.screen_queries_per_gpu, gpus);
    Wide references = 0;
    for (const Sample& sample : samples) {
        require(sample.size() <= std::numeric_limits<uint32_t>::max(), "Sample too large");
        references = checked_add(references, sample.size());
    }
    require(references <= uint64_t(std::numeric_limits<int64_t>::max()),
            "Ratio references exceed signed delta capacity");
    // Conservative bounds cover every permutation, not just the initial objective.
    const Wide maximum_excess = checked_mul(checked_mul(config.target_denominator, gpus), references);
    (void)checked_mul(gpus, checked_mul(maximum_excess, maximum_excess));
    (void)checked_mul(gpus, checked_mul(references, references));
    (void)checked_mul(config.target_numerator, references);
    (void)checked_mul(references, references); // exact R cross products
}

struct NoopObserver {
    void operator()(const State&) const noexcept {}
};

template<class Observer = NoopObserver>
class Search : private Observer {
public:
    Search(const Samples& samples, const Buckets& initial, const Config& config,
           Observer observer = Observer{})
        : Observer(std::move(observer)), samples_(samples), config_(config), state_(samples, initial),
          gpus_(uint32_t(initial.size())), capacity_(uint32_t(initial.front().size())),
          budget_(budget_product(config.evaluations_per_gpu, gpus_)),
          screen_budget_(budget_product(config.screen_queries_per_gpu, gpus_)),
          cycle_budget_(budget_ / 5), source_cursor_(gpus_, 0),
          return_cursor_(gpus_, 0), destination_cursor_(gpus_, 0),
          revisions_(gpus_, 0), return_cache_(gpus_) {
        result_.before = state_.volumes();
        current_ = measure(state_.volumes(), config_.target_numerator, config_.target_denominator);
        best_ = current_;
        best_buckets_ = state_.buckets();
        best_volumes_ = state_.volumes();
        peak0_ = current_.peak;
    }

    Result run() {
        const uint32_t scan = uint32_t(std::min<uint64_t>(capacity_, uint64_t(config_.source_samples) * 8));
        const uint64_t blocks = scan ? (uint64_t(capacity_) + scan - 1) / scan : 0;
        // A round covers every rank and source slot via explicit coverage scans.
        // Epochs only refresh priorities; they do NOT consume Config::rounds.
        for (uint32_t round = 0; round < config_.rounds && !halted(); ++round) {
            for (uint64_t block = 0; block < blocks && !halted(); ++block) {
                for (uint32_t covered_rank = 0; covered_rank < gpus_ && !halted(); ++covered_rank) {
                    for (uint32_t share = 0; share < 5 && !halted(); ++share) {
                        if (!epoch_open_) begin_epoch();
                        const bool coverage = share == 4;
                        const uint32_t source = coverage ? covered_rank : priority_[priority_cursor_];
                        if (!coverage) priority_cursor_ = (priority_cursor_ + 1) % priority_.size();
                        const uint32_t start = coverage ? uint32_t(block * scan) : source_cursor_[source];
                        const uint32_t count = coverage ? std::min(scan, capacity_ - start) : scan;
                        source_cursor_[source] = uint32_t((uint64_t(start) + count) % capacity_);
                        bool scanned;
                        if (!coverage && config_.cycles && current_.excess && gpus_ >= 3 &&
                            stalled_ >= 2 && !epoch_cycles_ && room(2, true)) {
                            epoch_cycles_ = true;
                            scanned = cycle_visit(source, start, count);
                        } else {
                            scanned = pair_visit(source, start, count);
                        }
                        epoch_coverage_ = epoch_coverage_ || (coverage && scanned);
                        if (result_.accepted - epoch_accepted_ >= 64 ||
                            result_.evaluations - epoch_evaluations_ >= 4095) finish_epoch();
                    }
                }
            }
            finish_epoch(); // A small complete coverage round is also a covered epoch.
        }
        finish_epoch();
        if (!capacity_) result_.stats.stop_reason = "empty_batch";
        else if (gpus_ < 2) result_.stats.stop_reason = "single_rank";
        else if (!current_.total) result_.stats.stop_reason = "zero_communication";
        else if (config_.stop_at_target && !best_.excess) result_.stats.stop_reason = "target_reached";
        else if (screen_blocked_) result_.stats.stop_reason = "screen_budget";
        else if (budget_ - result_.evaluations < 2) result_.stats.stop_reason = "evaluation_budget";
        else result_.stats.stop_reason = "coverage_rounds";

        result_.buckets = std::move(best_buckets_);
        validate_buckets(samples_, result_.buckets);
        result_.after = count_remote(samples_, result_.buckets);
        require(result_.after == best_volumes_, "Returned best state failed independent recount");
        const Metrics verified = measure(result_.after, config_.target_numerator, config_.target_denominator);
        require(same(verified, best_), "Returned best objective failed verification");
        require(verified.total <= measure(result_.before, config_.target_numerator,
                                         config_.target_denominator).total && verified.peak <= peak0_,
                "Ratio optimizer exceeded baseline bounds");
        require(result_.evaluations <= budget_ && result_.stats.screened_id_queries <= screen_budget_ &&
                result_.stats.cycle_evaluations <= cycle_budget_ / 2 &&
                result_.evaluations == result_.stats.pair_evaluations + 2 * result_.stats.cycle_evaluations &&
                result_.accepted == result_.stats.accepted_pairs + result_.stats.accepted_cycles,
                "Ratio accounting invariant failed");
        result_.stats.target_met = !verified.excess;
        return std::move(result_);
    }

private:
    struct Source { uint32_t slot, removed; };
    struct Destination { uint32_t rank, missing; Wide excess; };
    struct Return { uint32_t slot, inserted, removed; };
    struct ReturnCache {
        bool valid = false, coverage = false;
        uint32_t receiver = 0, keep = 0;
        uint64_t rank_revision = 0, receiver_revision = 0, epoch = 0;
        std::vector<Return> values;
    };
    struct Choice {
        bool found = false;
        bool reverse = false;
        uint32_t size = 2;
        std::array<uint32_t, 3> ranks{{0, 0, 0}}, slots{{0, 0, 0}};
        Metrics value;
    };

    bool halted() const {
        return gpus_ < 2 || !capacity_ || !current_.total || screen_blocked_ ||
               (config_.stop_at_target && !best_.excess) ||
               budget_ - result_.evaluations < 2;
    }

    bool room(uint64_t cost, bool cycle = false) const {
        // Reserve the exact re-evaluation inside apply_* before even proposing.
        if (result_.accepted - epoch_accepted_ >= 64 ||
            budget_ - result_.evaluations < 2 * cost ||
            4096 - (result_.evaluations - epoch_evaluations_) < 2 * cost) return false;
        return !cycle || cycle_budget_ - 2 * result_.stats.cycle_evaluations >= 2 * cost;
    }

    bool screen(const Sample& sample) {
        if (sample.size() > screen_budget_ - result_.stats.screened_id_queries) {
            screen_blocked_ = true;
            return false;
        }
        return true;
    }

    void begin_epoch() {
        epoch_open_ = true;
        epoch_coverage_ = false;
        epoch_cycles_ = false;
        epoch_evaluations_ = result_.evaluations;
        epoch_accepted_ = result_.accepted;
        priority_.clear();
        const Wide qg = checked_mul(config_.target_denominator, gpus_);
        const Wide threshold = checked_mul(config_.target_numerator, current_.total);
        for (uint32_t d = 0; d < gpus_; ++d)
            if (!current_.excess || checked_mul(qg, state_.volumes()[d]) > threshold)
                priority_.push_back(d);
        require(!priority_.empty(), "Ratio priority list is empty");
        std::sort(priority_.begin(), priority_.end(), [&](uint32_t a, uint32_t b) {
            return state_.volumes()[a] != state_.volumes()[b]
                ? state_.volumes()[a] > state_.volumes()[b] : a < b;
        });
        priority_cursor_ %= priority_.size();
    }

    void finish_epoch() {
        if (!epoch_open_) return;
        ++result_.stats.epochs;
        if (result_.accepted != epoch_accepted_) stalled_ = 0;
        else if (epoch_coverage_) stalled_ = std::min<uint32_t>(2, stalled_ + 1);
        epoch_open_ = false;
    }

    std::vector<Source> sources(uint32_t rank, uint32_t start, uint32_t count) {
        std::vector<Source> selected;
        selected.reserve(count);
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t slot = uint32_t((uint64_t(start) + k) % capacity_);
            if (!screen(samples_[state_.buckets()[rank][slot]])) break;
            selected.push_back({slot, state_.removable(rank, slot, &result_.stats.screened_id_queries)});
        }
        std::sort(selected.begin(), selected.end(), [](const Source& a, const Source& b) {
            return a.removed != b.removed ? a.removed > b.removed : a.slot < b.slot;
        });
        if (selected.size() > config_.source_samples) selected.resize(config_.source_samples);
        return selected;
    }

    std::vector<Destination> destinations(uint32_t source, const Sample& outgoing) {
        std::vector<uint32_t> hits(gpus_, 0), low(gpus_), local(gpus_);
        for (const Id id : outgoing) ++hits[id.owner];
        std::iota(low.begin(), low.end(), 0);
        std::iota(local.begin(), local.end(), 0);
        std::sort(low.begin(), low.end(), [&](uint32_t a, uint32_t b) {
            return std::tie(state_.volumes()[a], a) < std::tie(state_.volumes()[b], b);
        });
        std::sort(local.begin(), local.end(), [&](uint32_t a, uint32_t b) {
            return hits[a] != hits[b] ? hits[a] > hits[b] :
                   std::tie(state_.volumes()[a], a) < std::tie(state_.volumes()[b], b);
        });
        const uint32_t keep = std::min(config_.targets, gpus_ - 1);
        const uint32_t pool_size = uint32_t(std::min<uint64_t>(gpus_ - 1, uint64_t(keep) * 2));
        std::vector<uint32_t> pool;
        const auto append = [&](uint32_t rank) {
            if (rank != source && pool.size() < pool_size &&
                std::find(pool.begin(), pool.end(), rank) == pool.end()) pool.push_back(rank);
        };
        const uint32_t rotation = destination_cursor_[source];
        destination_cursor_[source] = uint32_t((uint64_t(rotation) + 1) % gpus_);
        for (uint32_t k = 0; k < gpus_ && pool.size() < pool_size; ++k) {
            append(low[k]);
            append(local[k]);
            append(uint32_t((uint64_t(source) + 1 + rotation + k) % gpus_));
        }
        std::vector<Destination> selected;
        const Wide threshold = checked_mul(config_.target_numerator, current_.total);
        const Wide qg = checked_mul(config_.target_denominator, gpus_);
        for (uint32_t rank : pool) {
            if (!screen(outgoing)) break;
            const uint32_t missing = state_.insertion_cost(rank, outgoing, &result_.stats.screened_id_queries);
            const Wide projected = checked_mul(qg, checked_add(state_.volumes()[rank], missing));
            selected.push_back({rank, missing, projected > threshold ? projected - threshold : 0});
        }
        std::sort(selected.begin(), selected.end(), [&](const Destination& a, const Destination& b) {
            return std::tie(a.excess, a.missing, state_.volumes()[a.rank], a.rank) <
                   std::tie(b.excess, b.missing, state_.volumes()[b.rank], b.rank);
        });
        if (selected.size() > keep) selected.resize(keep);
        return selected;
    }

    const std::vector<Return>& returns(uint32_t rank, uint32_t receiver, uint32_t keep, bool coverage) {
        auto& cached = return_cache_[rank];
        if (!cached.valid || cached.receiver != receiver || cached.keep != keep ||
            cached.coverage != coverage || cached.rank_revision != revisions_[rank] ||
            cached.receiver_revision != revisions_[receiver] || cached.epoch != result_.stats.epochs) {
            cached.values = screen_returns(rank, receiver, keep, coverage);
            cached.valid = !screen_blocked_;
            cached.receiver = receiver;
            cached.keep = keep;
            cached.coverage = coverage;
            cached.rank_revision = revisions_[rank];
            cached.receiver_revision = revisions_[receiver];
            cached.epoch = result_.stats.epochs;
        }
        return cached.values;
    }

    std::vector<Return> screen_returns(uint32_t rank, uint32_t receiver, uint32_t keep, bool coverage) {
        const uint32_t scan = uint32_t(std::min<uint64_t>(capacity_, uint64_t(config_.candidates) * 4));
        const uint32_t offset = return_cursor_[rank];
        return_cursor_[rank] = uint32_t((uint64_t(offset) + 1) % capacity_);
        std::vector<Return> sampled;
        sampled.reserve(scan);
        for (uint32_t k = 0; k < scan; ++k) {
            const uint32_t slot = uint32_t((uint64_t(k) * capacity_ / scan + offset) % capacity_);
            const Sample& sample = samples_[state_.buckets()[rank][slot]];
            if (!screen(sample)) break;
            const uint32_t inserted = state_.insertion_cost(receiver, sample, &result_.stats.screened_id_queries);
            if (!screen(sample)) break;
            const uint32_t removed = state_.removable(rank, slot, &result_.stats.screened_id_queries);
            sampled.push_back({slot, inserted, removed});
        }
        std::vector<Return> sorted = sampled;
        // For a fixed outgoing sample, its removal and destination insertion are
        // constants. This is the two-sided marginal total, then source insertion.
        std::sort(sorted.begin(), sorted.end(), [](const Return& a, const Return& b) {
            const int64_t left = int64_t(a.inserted) - a.removed;
            const int64_t right = int64_t(b.inserted) - b.removed;
            return std::tie(left, a.inserted, a.slot) < std::tie(right, b.inserted, b.slot);
        });
        keep = uint32_t(std::min<size_t>(keep, sorted.size()));
        const uint32_t covered = coverage ? uint32_t((uint64_t(keep) + 3) / 4) : 0;
        sorted.resize(keep - covered);
        for (uint32_t k = 0; k < covered; ++k) {
            size_t index = uint64_t(k) * sampled.size() / covered;
            while (std::any_of(sorted.begin(), sorted.end(), [&](const Return& r) {
                return r.slot == sampled[index].slot;
            })) index = (index + 1) % sampled.size();
            sorted.push_back(sampled[index]);
        }
        return sorted;
    }

    Metrics proposed(const std::array<uint32_t, 3>& ranks,
                     const std::array<int64_t, 3>& deltas, uint32_t count) {
        scratch_ = state_.volumes();
        for (uint32_t i = 0; i < count; ++i) {
            const uint64_t old = scratch_[ranks[i]];
            if (deltas[i] < 0) {
                const uint64_t amount = uint64_t(-(deltas[i] + 1)) + 1;
                require(old >= amount, "Negative ratio rank volume");
                scratch_[ranks[i]] = old - amount;
            } else {
                const Wide volume = checked_add(old, uint64_t(deltas[i]));
                require(volume <= uint64_t(std::numeric_limits<int64_t>::max()),
                        "Ratio rank volume overflow");
                scratch_[ranks[i]] = uint64_t(volume);
            }
        }
        return measure(scratch_, config_.target_numerator, config_.target_denominator);
    }

    void consider(Choice candidate, Choice& choice) {
        if (candidate.value.total > current_.total) { ++result_.stats.rejected_total; return; }
        if (candidate.value.peak > peak0_) { ++result_.stats.rejected_peak; return; }
        if (!improves(candidate.value, current_)) { ++result_.stats.rejected_objective; return; }
        // Rejections classify proposals against the current state, not apply calls
        // or improving proposals superseded by another proposal in the shortlist.
        if (!choice.found || improves(candidate.value, choice.value) ||
            (same(candidate.value, choice.value) &&
             std::tie(candidate.ranks, candidate.slots, candidate.reverse) <
             std::tie(choice.ranks, choice.slots, choice.reverse))) {
            candidate.found = true;
            choice = candidate;
        }
    }

    void commit(const Choice& choice) {
        if (!choice.found) return;
        const bool cycle = choice.size == 3;
        const uint64_t cost = cycle ? 2 : 1;
        require(budget_ - result_.evaluations >= cost &&
                4096 - (result_.evaluations - epoch_evaluations_) >= cost &&
                (!cycle || cycle_budget_ - 2 * result_.stats.cycle_evaluations >= cost),
                "Missing reserved ratio apply budget");
        const auto& r = choice.ranks;
        const auto& s = choice.slots;
        if (cycle) {
            state_.apply_cycle(r[0], s[0], r[1], s[1], r[2], s[2], choice.reverse);
            ++result_.stats.cycle_evaluations; // apply_cycle calls cycle_delta once
            ++result_.stats.accepted_cycles;
        } else {
            state_.apply_swap(r[0], s[0], r[1], s[1]);
            ++result_.stats.pair_evaluations; // apply_swap calls swap_delta once
            ++result_.stats.accepted_pairs;
        }
        result_.evaluations += cost;
        const Metrics verified = measure(state_.volumes(), config_.target_numerator, config_.target_denominator);
        require(same(verified, choice.value) && verified.total <= current_.total &&
                verified.peak <= peak0_ && improves(verified, current_),
                "Committed ratio move disagrees with proposal");
        current_ = verified;
        ++result_.accepted;
        for (uint32_t i = 0; i < choice.size; ++i) ++revisions_[r[i]];
        stalled_ = 0;
        if (better_return(current_, best_)) {
            best_ = current_;
            best_buckets_ = state_.buckets();
            best_volumes_ = state_.volumes();
        }
        static_cast<Observer&>(*this)(static_cast<const State&>(state_));
    }

    bool pair_visit(uint32_t source, uint32_t start, uint32_t count) {
        if (!room(1)) return false;
        const auto outgoing = sources(source, start, count);
        const bool scanned = !screen_blocked_;
        // Retain only the slot order after commits; no old deletion score is reused.
        for (const Source& x : outgoing) {
            if (halted() || !room(1)) break;
            Choice choice;
            const auto targets = destinations(source, samples_[state_.buckets()[source][x.slot]]);
            for (const Destination& d : targets) {
                if (screen_blocked_ || !room(1)) break;
                const auto& incoming = returns(d.rank, source, config_.candidates, true);
                for (const Return& y : incoming) {
                    if (screen_blocked_ || !room(1)) break;
                    const auto delta = state_.swap_delta(source, x.slot, d.rank, y.slot);
                    ++result_.stats.pair_evaluations;
                    ++result_.evaluations;
                    Choice candidate;
                    candidate.ranks = {{source, d.rank, 0}};
                    candidate.slots = {{x.slot, y.slot, 0}};
                    candidate.value = proposed(candidate.ranks, {{delta.first, delta.second, 0}}, 2);
                    consider(candidate, choice);
                }
            }
            commit(choice);
        }
        return scanned;
    }

    std::vector<Return> source_edge(const std::vector<Source>& outgoing,
                                    uint32_t source, uint32_t receiver) {
        std::vector<Return> edge;
        for (const Source& x : outgoing) {
            const Sample& sample = samples_[state_.buckets()[source][x.slot]];
            if (!screen(sample)) break;
            const uint32_t inserted = state_.insertion_cost(receiver, sample, &result_.stats.screened_id_queries);
            edge.push_back({x.slot, inserted, x.removed});
        }
        std::sort(edge.begin(), edge.end(), [](const Return& a, const Return& b) {
            const int64_t left = int64_t(a.inserted) - a.removed;
            const int64_t right = int64_t(b.inserted) - b.removed;
            return std::tie(left, a.inserted, a.slot) < std::tie(right, b.inserted, b.slot);
        });
        if (edge.size() > 8) edge.resize(8);
        return edge;
    }

    bool cycle_visit(uint32_t source, uint32_t start, uint32_t count) {
        const auto outgoing = sources(source, start, count);
        const bool scanned = !screen_blocked_;
        if (outgoing.empty() || screen_blocked_) return scanned;
        // Bound rank combinations too: use the strongest removable source sample's
        // insertion-aware destination shortlist, then top eight per directed edge.
        const auto targets = destinations(source, samples_[state_.buckets()[source][outgoing.front().slot]]);
        Choice choice;
        for (size_t i = 0; i < targets.size() && !screen_blocked_ && room(2, true); ++i) {
            for (size_t j = i + 1; j < targets.size() && !screen_blocked_ && room(2, true); ++j) {
                const uint32_t b = std::min(targets[i].rank, targets[j].rank);
                const uint32_t c = std::max(targets[i].rank, targets[j].rank);
                // Interleave orientations per slot combination so both directions
                // are tested even when an epoch ends in the first rank triple.
                std::array<std::vector<Return>, 2> xs, ys, zs;
                xs[0] = source_edge(outgoing, source, b);
                if (!screen_blocked_) ys[0] = returns(b, c, 8, false);
                if (!screen_blocked_) zs[0] = returns(c, source, 8, false);
                if (!screen_blocked_) xs[1] = source_edge(outgoing, source, c);
                if (!screen_blocked_) ys[1] = returns(b, source, 8, false);
                if (!screen_blocked_) zs[1] = returns(c, b, 8, false);
                for (size_t x = 0; x < 8 && !screen_blocked_ && room(2, true); ++x) {
                    for (size_t y = 0; y < 8 && !screen_blocked_ && room(2, true); ++y) {
                        for (size_t z = 0; z < 8 && !screen_blocked_ && room(2, true); ++z) {
                            for (uint32_t direction = 0; direction < 2 && !screen_blocked_ && room(2, true); ++direction) {
                                if (x >= xs[direction].size() || y >= ys[direction].size() ||
                                    z >= zs[direction].size()) continue;
                                Choice candidate;
                                candidate.size = 3;
                                candidate.reverse = direction != 0;
                                candidate.ranks = {{source, b, c}};
                                candidate.slots = {{xs[direction][x].slot, ys[direction][y].slot, zs[direction][z].slot}};
                                const auto delta = state_.cycle_delta(source, candidate.slots[0], b, candidate.slots[1],
                                                                      c, candidate.slots[2], candidate.reverse);
                                ++result_.stats.cycle_evaluations;
                                result_.evaluations += 2;
                                candidate.value = proposed(candidate.ranks, delta, 3);
                                consider(candidate, choice);
                            }
                        }
                    }
                }
            }
        }
        commit(choice);
        return scanned;
    }

    const Samples& samples_;
    const Config& config_;
    State state_;
    uint32_t gpus_, capacity_;
    uint64_t budget_, screen_budget_, cycle_budget_, peak0_ = 0;
    Result result_;
    Metrics current_, best_;
    Buckets best_buckets_;
    std::vector<uint64_t> best_volumes_, scratch_;
    std::vector<uint32_t> source_cursor_, return_cursor_, destination_cursor_, priority_;
    std::vector<uint64_t> revisions_;
    std::vector<ReturnCache> return_cache_;
    size_t priority_cursor_ = 0;
    uint64_t epoch_evaluations_ = 0, epoch_accepted_ = 0;
    uint32_t stalled_ = 0;
    bool epoch_open_ = false, epoch_coverage_ = false, epoch_cycles_ = false, screen_blocked_ = false;
};

} // namespace ratio_detail

inline Result optimize_ratio(const Samples& samples, const Buckets& initial, const Config& config) {
    ratio_detail::validate(samples, initial, config);
    return ratio_detail::Search<>(samples, initial, config).run();
}

inline Result optimize(const Samples& samples, const Buckets& initial, const Config& config) {
    if (config.algorithm == Algorithm::Ratio) return optimize_ratio(samples, initial, config);
    require(config.algorithm == Algorithm::Weighted, "Unknown swap algorithm");
    require(config.source_samples && config.candidates && config.targets,
            "Candidate search sizes must be positive");
    State state(samples, initial);
    Result result;
    result.before = state.volumes();
    const uint32_t gpus = static_cast<uint32_t>(initial.size());
    const uint32_t capacity = static_cast<uint32_t>(initial.front().size());
    const uint64_t total0 = std::accumulate(result.before.begin(), result.before.end(), uint64_t(0));
    const uint64_t peak0 = *std::max_element(result.before.begin(), result.before.end());
    require(config.evaluations_per_gpu <= std::numeric_limits<uint64_t>::max() / gpus,
            "Search evaluation budget overflow");
    const uint64_t budget = config.evaluations_per_gpu * gpus;
    uint64_t total = total0;
    long double sum_squares = 0;
    for (uint64_t volume : result.before) sum_squares += squared(volume);
    const auto objective = [&](uint64_t sum, uint64_t peak) {
        return static_cast<long double>(sum) + static_cast<long double>(config.weight) * gpus * peak;
    };
    long double value = objective(total0, peak0);

    for (uint32_t round = 0; round < config.rounds && gpus > 1 && capacity && result.evaluations < budget; ++round) {
        const uint64_t accepted_before = result.accepted;
        std::vector<uint32_t> ranks(gpus);
        std::iota(ranks.begin(), ranks.end(), 0);
        std::stable_sort(ranks.begin(), ranks.end(), [&](uint32_t a, uint32_t b) {
            return state.volumes()[a] > state.volumes()[b];
        });
        if (config.weight == 0) std::rotate(ranks.begin(), ranks.begin() + round % gpus, ranks.end());
        for (uint32_t source : ranks) {
            if (result.evaluations >= budget) break;
            const uint32_t scan = uint32_t(std::min<uint64_t>(capacity, uint64_t(config.source_samples) * 2));
            std::vector<std::pair<uint32_t, uint32_t>> sources;
            sources.reserve(scan);
            for (uint32_t k = 0; k < scan; ++k) {
                const uint32_t slot = uint32_t((uint64_t(k) * capacity / scan + uint64_t(round) * 251) % capacity);
                sources.emplace_back(state.removable(source, slot), slot);
            }
            std::sort(sources.begin(), sources.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            if (sources.size() > config.source_samples) sources.resize(config.source_samples);
            for (const auto& candidate : sources) {
                if (result.evaluations >= budget) break;
                const uint32_t slot = candidate.second;
                const Sample& outgoing = samples[state.buckets()[source][slot]];
                std::vector<uint32_t> local_hits(gpus, 0);
                for (const Id id : outgoing) ++local_hits[id.owner];
                std::vector<uint32_t> low(gpus), local(gpus);
                std::iota(low.begin(), low.end(), 0);
                std::iota(local.begin(), local.end(), 0);
                std::stable_sort(low.begin(), low.end(), [&](uint32_t a, uint32_t b) {
                    return state.volumes()[a] < state.volumes()[b];
                });
                std::stable_sort(local.begin(), local.end(), [&](uint32_t a, uint32_t b) {
                    return local_hits[a] != local_hits[b] ? local_hits[a] > local_hits[b] :
                           state.volumes()[a] < state.volumes()[b];
                });
                std::vector<uint32_t> destinations;
                const uint32_t target_count = std::min(config.targets, gpus - 1);
                const auto append = [&](uint32_t d) {
                    if (d != source && destinations.size() < target_count &&
                        std::find(destinations.begin(), destinations.end(), d) == destinations.end())
                        destinations.push_back(d);
                };
                if (round % 2) append((source + round) % gpus);
                for (size_t k = 0; k < gpus && destinations.size() < target_count; ++k) {
                    append(low[k]);
                    append(local[k]);
                }
                uint32_t best_rank = source, best_slot = slot;
                uint64_t best_total = total;
                long double best_value = value, best_squares = sum_squares;
                for (uint32_t destination : destinations) {
                    const uint32_t trials = std::min(config.candidates, capacity);
                    const uint64_t seed = mixed((uint64_t(source) << 48) ^ (uint64_t(slot) << 16) ^
                                               (uint64_t(destination) << 32) ^ round);
                    for (uint32_t k = 0; k < trials && result.evaluations < budget; ++k) {
                        const uint32_t other = uint32_t((uint64_t(k) * capacity / trials + seed % capacity) % capacity);
                        const auto delta = state.swap_delta(source, slot, destination, other);
                        ++result.evaluations;
                        const int64_t new_total_signed = int64_t(total) + delta.first + delta.second;
                        require(new_total_signed >= 0, "Negative communication total");
                        const uint64_t new_total = uint64_t(new_total_signed);
                        if (new_total > total0) continue;
                        const auto& volumes = state.volumes();
                        const uint64_t a = uint64_t(int64_t(volumes[source]) + delta.first);
                        const uint64_t b = uint64_t(int64_t(volumes[destination]) + delta.second);
                        uint64_t new_peak = std::max(a, b);
                        for (uint32_t d = 0; d < gpus; ++d)
                            if (d != source && d != destination) new_peak = std::max(new_peak, volumes[d]);
                        if (new_peak > peak0) continue;
                        const long double new_value = objective(new_total, new_peak);
                        const long double new_squares = sum_squares - squared(volumes[source]) -
                            squared(volumes[destination]) + squared(a) + squared(b);
                        if (new_value < best_value || (new_value == best_value && new_squares < best_squares)) {
                            best_value = new_value;
                            best_squares = new_squares;
                            best_total = new_total;
                            best_rank = destination;
                            best_slot = other;
                        }
                    }
                }
                if (best_rank != source) {
                    state.apply_swap(source, slot, best_rank, best_slot);
                    total = best_total;
                    value = best_value;
                    sum_squares = best_squares;
                    ++result.accepted;
                }
            }
        }
        if (result.accepted == accepted_before) break;
    }
    result.buckets = state.buckets();
    result.after = count_remote(samples, result.buckets);
    validate_buckets(samples, result.buckets);
    require(result.after == state.volumes(), "Full recount disagrees with incremental optimizer");
    require(std::accumulate(result.after.begin(), result.after.end(), uint64_t(0)) <= total0 &&
            *std::max_element(result.after.begin(), result.after.end()) <= peak0,
            "Optimizer exceeded baseline communication bounds");
    return result;
}

}
