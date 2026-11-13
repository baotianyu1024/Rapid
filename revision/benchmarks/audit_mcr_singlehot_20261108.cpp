// Exact cold-ID MCR on the 26 original columns, without synthetic multi-hot IDs.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

template<class T> struct Mapping {
    const T* data;
    size_t size;
    explicit Mapping(const char* path) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open input");
        struct stat statbuf;
        if (fstat(fd, &statbuf) || statbuf.st_size % sizeof(T))
            throw std::runtime_error("invalid input length");
        size = statbuf.st_size / sizeof(T);
        data = static_cast<const T*>(mmap(nullptr, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0));
        close(fd);
        if (data == MAP_FAILED) throw std::runtime_error("mmap failed");
    }
};

int main(int argc, char** argv) {
    if (argc != 4) throw std::runtime_error("usage: executable compact_rows.u32 frequency.u32 output.u32");
    Mapping<uint32_t> raw(argv[1]), freq(argv[2]);
    if (raw.size % 26) throw std::runtime_error("expected 26 columns");
    const size_t rows = raw.size / 26, keys = freq.size;
    const double cold_cutoff = rows * .0001;
    std::vector<uint64_t> begin(keys + 1, 0);
    std::vector<uint32_t> targets;
    for (size_t u = 0; u < keys; ++u) {
        begin[u+1] = begin[u] + (freq.data[u] < cold_cutoff ? freq.data[u] : 0);
        if (freq.data[u] >= 10 && freq.data[u] < cold_cutoff) targets.push_back(u);
    }
    auto cursor = begin;
    std::vector<uint32_t> posts(begin.back());
    for (size_t r = 0; r < rows; ++r)
        for (size_t t = 0; t < 26; ++t) {
            uint32_t u = raw.data[r*26+t];
            if (u >= keys) throw std::runtime_error("ID outside frequency array");
            if (freq.data[u] < cold_cutoff) posts[cursor[u]++] = r;
        }
    for (size_t u = 0; u < keys; ++u)
        if (cursor[u] != begin[u+1]) throw std::runtime_error("frequency mismatch");
    struct Result { uint32_t id, support, best, partner; };
    std::vector<Result> results(targets.size());
    std::cout << "rows=" << rows << " active_keys=" << keys << " targets=" << targets.size() << std::endl;
    #pragma omp parallel num_threads(16)
    {
        std::vector<uint16_t> counts(keys, 0);
        std::vector<uint32_t> touched;
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < targets.size(); ++i) {
            uint32_t u = targets[i], best = 0, partner = UINT32_MAX;
            touched.clear();
            for (uint64_t j = begin[u]; j < begin[u+1]; ++j) {
                const auto r = posts[j];
                for (size_t t = 0; t < 26; ++t) {
                    uint32_t v = raw.data[r*26+t];
                    if (v == u || freq.data[v] >= cold_cutoff) continue;
                    if (!counts[v]) touched.push_back(v);
                    uint32_t c = ++counts[v];
                    if (c > best) { best = c; partner = v; }
                }
            }
            for (auto v: touched) counts[v] = 0;
            results[i] = {u, freq.data[u], best, partner};
        }
    }
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(results.data()), results.size()*sizeof(Result));
    for (uint32_t threshold: {10, 20, 50, 100}) {
        uint64_t eligible = 0, high = 0;
        for (auto r: results) if (r.support >= threshold) {
            ++eligible;
            high += uint64_t(r.best)*100 > uint64_t(r.support)*95;
        }
        std::cout << "RESULT " << threshold << " " << eligible << " " << high << " "
                  << (eligible ? double(high)/eligible : 0) << std::endl;
    }
}
