// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/lmdb_cache.h"
#include "sgl/core/timer.h"

#include <algorithm>
#include <vector>
#include <set>
#include <chrono>
#include <thread>
#include <mutex>

#define PRINT_DIAGNOSTICS 0

using namespace sgl;

TEST_SUITE_BEGIN("lmdb_cache");

using Blob = std::vector<uint8_t>;

struct CacheEntry {
    Blob key;
    Blob value;
    uint64_t last_access{0};
};

static uint32_t rng()
{
    static constexpr uint32_t A = 1664525u;
    static constexpr uint32_t C = 1013904223u;
    static uint32_t state = 0xdeadbeef;
    state = (A * state + C);
    return state;
}

Blob random_data(size_t size)
{
    Blob data(size);
    uint8_t* ptr = data.data();
    for (size_t i = 0; i < size; ++i)
        ptr[i] = static_cast<uint8_t>((rng() >> 24) & 0xff);
    return data;
}

std::vector<CacheEntry>
generate_random_entries(size_t count, size_t key_size = 32, size_t min_value_size = 64, size_t max_value_size = 1024)
{
    std::vector<CacheEntry> entries;
    entries.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        Blob key = random_data(key_size);
        size_t value_size = min_value_size + rand() % (max_value_size - min_value_size + 1);
        Blob value = random_data(value_size);
        entries.push_back({key, value});
    }
    return entries;
}

inline void print_stats(const LMDBCache::Stats& stats)
{
    fmt::println("Cache stats: entries={} size={} evictions={}", stats.entries, stats.size, stats.evictions);
}

inline void print_usage(const LMDBCache::Usage& usage)
{
    fmt::println(
        "Cache usage: reserved_size={} committed_size={} used_size={}",
        usage.reserved_size,
        usage.committed_size,
        usage.used_size
    );
}

#define CACHE_CHECK_STATS(cache, expected_entries, expected_size)                                                      \
    do {                                                                                                               \
        auto stats = cache.stats();                                                                                    \
        CHECK(stats.entries == (expected_entries));                                                                    \
        CHECK(stats.size == (expected_size));                                                                          \
        if (PRINT_DIAGNOSTICS) {                                                                                       \
            print_stats(stats);                                                                                        \
            print_usage(cache.usage());                                                                                \
        }                                                                                                              \
    } while (0)

TEST_CASE("simple")
{
    auto cache_dir = testing::get_test_temp_directory() / "cache";
    LMDBCache cache(cache_dir);

    Blob key1 = random_data(32);
    Blob value1 = random_data(128);
    Blob key2 = random_data(32);
    Blob value2 = random_data(256);

    std::vector<uint8_t> temp_value;

    // Check initial state of the cache
    CACHE_CHECK_STATS(cache, 0, 0);

    // Make sure key1 and key2 do not exist in the cache
    CHECK(cache.get(key1, temp_value) == false);
    CHECK(cache.del(key1) == false);
    CHECK(cache.get(key2, temp_value) == false);
    CHECK(cache.del(key2) == false);

    // Set key1 and value1
    cache.set(key1, value1);

    // Check cache stats after setting key1
    CACHE_CHECK_STATS(cache, 1, 128);

    // Make sure key1 exists and has the correct value
    CHECK(cache.get(key1, temp_value));
    CHECK(temp_value == value1);

    // Make sure key2 still does not exist
    CHECK(cache.get(key2, temp_value) == false);
    CHECK(cache.del(key2) == false);

    // Set key2 and value2
    cache.set(key2, value2);

    // Check cache stats after setting key2
    CACHE_CHECK_STATS(cache, 2, 128 + 256);

    // Make sure key2 exists and has the correct value
    CHECK(cache.get(key2, temp_value));
    CHECK(temp_value == value2);

    // Overwrite key1 with a new value
    Blob new_value1 = random_data(512);
    cache.set(key1, new_value1);

    // Check cache stats after overwriting key1
    CACHE_CHECK_STATS(cache, 2, 512 + 256);

    // Make sure key1 has the new value
    CHECK(cache.get(key1, temp_value));
    CHECK(temp_value == new_value1);

    // Delete key2
    CHECK(cache.del(key2));

    // Check cache stats after deleting key2
    CACHE_CHECK_STATS(cache, 1, 512);

    // Make sure key2 does not exist anymore
    CHECK(cache.get(key2, temp_value) == false);
    CHECK(cache.del(key2) == false);

    // Delete key1
    CHECK(cache.del(key1));

    // Check cache stats after deleting key1
    CACHE_CHECK_STATS(cache, 0, 0);

    // Make sure key1 does not exist anymore
    CHECK(cache.get(key1, temp_value) == false);
    CHECK(cache.del(key1) == false);
}

TEST_CASE("persistence")
{
    auto cache_dir = testing::get_case_temp_directory() / "cache";

    std::vector<CacheEntry> entries = generate_random_entries(1000);

    {
        LMDBCache cache(cache_dir);

        // fill cache
        for (const auto& entry : entries) {
            cache.set(entry.key, entry.value);
        }

        // verify cache
        size_t total_size = 0;
        for (const auto& entry : entries) {
            Blob value;
            CHECK(cache.get(entry.key, value));
            CHECK(value == entry.value);
            total_size += entry.value.size();
        }

        // check cache stats
        CACHE_CHECK_STATS(cache, entries.size(), total_size);
    }

    // close and reopen cache

    {
        LMDBCache cache(cache_dir);

        // verify cache
        size_t total_size = 0;
        for (const auto& entry : entries) {
            Blob value;
            CHECK(cache.get(entry.key, value));
            CHECK(value == entry.value);
            total_size += entry.value.size();
        }

        // check cache stats
        CACHE_CHECK_STATS(cache, entries.size(), total_size);
    }
}

inline uint64_t get_current_time_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}

/// Helper class to run a stress test.
/// When running, each iteration works as follows:
/// - with probability 'delete_ratio', delete a random entry in the cache
/// - otherwise, select a random entry, try to get it from the cache, if not found, set it
///   - with probability 'hot_ratio', choose one of the first 'hot_candidates' entries,
///     otherwise choose any of the remaining entries
///   - when we successfully accessed an entry or written it, remember the last access time
/// After running this test from one or multiple threads, we can verify the cache by
/// checking that the entries in the cache correspond to the most recently accessed entries.
struct StressTest {
    struct Options {
        std::filesystem::path path;
        /// Cache size (default: 32 MB)
        size_t cache_size{32 * 1024 * 1024};
        /// Key size in bytes.
        size_t key_size{40};
        /// Minimum value size in bytes.
        size_t min_value_size{512};
        /// Maximum value size in bytes.
        size_t max_value_size{64 * 1024};
        /// Number of candidate entries to generate.
        size_t candidate_count{5000};
        /// Number of "hot" candidates that are accessed more frequently.
        size_t hot_candidates{100};
        /// Ratio of accesses to "hot" candidates (0.0 - 1.0).
        double hot_ratio = 0.7;
        /// Ratio of delete vs access operations(0.0 - 1.0).
        double delete_ratio = 0.1;
    };

    struct RunStats {
        size_t hits;
        size_t misses;

        uint64_t total_set_calls;
        double total_set_time;
        uint64_t total_set_bytes;

        uint64_t total_get_calls;
        double total_get_time;
        uint64_t total_get_bytes;

        uint64_t total_del_calls;
        double total_del_time;

        void print()
        {
            fmt::println(
                "hitrate={:.1f}% hits={} misses={}", //
                100.0 * hits / (hits + misses),
                hits,
                misses
            );
            fmt::println(
                "set: {} calls | {:.1f} calls/s | {:.1f} MB/s", //
                total_set_calls,
                total_set_calls / total_set_time,
                (total_set_bytes * 1e-6) / total_set_time
            );
            fmt::println(
                "get: {} calls | {:.1f} calls/s | {:.1f} MB/s", //
                total_get_calls,
                total_get_calls / total_get_time,
                (total_get_bytes * 1e-6) / total_get_time
            );
            fmt::println(
                "del: {} calls | {:.1f} calls/s", //
                total_del_calls,
                total_del_calls / total_del_time
            );
        }

        static RunStats accumulate(std::span<const RunStats> runs)
        {
            RunStats result = {};
            for (const auto& stats : runs) {
                result.hits += stats.hits;
                result.misses += stats.misses;
                result.total_set_calls += stats.total_set_calls;
                result.total_set_time += stats.total_set_time;
                result.total_set_bytes += stats.total_set_bytes;
                result.total_get_calls += stats.total_get_calls;
                result.total_get_time += stats.total_get_time;
                result.total_get_bytes += stats.total_get_bytes;
                result.total_del_calls += stats.total_del_calls;
                result.total_del_time += stats.total_del_time;
            }
            return result;
        }
    };

    Options options;
    LMDBCache cache;
    std::vector<CacheEntry> entries;
    std::mutex mutex;

    StressTest(const Options& options_)
        : options(options_)
        , cache(options.path, LMDBCache::Options{.max_size = options.cache_size})
    {
        entries = generate_random_entries(
            options.candidate_count,
            options.key_size,
            options.min_value_size,
            options.max_value_size
        );
    }

    RunStats run(size_t iterations)
    {
        RunStats stats = {};
        std::vector<uint8_t> temp_value;
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
            if (PRINT_DIAGNOSTICS) {
                if (iteration % 1000 == 0) {
                    fmt::println("Iteration {}, hits={}, misses={}", iteration, stats.hits, stats.misses);
                    print_usage(cache.usage());
                }
            }
            double r = static_cast<double>(rand()) / RAND_MAX;
            if (r < options.delete_ratio) {
                // delete entry
                size_t entry_index = rand() % entries.size();
                auto& entry = entries[entry_index];
                Timer timer;
                bool success = cache.del(entry.key);
                stats.total_del_calls++;
                stats.total_del_time += timer.elapsed_s();
                if (success) {
                    std::lock_guard lock(mutex);
                    entry.last_access = 0;
                }
            } else {
                // access entry, write if not present
                r = static_cast<double>(rand()) / RAND_MAX;
                size_t entry_index = r < options.hot_ratio
                    ? (rand() % options.hot_candidates)
                    : (options.hot_candidates + rand() % (entries.size() - options.hot_candidates));
                auto& entry = entries[entry_index];
                // try to get the entry
                Timer timer;
                bool success = cache.get(entry.key, temp_value);
                stats.total_get_calls++;
                stats.total_get_time += timer.elapsed_s();
                stats.total_get_bytes += success ? temp_value.size() : 0;
                if (success) {
                    stats.hits++;
                    {
                        std::lock_guard lock(mutex);
                        entry.last_access = get_current_time_ns();
                    }
                } else {
                    stats.misses++;
                    // set the entry
                    try {
                        timer.reset();
                        cache.set(entry.key, entry.value);
                        stats.total_set_calls++;
                        stats.total_set_time += timer.elapsed_s();
                        stats.total_set_bytes += entry.value.size();
                        {
                            std::lock_guard lock(mutex);
                            entry.last_access = get_current_time_ns();
                        }
                    } catch (const std::exception& e) {
                        if (PRINT_DIAGNOSTICS) {
                            fmt::println("set operation failed: {}", e.what());
                        }
                    }
                }
            }
        }
        return stats;
    }

    /// Check that entries in the cache are the most recently accessed ones.
    void verify()
    {
        LMDBCache::Stats stats = cache.stats();

        std::vector<CacheEntry> expected_entries = entries;
        std::sort(
            expected_entries.begin(),
            expected_entries.end(),
            [](const CacheEntry& a, const CacheEntry& b) { return a.last_access > b.last_access; }
        );
        for (size_t i = 0; i < stats.entries; ++i) {
            Blob value;
            CHECK(cache.get(expected_entries[i].key, value));
            CHECK(value == expected_entries[i].value);
        }
    }
};

TEST_CASE("stress-single-threaded")
{
    StressTest::Options options;
    options.path = testing::get_case_temp_directory() / "cache";
    StressTest test(options);

    size_t iterations = 100000;
    StressTest::RunStats run_stats = test.run(iterations);

    test.verify();

    if (PRINT_DIAGNOSTICS) {
        print_stats(test.cache.stats());
        run_stats.print();
    }
}

TEST_CASE("stress-multi-threaded")
{
    StressTest::Options options;
    options.path = testing::get_case_temp_directory() / "cache";
    StressTest test(options);

    const size_t thread_count = 4;
    const size_t iterations_per_thread = 25000;

    std::vector<std::thread> threads;
    std::vector<StressTest::RunStats> thread_run_stats(thread_count);

    for (size_t i = 0; i < thread_count; ++i)
        threads.emplace_back([&, i]() { thread_run_stats[i] = test.run(iterations_per_thread); });
    for (auto& thread : threads)
        thread.join();

    test.verify();

    if (PRINT_DIAGNOSTICS) {
        print_stats(test.cache.stats());
        StressTest::RunStats run_stats = StressTest::RunStats::accumulate(thread_run_stats);
        run_stats.print();
    }
}

TEST_SUITE_END();
