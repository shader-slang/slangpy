// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/persistent_cache.h"

#include <algorithm>
#include <map>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("persistent_cache");

using Blob = std::vector<uint8_t>;

struct CacheEntry {
    Blob key;
    Blob value;
};

using Cache = std::map<Blob, Blob>;

Blob random_data(size_t size)
{
    Blob data(size);
    std::generate(data.begin(), data.end(), []() { return static_cast<uint8_t>(rand() % 256); });
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

#define CACHE_CHECK(expr)                                                                                              \
    do {                                                                                                               \
        bool res = (expr);                                                                                             \
        if (!res)                                                                                                      \
            MESSAGE(doctest::String(cache.last_error().c_str()));                                                      \
        CHECK(res);                                                                                                    \
    } while (0)

#define CACHE_CHECK_STATS(cache, expected_entries, expected_size)                                                      \
    do {                                                                                                               \
        auto stats = cache.stats();                                                                                    \
        CHECK(stats.entries == (expected_entries));                                                                    \
        CHECK(stats.size == (expected_size));                                                                          \
    } while (0)

TEST_CASE("simple")
{
    auto cache_dir = testing::get_test_temp_directory() / "cache";
    PersistentCache cache;
    CACHE_CHECK(cache.open(cache_dir));

    Blob key1 = random_data(32);
    Blob value1 = random_data(128);
    Blob key2 = random_data(32);
    Blob value2 = random_data(256);

    std::vector<uint8_t> temp_value;

    // Check initial state of the cache
    CACHE_CHECK_STATS(cache, 0, 0);

    // Make sure key1 and key2 do not exist in the cache
    CACHE_CHECK(cache.get(key1, temp_value) == false);
    CACHE_CHECK(cache.get(key2, temp_value) == false);

    // Set key1 and value1
    CACHE_CHECK(cache.set(key1, value1));

    // Check cache stats after setting key1
    CACHE_CHECK_STATS(cache, 1, 128);

    // Make sure key1 exists and has the correct value
    CACHE_CHECK(cache.get(key1, temp_value));
    CACHE_CHECK(temp_value == value1);

    // Make sure key2 still does not exist
    CACHE_CHECK(cache.get(key2, temp_value) == false);

    // Set key2 and value2
    CACHE_CHECK(cache.set(key2, value2));

    // Check cache stats after setting key2
    CACHE_CHECK_STATS(cache, 2, 128 + 256);

    // Make sure key2 exists and has the correct value
    CACHE_CHECK(cache.get(key2, temp_value));
    CACHE_CHECK(temp_value == value2);

    // Overwrite key1 with a new value
    Blob new_value1 = random_data(512);
    CACHE_CHECK(cache.set(key1, new_value1));

    // Check cache stats after overwriting key1
    CACHE_CHECK_STATS(cache, 2, 512 + 256);

    // Make sure key1 has the new value
    CACHE_CHECK(cache.get(key1, temp_value));
    CACHE_CHECK(temp_value == new_value1);

    // Delete key2
    CACHE_CHECK(cache.del(key2));

    // Check cache stats after deleting key2
    CACHE_CHECK_STATS(cache, 1, 512);

    // Make sure key2 does not exist anymore
    CACHE_CHECK(cache.get(key2, temp_value) == false);

    // Delete key1
    CACHE_CHECK(cache.del(key1));

    // Check cache stats after deleting key1
    CACHE_CHECK_STATS(cache, 0, 0);

    // Make sure key1 does not exist anymore
    CACHE_CHECK(cache.get(key1, temp_value) == false);
}

TEST_CASE("persistent")
{
    auto cache_dir = testing::get_test_temp_directory() / "cache";
    PersistentCache cache;
    CACHE_CHECK(cache.open(cache_dir));

    std::vector<CacheEntry> entries = generate_random_entries(1000);

    // fill cache
    for (const auto& entry : entries) {
        CACHE_CHECK(cache.set(entry.key, entry.value));
    }

    // verify cache
    size_t total_size = 0;
    for (const auto& entry : entries) {
        Blob value;
        CACHE_CHECK(cache.get(entry.key, value));
        CACHE_CHECK(value == entry.value);
        total_size += entry.value.size();
    }

    // check cache stats
    CACHE_CHECK_STATS(cache, entries.size(), total_size);

    // close and reopen cache
    cache.close();
    cache.open(cache_dir);

    // verify cache
    for (const auto& entry : entries) {
        Blob value;
        CACHE_CHECK(cache.get(entry.key, value));
        CACHE_CHECK(value == entry.value);
    }

    // check cache stats
    CACHE_CHECK_STATS(cache, entries.size(), total_size);
}

TEST_SUITE_END();
