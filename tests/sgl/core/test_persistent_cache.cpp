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

TEST_CASE("simple")
{
    auto cache_dir = testing::get_test_temp_directory() / "cache";
    ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir);

    Blob key1 = random_data(32);
    Blob value1 = random_data(128);
    Blob key2 = random_data(32);
    Blob value2 = random_data(256);

    std::vector<uint8_t> temp_value;

    // Make sure key1 and key2 do not exist in the cache
    CHECK(cache->get(key1, temp_value) == false);
    CHECK(cache->get(key2, temp_value) == false);

    // Set key1 and value1
    CHECK(cache->set(key1, value1));

    // Make sure key1 exists and has the correct value
    CHECK(cache->get(key1, temp_value));
    CHECK(temp_value == value1);

    // Make sure key2 still does not exist
    CHECK(cache->get(key2, temp_value) == false);

    // Set key2 and value2
    CHECK(cache->set(key2, value2));
    // Make sure key2 exists and has the correct value
    CHECK(cache->get(key2, temp_value));
    CHECK(temp_value == value2);

    // Overwrite key1 with a new value
    Blob new_value1 = random_data(512);
    CHECK(cache->set(key1, new_value1));

    // Make sure key1 has the new value
    CHECK(cache->get(key1, temp_value));
    CHECK(temp_value == new_value1);

    // Delete key2
    CHECK(cache->del(key2));

    // Make sure key2 does not exist anymore
    CHECK(cache->get(key2, temp_value) == false);

    // Delete key1
    CHECK(cache->del(key1));

    // Make sure key1 does not exist anymore
    CHECK(cache->get(key1, temp_value) == false);
}

TEST_CASE("persistent")
{
    auto cache_dir = testing::get_test_temp_directory() / "cache";
    ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir);

    std::vector<CacheEntry> entries = generate_random_entries(1000);

    // fill cache
    for (const auto& entry : entries) {
        CHECK(cache->set(entry.key, entry.value));
    }

    // verify cache
    for (const auto& entry : entries) {
        Blob value;
        CHECK(cache->get(entry.key, value));
        CHECK(value == entry.value);
    }

    // reload cache
    cache = make_ref<PersistentCache>(cache_dir);

    // verify cache
    for (const auto& entry : entries) {
        Blob value;
        CHECK(cache->get(entry.key, value));
        CHECK(value == entry.value);
    }
}

TEST_SUITE_END();
