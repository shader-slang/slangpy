// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/cache_writer.h"
#include "sgl/device/persistent_cache.h"

#include <slang-rhi.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("persistent_cache");

using ByteBlob = std::vector<uint8_t>;

static Slang::ComPtr<ISlangBlob> make_blob(const ByteBlob& data)
{
    Slang::ComPtr<ISlangBlob> blob;
    REQUIRE(SLANG_SUCCEEDED(rhi::getRHI()->createBlob(data.data(), data.size(), blob.writeRef())));
    return blob;
}

static ByteBlob copy_blob(ISlangBlob* blob)
{
    REQUIRE(blob);
    ByteBlob result(blob->getBufferSize());
    if (!result.empty())
        std::memcpy(result.data(), blob->getBufferPointer(), result.size());
    return result;
}

TEST_CASE("cache_writer_flush")
{
    ref<CacheWriter> writer = make_ref<CacheWriter>();
    std::atomic<int> value{0};

    writer->enqueue(
        1,
        [&]
        {
            value.store(42);
        }
    );
    writer->flush();

    CHECK(value.load() == 42);
}

TEST_CASE("cache_writer_prepare_failure_releases_reservation")
{
    ref<CacheWriter> writer = make_ref<CacheWriter>(1);

    CHECK_THROWS(writer->enqueue(
        1,
        []
        {
            throw std::runtime_error("prepare failed");
        },
        []
        {
        }
    ));

    std::atomic<bool> value{false};
    CHECK(writer->enqueue(
        1,
        [&]
        {
            value.store(true);
        }
    ));
    writer->flush();
    CHECK(value.load());
}

TEST_CASE("cache_writer_oversized_job_is_rejected")
{
    ref<CacheWriter> writer = make_ref<CacheWriter>(1);
    std::atomic<bool> value{false};

    CHECK(
        writer->enqueue(
            2,
            [&]
            {
                value.store(true);
            }
        )
        == false
    );
    writer->flush();
    CHECK(value.load() == false);

    CHECK(writer->enqueue(
        1,
        [&]
        {
            value.store(true);
        }
    ));
    writer->flush();
    CHECK(value.load());
}

TEST_CASE("cache_writer_flush_waits_for_reserved_job")
{
    ref<CacheWriter> writer = make_ref<CacheWriter>();

    std::promise<void> prepare_started;
    std::promise<void> release_prepare;
    auto prepare_started_future = prepare_started.get_future();
    auto release_prepare_future = release_prepare.get_future().share();
    std::atomic<bool> value{false};

    auto enqueue_future = std::async(
        std::launch::async,
        [&]
        {
            return writer->enqueue(
                1,
                [&]
                {
                    prepare_started.set_value();
                    release_prepare_future.wait();
                },
                [&]
                {
                    value.store(true);
                }
            );
        }
    );
    prepare_started_future.wait();

    auto flush_future = std::async(
        std::launch::async,
        [&]
        {
            writer->flush();
        }
    );
    CHECK(flush_future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout);

    release_prepare.set_value();
    CHECK(enqueue_future.get());
    flush_future.get();
    CHECK(value.load());
}

TEST_CASE("async_write_visible_after_flush_and_reopen")
{
    const std::filesystem::path cache_dir = testing::get_case_temp_directory() / "cache";
    const ByteBlob key{1, 2, 3, 4};
    const ByteBlob value{5, 6, 7, 8, 9};

    {
        ref<CacheWriter> writer = make_ref<CacheWriter>();
        ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir, 1024 * 1024, writer);
        Slang::ComPtr<ISlangBlob> key_blob = make_blob(key);
        Slang::ComPtr<ISlangBlob> value_blob = make_blob(value);

        std::promise<void> blocker_started;
        std::promise<void> release_blocker;
        auto blocker_started_future = blocker_started.get_future();
        auto release_blocker_future = release_blocker.get_future().share();

        REQUIRE(writer->enqueue(
            1,
            [&]
            {
                blocker_started.set_value();
                release_blocker_future.wait();
            }
        ));
        blocker_started_future.wait();

        CHECK(cache->writeCache(key_blob, value_blob) == SLANG_OK);

        Slang::ComPtr<ISlangBlob> queried;
        rhi::Result query_result = cache->queryCache(key_blob, queried.writeRef());
        CHECK(query_result != SLANG_OK);
        CHECK(queried == nullptr);

        release_blocker.set_value();
        cache->flush();

        query_result = cache->queryCache(key_blob, queried.writeRef());
        REQUIRE(query_result == SLANG_OK);
        CHECK(copy_blob(queried) == value);

        PersistentCacheStats stats = cache->stats();
        CHECK(stats.entry_count == 1);
        CHECK(stats.hit_count == 1);
        CHECK(stats.miss_count == 1);
    }

    {
        ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir, 1024 * 1024);
        Slang::ComPtr<ISlangBlob> key_blob = make_blob(key);
        Slang::ComPtr<ISlangBlob> queried;

        REQUIRE(cache->queryCache(key_blob, queried.writeRef()) == SLANG_OK);
        CHECK(copy_blob(queried) == value);
        cache->flush();
    }
}

TEST_CASE("oversized_key_is_rejected_without_pending_entry")
{
    const std::filesystem::path cache_dir = testing::get_case_temp_directory() / "cache";
    const ByteBlob key(4096, 1);
    const ByteBlob value{2, 3, 4};

    ref<CacheWriter> writer = make_ref<CacheWriter>();
    ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir, 1024 * 1024, writer);
    Slang::ComPtr<ISlangBlob> key_blob = make_blob(key);
    Slang::ComPtr<ISlangBlob> value_blob = make_blob(value);

    CHECK(cache->writeCache(key_blob, value_blob) == SLANG_FAIL);
    cache->flush();

    Slang::ComPtr<ISlangBlob> queried;
    CHECK(cache->queryCache(key_blob, queried.writeRef()) != SLANG_OK);
    CHECK(queried == nullptr);
}

TEST_CASE("query_null_out_data_returns_failure")
{
    const std::filesystem::path cache_dir = testing::get_case_temp_directory() / "cache";
    const ByteBlob key{5, 6, 7, 8};

    ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir, 1024 * 1024);
    Slang::ComPtr<ISlangBlob> key_blob = make_blob(key);

    CHECK(cache->queryCache(key_blob, nullptr) == SLANG_FAIL);
}

TEST_CASE("same_key_queued_writes_commit_latest_value")
{
    const std::filesystem::path cache_dir = testing::get_case_temp_directory() / "cache";
    const ByteBlob key{20, 21, 22, 23};
    const ByteBlob value1{24, 25, 26};
    const ByteBlob value2{27, 28, 29, 30};

    ref<CacheWriter> writer = make_ref<CacheWriter>();
    ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir, 1024 * 1024, writer);
    Slang::ComPtr<ISlangBlob> key_blob = make_blob(key);
    Slang::ComPtr<ISlangBlob> value1_blob = make_blob(value1);
    Slang::ComPtr<ISlangBlob> value2_blob = make_blob(value2);

    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    auto blocker_started_future = blocker_started.get_future();
    auto release_blocker_future = release_blocker.get_future().share();

    REQUIRE(writer->enqueue(
        1,
        [&]
        {
            blocker_started.set_value();
            release_blocker_future.wait();
        }
    ));
    blocker_started_future.wait();

    CHECK(cache->writeCache(key_blob, value1_blob) == SLANG_OK);
    CHECK(cache->writeCache(key_blob, value2_blob) == SLANG_OK);

    Slang::ComPtr<ISlangBlob> queried;
    rhi::Result query_result = cache->queryCache(key_blob, queried.writeRef());
    CHECK(query_result != SLANG_OK);
    CHECK(queried == nullptr);

    release_blocker.set_value();
    cache->flush();

    REQUIRE(cache->queryCache(key_blob, queried.writeRef()) == SLANG_OK);
    CHECK(copy_blob(queried) == value2);
}

TEST_CASE("query_hit_updates_metadata_while_writer_is_busy")
{
    const std::filesystem::path cache_dir = testing::get_case_temp_directory() / "cache";
    const ByteBlob queried_key{10};
    const ByteBlob trigger_key{255};
    const ByteBlob value(128 * 1024, 14);
    const ByteBlob trigger_value(128 * 1024, 15);

    ref<CacheWriter> writer = make_ref<CacheWriter>();
    ref<PersistentCache> cache = make_ref<PersistentCache>(cache_dir, 8ull * 1024 * 1024, writer);

    Slang::ComPtr<ISlangBlob> queried_key_blob = make_blob(queried_key);
    Slang::ComPtr<ISlangBlob> trigger_key_blob = make_blob(trigger_key);
    Slang::ComPtr<ISlangBlob> value_blob = make_blob(value);
    Slang::ComPtr<ISlangBlob> trigger_value_blob = make_blob(trigger_value);

    REQUIRE(cache->writeCache(queried_key_blob, value_blob) == SLANG_OK);
    for (uint8_t i = 11; i < 54; ++i) {
        Slang::ComPtr<ISlangBlob> key_blob = make_blob(ByteBlob{i});
        REQUIRE(cache->writeCache(key_blob, value_blob) == SLANG_OK);
    }
    cache->flush();

    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    auto blocker_started_future = blocker_started.get_future();
    auto release_blocker_future = release_blocker.get_future().share();

    REQUIRE(writer->enqueue(
        1,
        [&]
        {
            blocker_started.set_value();
            release_blocker_future.wait();
        }
    ));
    blocker_started_future.wait();

    Slang::ComPtr<ISlangBlob> queried;
    rhi::Result query_result = cache->queryCache(queried_key_blob, queried.writeRef());
    CHECK(query_result == SLANG_OK);
    if (query_result == SLANG_OK)
        CHECK(copy_blob(queried) == value);

    release_blocker.set_value();
    cache->flush();

    REQUIRE(cache->writeCache(trigger_key_blob, trigger_value_blob) == SLANG_OK);
    cache->flush();
    CHECK(cache->stats().entry_count < 45);

    queried.setNull();
    REQUIRE(cache->queryCache(queried_key_blob, queried.writeRef()) == SLANG_OK);
    CHECK(copy_blob(queried) == value);
}

TEST_SUITE_END();
