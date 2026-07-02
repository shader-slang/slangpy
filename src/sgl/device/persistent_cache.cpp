// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "persistent_cache.h"

#include "sgl/core/error.h"
#include "sgl/core/logger.h"
#include "sgl/core/lmdb_cache.h"
#include "sgl/device/cache_writer.h"

#include <cstring>
#include <memory>
#include <vector>

namespace sgl {

static std::vector<uint8_t> copy_blob(ISlangBlob* blob)
{
    SGL_CHECK(blob, "Cannot copy a null cache blob.");
    std::vector<uint8_t> result(blob->getBufferSize());
    if (!result.empty())
        std::memcpy(result.data(), blob->getBufferPointer(), result.size());
    return result;
}

PersistentCache::PersistentCache(const std::filesystem::path& path, size_t max_size, ref<CacheWriter> cache_writer)
    : m_path(path)
    , m_cache_writer(cache_writer ? std::move(cache_writer) : make_ref<CacheWriter>())
{
    LMDBCache::Options options{
        .max_size = max_size,
    };
    const int MAX_ATTEMPTS = 3;
    for (int attempt = 1; attempt <= MAX_ATTEMPTS; ++attempt) {
        try {
            m_cache = make_ref<LMDBCache>(m_path, options);
            break;
        } catch (const std::exception& e) {
            log_error("Failed to open cache in \"{}\" (attempt {}/{}): {} ", m_path, attempt, MAX_ATTEMPTS, e.what());
            if (attempt < MAX_ATTEMPTS) {
                // Try deleting the cache directory so next attempt creates a new cache.
                std::error_code ec;
                std::filesystem::remove_all(m_path, ec);
                if (ec) {
                    log_warn("Failed to delete cache directory \"{}\": {}", m_path, ec.message());
                }
            }
        }
    }
    if (!m_cache) {
        SGL_THROW("Failed to open cache in \"{}\" after {} attempts!", m_path, MAX_ATTEMPTS);
    }
}

PersistentCache::~PersistentCache()
{
    flush();
    m_cache.reset();
}

PersistentCacheStats PersistentCache::stats() const
{
    flush();
    return {
        .entry_count = m_cache->stats().entries,
        .hit_count = m_hit_count.load(),
        .miss_count = m_miss_count.load(),
    };
}

void PersistentCache::flush() const
{
    m_cache_writer->flush();
}

SlangResult PersistentCache::queryInterface(const SlangUUID& uuid, void** outObject)
{
    *outObject = nullptr;
    if (uuid == ISlangUnknown::getTypeGuid() || uuid == rhi::IPersistentCache::getTypeGuid())
        *outObject = this;
    return SLANG_OK;
}

rhi::Result PersistentCache::writeCache(ISlangBlob* key, ISlangBlob* data)
{
    try {
        SGL_CHECK(key, "Cannot write a cache entry with a null key.");
        SGL_CHECK(data, "Cannot write a cache entry with null data.");
        SGL_CHECK(key->getBufferSize() > 0, "Cache key size must be greater than 0.");
        SGL_CHECK(key->getBufferSize() <= m_cache->max_key_size(), "Cache key size exceeds maximum allowed size.");

        struct WriteState {
            std::vector<uint8_t> key;
            std::vector<uint8_t> value;
        };

        auto state = std::make_shared<WriteState>();
        const size_t byte_size = key->getBufferSize() + data->getBufferSize();

        if (!m_cache_writer->enqueue(
                byte_size,
                [key, data, state]
                {
                    state->key = copy_blob(key);
                    state->value = copy_blob(data);
                },
                [cache = m_cache, path = m_path, state]() mutable
                {
                    try {
                        cache->set(state->key, state->value);
                    } catch (const std::exception& e) {
                        log_warn("Failed to write to cache in \"{}\": {}", path, e.what());
                    } catch (...) {
                        log_warn("Failed to write to cache in \"{}\": unknown exception", path);
                    }
                }
            )) {
            log_warn("Failed to enqueue cache write in \"{}\".", m_path);
            return SLANG_FAIL;
        }

        return SLANG_OK;
    } catch (const std::exception& e) {
        log_error("Failed to write to cache in \"{}\": {}", m_path, e.what());
    }
    return SLANG_FAIL;
}

rhi::Result PersistentCache::queryCache(ISlangBlob* key, ISlangBlob** outData)
{
    try {
        SGL_CHECK(outData, "Cannot query cache with a null output blob pointer.");
        *outData = nullptr;

        std::vector<uint8_t> key_data = copy_blob(key);

        struct Context {
            Slang::ComPtr<ISlangBlob> value;
            rhi::Result result{SLANG_E_NOT_FOUND};
        } context;

        bool success = m_cache->get_readonly(
            key_data.data(),
            key_data.size(),
            [](const void* data, size_t size, void* user_data)
            {
                Context* ctx = static_cast<Context*>(user_data);
                ctx->result = rhi::getRHI()->createBlob(data, size, ctx->value.writeRef());
            },
            &context
        );

        if (success) {
            m_hit_count.fetch_add(1);
            const size_t touch_byte_size = key_data.size();
            if (!m_cache_writer->enqueue(
                    touch_byte_size,
                    [cache = m_cache, path = m_path, key_data = std::move(key_data)]
                    {
                        try {
                            cache->touch(key_data);
                        } catch (const std::exception& e) {
                            log_warn("Failed to touch cache entry in \"{}\": {}", path, e.what());
                        } catch (...) {
                            log_warn("Failed to touch cache entry in \"{}\": unknown exception", path);
                        }
                    }
                )) {
                log_warn("Failed to enqueue cache entry touch in \"{}\".", m_path);
            }
        } else {
            m_miss_count.fetch_add(1);
        }

        if (SLANG_SUCCEEDED(context.result))
            *outData = context.value.detach();
        return context.result;
    } catch (const std::exception& e) {
        log_error("Failed to query cache in \"{}\": {}", m_path, e.what());
    }
    return SLANG_FAIL;
}

} // namespace sgl
