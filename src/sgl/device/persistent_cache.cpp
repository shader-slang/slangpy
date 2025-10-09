// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "persistent_cache.h"

#include "sgl/core/error.h"
#include "sgl/core/logger.h"

namespace sgl {

PersistentCache::PersistentCache(const std::filesystem::path& path, size_t max_size)
    : m_path(path)
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
    m_cache.reset();
}

PersistentCacheStats PersistentCache::stats() const
{
    return {
        .entry_count = m_cache->stats().entries,
        .hit_count = m_hit_count.load(),
        .miss_count = m_miss_count.load(),
    };
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
        m_cache->set(key->getBufferPointer(), key->getBufferSize(), data->getBufferPointer(), data->getBufferSize());
        return SLANG_OK;
    } catch (const LMDBException& e) {
        log_error("Failed to write to cache in \"{}\": {}", m_path, e.what());
    }
    return SLANG_FAIL;
}

rhi::Result PersistentCache::queryCache(ISlangBlob* key, ISlangBlob** outData)
{
    try {
        struct Context {
            Slang::ComPtr<ISlangBlob> value;
            rhi::Result result{SLANG_E_NOT_FOUND};
        } context;
        bool success = m_cache->get(
            key->getBufferPointer(),
            key->getBufferSize(),
            [](const void* data, size_t size, void* user_data)
            {
                Context* ctx = static_cast<Context*>(user_data);
                ctx->result = rhi::getRHI()->createBlob(data, size, ctx->value.writeRef());
            },
            &context
        );

        if (success)
            m_hit_count.fetch_add(1);
        else
            m_miss_count.fetch_add(1);

        if (SLANG_SUCCEEDED(context.result))
            *outData = context.value.detach();
        return context.result;
    } catch (const LMDBException& e) {
        log_error("Failed to write to cache in \"{}\": {}", m_path, e.what());
    }
    return SLANG_FAIL;
}

} // namespace sgl
