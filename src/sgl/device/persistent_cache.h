// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/fwd.h"
#include "sgl/core/crypto.h"
#include "sgl/device/fwd.h"

#include <slang-rhi.h>

#include <atomic>
#include <map>
#include <mutex>
#include <vector>

namespace sgl {

struct PersistentCacheStats {
    uint64_t entry_count;
    uint64_t hit_count;
    uint64_t miss_count;
};

/// Wrapper around `LMDBCache` that implements the `rhi::IPersistentCache` interface.
class SGL_API PersistentCache : public Object, public rhi::IPersistentCache {
    SGL_OBJECT(PersistentCache)
public:
    PersistentCache(
        const std::filesystem::path& path,
        size_t max_size = 64ull * 1024 * 1024,
        ref<CacheWriter> cache_writer = nullptr
    );
    ~PersistentCache() override;

    PersistentCacheStats stats() const;

    void flush() const;

    /// Require future cache reads for this key to match validated code; mismatches become misses.
    void expect_entry(ISlangBlob* key, ISlangBlob* data);

    // ISlangUnknown interface
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid, void** outObject) override;
    // We don't want RHI to do reference counting on this object.
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 2; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 2; }

    // IPersistentCache interface
    virtual SLANG_NO_THROW rhi::Result SLANG_MCALL writeCache(ISlangBlob* key, ISlangBlob* data) override;
    virtual SLANG_NO_THROW rhi::Result SLANG_MCALL queryCache(ISlangBlob* key, ISlangBlob** outData) override;

private:
    std::filesystem::path m_path;
    ref<LMDBCache> m_cache;
    ref<CacheWriter> m_cache_writer;

    std::mutex m_expected_entries_mutex;
    std::map<std::vector<uint8_t>, SHA1::Digest> m_expected_entries;

    std::atomic<uint64_t> m_hit_count{0};
    std::atomic<uint64_t> m_miss_count{0};
};

} // namespace sgl
