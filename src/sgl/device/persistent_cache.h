// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/lmdb_cache.h"

#include <slang-rhi.h>

namespace sgl {

/// Wrapper around `LMDBCache` that implements the `rhi::IPersistentCache` interface.
class SGL_API PersistentCache : public Object, public rhi::IPersistentCache {
    SGL_OBJECT(PersistentCache)
public:
    PersistentCache(const std::filesystem::path& path, size_t max_size = 64ull * 1024 * 1024);
    ~PersistentCache() override;

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
};

} // namespace sgl
