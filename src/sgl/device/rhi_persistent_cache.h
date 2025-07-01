// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/persistent_cache.h"

#include <slang-rhi.h>

namespace sgl {

/// Wrapper around `PersistentCache` that implements the `rhi::IPersistentCache` interface.
class RHIPersistentCache : public PersistentCache, public rhi::IPersistentCache {
    SGL_OBJECT(RHIPersistentCache)
public:
    RHIPersistentCache(const std::filesystem::path& path)
        : PersistentCache(path)
    {
    }

    // ISlangUnknown interface
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid, void** outObject) override
    {
        *outObject = nullptr;
        if (uuid == ISlangUnknown::getTypeGuid() || uuid == rhi::IPersistentCache::getTypeGuid())
            *outObject = this;
        return SLANG_OK;
    }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override
    {
        inc_ref();
        return static_cast<uint32_t>(ref_count());
    }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override
    {
        dec_ref();
        return static_cast<uint32_t>(ref_count());
    }

    // IPersistentCache interface
    virtual SLANG_NO_THROW rhi::Result SLANG_MCALL writeCache(ISlangBlob* key, ISlangBlob* data) override
    {
        return set(key->getBufferPointer(), key->getBufferSize(), data->getBufferPointer(), data->getBufferSize())
            ? SLANG_OK
            : SLANG_FAIL;
    }
    virtual SLANG_NO_THROW rhi::Result SLANG_MCALL queryCache(ISlangBlob* key, ISlangBlob** outData) override
    {
        struct Context {
            Slang::ComPtr<ISlangBlob> value;
            rhi::Result result{SLANG_FAIL};
        } context;
        if (!get(
                key->getBufferPointer(),
                key->getBufferSize(),
                [](const void* data, size_t size, void* user_data)
                {
                    Context* ctx = static_cast<Context*>(user_data);
                    rhi::getRHI()->createBlob(data, size, ctx->value.writeRef());
                },
                &context
            ))
            return SLANG_E_NOT_FOUND;
        *outData = context.value.detach();
        return context.result;
    }
};

} // namespace sgl
