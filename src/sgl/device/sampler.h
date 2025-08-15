// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/device_child.h"
#include "sgl/device/types.h"
#include "sgl/device/native_handle.h"

#include "sgl/core/macros.h"
#include "sgl/core/object.h"

#include "sgl/math/vector.h"

#include <slang-rhi.h>

namespace sgl {

struct SamplerDesc {
    TextureFilteringMode min_filter{TextureFilteringMode::linear};
    TextureFilteringMode mag_filter{TextureFilteringMode::linear};
    TextureFilteringMode mip_filter{TextureFilteringMode::linear};
    TextureReductionOp reduction_op{TextureReductionOp::average};
    TextureAddressingMode address_u{TextureAddressingMode::wrap};
    TextureAddressingMode address_v{TextureAddressingMode::wrap};
    TextureAddressingMode address_w{TextureAddressingMode::wrap};
    float mip_lod_bias{0.f};
    uint32_t max_anisotropy{1};
    ComparisonFunc comparison_func{ComparisonFunc::never};
    float4 border_color{1.f, 1.f, 1.f, 1.f};
    float min_lod{-1000.f};
    float max_lod{1000.f};
    std::string label;
};

class SGL_API Sampler : public DeviceChild {
    SGL_OBJECT(Sampler)
public:
    Sampler(ref<Device> device, SamplerDesc desc);
    ~Sampler();

    virtual void _release_rhi_resources() override { m_rhi_sampler.setNull(); }

    const SamplerDesc& desc() const { return m_desc; }

    DescriptorHandle descriptor_handle() const;

    rhi::ISampler* rhi_sampler() const { return m_rhi_sampler; }

    /// Get the native sampler handle.
    NativeHandle native_handle() const;

    std::string to_string() const override;

private:
    SamplerDesc m_desc;
    Slang::ComPtr<rhi::ISampler> m_rhi_sampler;
};

} // namespace sgl
