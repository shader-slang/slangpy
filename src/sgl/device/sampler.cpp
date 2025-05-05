// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sampler.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"

#include "sgl/core/config.h"
#include "sgl/core/error.h"
#include "sgl/core/string.h"

namespace sgl {

Sampler::Sampler(ref<Device> device, SamplerDesc desc)
    : DeviceResource(std::move(device))
    , m_desc(std::move(desc))
{
    rhi::SamplerDesc rhi_desc{
        .minFilter = static_cast<rhi::TextureFilteringMode>(m_desc.min_filter),
        .magFilter = static_cast<rhi::TextureFilteringMode>(m_desc.mag_filter),
        .mipFilter = static_cast<rhi::TextureFilteringMode>(m_desc.mip_filter),
        .reductionOp = static_cast<rhi::TextureReductionOp>(m_desc.reduction_op),
        .addressU = static_cast<rhi::TextureAddressingMode>(m_desc.address_u),
        .addressV = static_cast<rhi::TextureAddressingMode>(m_desc.address_v),
        .addressW = static_cast<rhi::TextureAddressingMode>(m_desc.address_w),
        .mipLODBias = m_desc.mip_lod_bias,
        .maxAnisotropy = m_desc.max_anisotropy,
        .comparisonFunc = static_cast<rhi::ComparisonFunc>(m_desc.comparison_func),
        .borderColor = {m_desc.border_color.x, m_desc.border_color.y, m_desc.border_color.z, m_desc.border_color.w},
        .minLOD = m_desc.min_lod,
        .maxLOD = m_desc.max_lod,
        .label = m_desc.label.empty() ? nullptr : m_desc.label.c_str(),
    };
    SLANG_RHI_CALL(m_device->rhi_device()->createSampler(rhi_desc, m_rhi_sampler.writeRef()));
}

Sampler::~Sampler() { }

DescriptorHandle Sampler::descriptor_handle() const
{
    rhi::DescriptorHandle rhi_handle = {};
    m_rhi_sampler->getDescriptorHandle(&rhi_handle);
    return DescriptorHandle(rhi_handle);
}

NativeHandle Sampler::native_handle() const
{
    rhi::NativeHandle rhi_handle = {};
    m_rhi_sampler->getNativeHandle(&rhi_handle);
    return NativeHandle(rhi_handle);
}

std::string Sampler::to_string() const
{
    return fmt::format(
        "Sampler(\n"
        "  device = {},\n"
        "  min_filter = {},\n"
        "  mag_filter = {},\n"
        "  mip_filter = {},\n"
        "  reduction_op = {},\n"
        "  address_u = {},\n"
        "  address_v = {},\n"
        "  address_w = {},\n"
        "  mip_lod_bias = {},\n"
        "  max_anisotropy = {},\n"
        "  comparison_func = {},\n"
        "  border_color = {},\n"
        "  min_lod = {},\n"
        "  max_lod = {},\n"
        "  label = {}\n"
        ")",
        m_device,
        m_desc.min_filter,
        m_desc.mag_filter,
        m_desc.mip_filter,
        m_desc.reduction_op,
        m_desc.address_u,
        m_desc.address_v,
        m_desc.address_w,
        m_desc.mip_lod_bias,
        m_desc.max_anisotropy,
        m_desc.comparison_func,
        m_desc.border_color,
        m_desc.min_lod,
        m_desc.max_lod,
        m_desc.label
    );
}

} // namespace sgl
