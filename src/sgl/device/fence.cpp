// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fence.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"

#include "sgl/core/config.h"
#include "sgl/core/macros.h"
#include "sgl/core/string.h"

namespace sgl {

Fence::Fence(ref<Device> device, FenceDesc desc)
    : DeviceChild(std::move(device))
    , m_desc(std::move(desc))
{
    SGL_ASSERT(m_device);

    rhi::FenceDesc rhi_desc{
        .initialValue = m_desc.initial_value,
        .isShared = m_desc.shared,
    };

    SLANG_RHI_CALL(m_device->rhi_device()->createFence(rhi_desc, m_rhi_fence.writeRef()));

    m_signaled_value = m_desc.initial_value;
}

uint64_t Fence::signal(uint64_t value)
{
    uint64_t signal_value = update_signaled_value(value);
    SLANG_RHI_CALL(m_rhi_fence->setCurrentValue(signal_value));
    return signal_value;
}

void Fence::wait(uint64_t value, uint64_t timeout_ns)
{
    uint64_t wait_value = value == AUTO ? m_signaled_value : value;
    uint64_t cur_value = current_value();
    if (cur_value < wait_value) {
        rhi::IFence* fences[] = {m_rhi_fence};
        uint64_t wait_values[] = {wait_value};
        SLANG_RHI_CALL(m_device->rhi_device()->waitForFences(1, fences, wait_values, true, timeout_ns));
    }
}

uint64_t Fence::current_value() const
{
    uint64_t value;
    SLANG_RHI_CALL(m_rhi_fence->getCurrentValue(&value));
    return value;
}

uint64_t Fence::update_signaled_value(uint64_t value)
{
    m_signaled_value = value == AUTO ? m_signaled_value + 1 : value;
    return m_signaled_value;
}

NativeHandle Fence::shared_handle() const
{
    rhi::NativeHandle rhi_handle = {};
    m_rhi_fence->getSharedHandle(&rhi_handle);
    return NativeHandle(rhi_handle);
}

NativeHandle Fence::native_handle() const
{
    rhi::NativeHandle rhi_handle = {};
    m_rhi_fence->getNativeHandle(&rhi_handle);
    return NativeHandle(rhi_handle);
}

std::string Fence::to_string() const
{
    return fmt::format(
        "Fence(\n"
        "  device = {},\n"
        "  initial_value = {}\n"
        "  shared = {}\n"
        "  current_value = {},\n"
        "  signaled_value = {}\n"
        ")",
        m_device,
        m_desc.initial_value,
        m_desc.shared,
        current_value(),
        m_signaled_value
    );
}

} // namespace sgl
