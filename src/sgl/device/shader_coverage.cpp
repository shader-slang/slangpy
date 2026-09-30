// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shader_coverage.h"
#include "sgl/device/device.h"
#include "sgl/device/helpers.h"
#include "sgl/core/crypto.h"
#include <random>

namespace sgl {

ShaderCoverageCollector::ShaderCoverageCollector(ref<Device> device)
    : DeviceChild(std::move(device))
{
}

ShaderCoverageCapabilities ShaderCoverageCollector::capabilities() const
{
    ShaderCoverageCapabilities result;
    if (m_device->is_closed()) {
        result.reason = "Device is closed";
    } else if (m_device->type() != DeviceType::vulkan && m_device->type() != DeviceType::cuda) {
        result.reason = "Shader coverage currently supports Vulkan and CUDA compute programs only";
    } else {
        result.supported = true;
        result.counter_widths.push_back(32);
        result.boolean_counter_widths.push_back(32);
        if (m_device->has_feature(Feature::int64))
            result.boolean_counter_widths.push_back(64);
        if (m_device->has_feature(Feature::atomic_int64))
            result.counter_widths.push_back(64);
    }
    return result;
}

ShaderCoverageSnapshot ShaderCoverageCollector::snapshot(bool reset)
{
    return m_device->_capture_shader_coverage(true, reset);
}

void ShaderCoverageCollector::reset()
{
    m_device->_capture_shader_coverage(false, true);
}

ref<ShaderCoverageCollector> Device::shader_coverage()
{
    SGL_CHECK(!m_closed, "Device is closed");
    return make_ref<ShaderCoverageCollector>(ref<Device>(this));
}

ShaderCoverageState& Device::_shader_coverage_state()
{
    // Caller holds m_coverage_mutex. State is allocated only on explicit use or registration.
    if (!m_coverage_state) {
        auto state = std::make_unique<ShaderCoverageState>();
        std::random_device random;
        SHA1 hash;
        for (int i = 0; i < 8; ++i)
            hash.update(random());
        state->collection_id = hash.hex_digest();
        m_coverage_state = std::move(state);
    }
    return *m_coverage_state;
}

void Device::_register_shader_coverage(SlangSessionBuild& build)
{
    std::lock_guard lock(m_coverage_mutex);
    SGL_CHECK(!m_closed, "Device is closed");
    bool has_new_programs = false;
    for (const auto& [program, data] : build.programs)
        has_new_programs |= data->coverage_counter_width && !data->coverage_generation_id;
    if (!has_new_programs)
        return;
    auto& state = _shader_coverage_state();
    // Keep old generations for pending commands. Never silently evict counts.
    // Stage all records before committing so a failed reload cannot partly register its programs.
    constexpr size_t max_bytes = 256 * 1024 * 1024;
    size_t pending_bytes = 0;
    std::vector<ShaderCoverageState::Program> pending;
    std::vector<ShaderProgramData*> generations;
    for (const auto& [program, data] : build.programs) {
        if (!data->coverage_counter_width || data->coverage_generation_id)
            continue;
        auto buffer = data->coverage_buffer ? data->coverage_buffer->rhi_buffer() : nullptr;
        const size_t bytes = (buffer ? buffer->getDesc().size : 0) + data->coverage_manifest.size();
        SGL_CHECK(
            state.programs.size() + pending.size() < 4096 && bytes <= max_bytes
                && state.retained_bytes + pending_bytes <= max_bytes - bytes,
            "Shader coverage retention limit reached (4096 generations or 256 MiB); capture and use a new device"
        );
        pending.push_back(
            {state.next_generation_id + pending.size(),
             Slang::ComPtr<rhi::IBuffer>(buffer),
             program->desc().label,
             data->coverage_manifest,
             data->coverage_counter_width}
        );
        generations.push_back(data.get());
        pending_bytes += bytes;
    }
    state.programs.reserve(state.programs.size() + pending.size());
    for (auto& program : pending)
        state.programs.push_back(std::move(program));
    for (size_t i = 0; i < generations.size(); ++i)
        generations[i]->coverage_generation_id = state.next_generation_id + i;
    state.next_generation_id += generations.size();
    state.retained_bytes += pending_bytes;
}

ShaderCoverageSnapshot Device::_capture_shader_coverage(bool read, bool reset)
{
    // Serialize collectors and close, but release the registry/submission lock before waiting on the GPU.
    std::lock_guard capture_lock(m_coverage_capture_mutex);
    std::unique_lock lock(m_coverage_mutex);
    SGL_CHECK(!m_closed, "Device is closed");
    auto& state = _shader_coverage_state();
    ShaderCoverageSnapshot result;
    result.collection_id = state.collection_id;
    result.capture_id = state.next_capture_id++;
    result.interval_id = state.interval_id;
    result.reset_after = reset;

    Slang::ComPtr<rhi::ICommandEncoder> encoder;
    Slang::ComPtr<rhi::ICommandBuffer> command_buffer;
    Slang::ComPtr<rhi::IFence> fence;
    std::vector<Slang::ComPtr<rhi::IBuffer>> staging;
    staging.reserve(state.programs.size());
    result.programs.reserve(state.programs.size());
    if (!state.programs.empty()) {
        SLANG_RHI_CALL(m_rhi_graphics_queue->createCommandEncoder(encoder.writeRef()), this);
        SLANG_RHI_CALL(m_rhi_device->createFence({}, fence.writeRef()), this);
    }
    for (const auto& program : state.programs) {
        Slang::ComPtr<rhi::IBuffer> readback;
        const uint64_t size = program.buffer ? program.buffer->getDesc().size : 0;
        if (read) {
            ShaderCoverageProgramSnapshot snapshot;
            snapshot.generation_id = program.generation_id;
            snapshot.label = program.label;
            snapshot.manifest = program.manifest;
            snapshot.counter_width = program.counter_width;
            snapshot.counters.resize(size / (program.counter_width / 8));
            result.programs.push_back(std::move(snapshot));
            if (size) {
                rhi::BufferDesc desc;
                desc.size = size;
                desc.memoryType = rhi::MemoryType::ReadBack;
                desc.usage = rhi::BufferUsage::CopyDestination;
                SLANG_RHI_CALL(m_rhi_device->createBuffer(desc, nullptr, readback.writeRef()), this);
                encoder->copyBuffer(readback, 0, program.buffer, 0, size);
            }
            staging.push_back(std::move(readback));
        }
        if (reset && size)
            encoder->clearBuffer(program.buffer);
    }
    if (encoder) {
        SLANG_RHI_CALL(encoder->finish(command_buffer.writeRef()), this);
        rhi::ICommandBuffer* buffers[] = {command_buffer};
        rhi::IFence* fences[] = {fence};
        const uint64_t value = 1;
        rhi::SubmitDesc desc;
        desc.commandBuffers = buffers;
        desc.commandBufferCount = 1;
        desc.signalFences = fences;
        desc.signalFenceValues = &value;
        desc.signalFenceCount = 1;
        // Direct internal submission avoids hot-reload and user callbacks during collection.
        SLANG_RHI_CALL(m_rhi_graphics_queue->submit(desc), this);
    }
    if (reset)
        ++state.interval_id;
    lock.unlock();
    if (fence) {
        rhi::IFence* fences[] = {fence};
        const uint64_t value = 1;
        SLANG_RHI_CALL(m_rhi_device->waitForFences(1, fences, &value, true, rhi::kTimeoutInfinite), this);
    }
    for (size_t i = 0; i < staging.size(); ++i) {
        if (!staging[i])
            continue;
        void* mapped = nullptr;
        SLANG_RHI_CALL(m_rhi_device->mapBuffer(staging[i], rhi::CpuAccessMode::Read, &mapped), this);
        auto& program = result.programs[i];
        const size_t stride = program.counter_width / 8;
        // Decode without narrowing 64-bit counts or relying on mapped pointer alignment.
        for (size_t j = 0; j < program.counters.size(); ++j) {
            uint64_t value = 0;
            const auto* bytes = static_cast<const uint8_t*>(mapped) + j * stride;
            for (size_t k = 0; k < stride; ++k)
                value |= uint64_t(bytes[k]) << (8 * k);
            program.counters[j] = value;
        }
        SLANG_RHI_CALL(m_rhi_device->unmapBuffer(staging[i]), this);
    }
    return result;
}

} // namespace sgl
