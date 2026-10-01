// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shader_coverage.h"
#include "sgl/device/shader_coverage_internal.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"
#include "sgl/device/cuda_utils.h"
#include "sgl/core/crypto.h"
#include <random>
#include <slang-rhi/synthetic-bindings.h>

namespace sgl {

void SlangCompilerOptions::validate_coverage(Device* device) const
{
    if (coverage) {
        const auto& options = *coverage;
        SGL_CHECK(options.lines || options.functions || options.branches, "At least one coverage mode must be enabled");
        SGL_CHECK(
            options.counter_width == 32 || options.counter_width == 64,
            "Coverage counter width must be 32 or 64 bits"
        );
        SGL_CHECK(
            device->type() == DeviceType::vulkan || device->type() == DeviceType::cuda,
            "Experimental shader coverage supports Vulkan and CUDA only"
        );
        SGL_CHECK(
            options.counter_width != 64 || options.boolean || device->has_feature(Feature::atomic_int64),
            "64-bit shader coverage requires atomic_int64; explicitly select coverage.counter_width=32 on this device"
        );
        SGL_CHECK(
            options.counter_width != 64 || !options.boolean || device->has_feature(Feature::int64),
            "64-bit boolean shader coverage requires int64; explicitly select coverage.counter_width=32 on this device"
        );
    }
}

SlangResult ShaderCoverageProgramData::create_program(
    Device* device,
    const ShaderCoverageOptions& options,
    const rhi::ShaderProgramDesc& desc,
    ShaderProgramData& data,
    ISlangBlob** out_diagnostics
)
{
    auto* linked_program = desc.slangGlobalScope;
    data.coverage = std::make_unique<ShaderCoverageProgramData>();
    Slang::ComPtr<slang::IMetadata> coverage_owner;
    std::vector<rhi::SyntheticResourceBindingDesc> synthetic_resources;
    rhi::ShaderProgramSyntheticResourcesDesc synthetic_desc;
    SGL_CHECK(
        linked_program->getLayout()->getEntryPointCount() == 1
            && linked_program->getLayout()->getEntryPointByIndex(0)->getStage() == SLANG_STAGE_COMPUTE,
        "Experimental shader coverage requires a single compute entry point"
    );
    Slang::ComPtr<ISlangBlob> diagnostics;
    SLANG_CALL(linked_program->getEntryPointMetadata(0, 0, coverage_owner.writeRef(), diagnostics.writeRef()));
    if (diagnostics)
        log_warn("Slang compiler warnings:\n{}", static_cast<const char*>(diagnostics->getBufferPointer()));
    auto coverage = static_cast<slang::ICoverageTracingMetadata*>(
        coverage_owner->castAs(slang::ICoverageTracingMetadata::getTypeGuid())
    );
    auto synthetic = static_cast<slang::ISyntheticResourceMetadata*>(
        coverage_owner->castAs(slang::ISyntheticResourceMetadata::getTypeGuid())
    );
    SGL_CHECK(coverage, "Compiler did not provide shader coverage metadata");
    data.coverage->counter_width = options.counter_width;
    Slang::ComPtr<ISlangBlob> manifest;
    SLANG_CALL(slang_writeCoverageManifestJson(coverage, manifest.writeRef()));
    data.coverage->manifest.assign(static_cast<const char*>(manifest->getBufferPointer()), manifest->getBufferSize());
    if (coverage->getCounterCount()) {
        SGL_CHECK(synthetic && synthetic->getResourceCount() == 1, "Expected one coverage resource");
        slang::SyntheticResourceInfo info;
        SLANG_CALL(synthetic->getResourceInfo(0, &info));
        SGL_CHECK(
            info.scope == slang::SyntheticResourceScope::Global && info.arraySize == 1
                && info.access == slang::SyntheticResourceAccess::ReadWrite
                && info.bindingType == slang::BindingType::MutableRawBuffer,
            "Unsupported coverage resource layout"
        );
        rhi::SyntheticResourceBindingDesc binding;
        binding.id = info.id;
        binding.bindingType = info.bindingType;
        binding.scope = rhi::SyntheticResourceScope::Global;
        binding.access = rhi::SyntheticResourceAccess::ReadWrite;
        binding.space = info.space;
        binding.binding = info.binding;
        binding.uniformOffset = info.uniformOffset;
        binding.uniformStride = info.uniformStride;
        binding.debugName = info.debugName;
        synthetic_resources.push_back(binding);
        synthetic_desc.resources = synthetic_resources.data();
        synthetic_desc.resourceCount = uint32_t(synthetic_resources.size());

        slang::CoverageBufferInfo buffer_info;
        SLANG_CALL(coverage->getBufferInfo(&buffer_info));
        SGL_CHECK(
            buffer_info.elementByteWidth * 8 == options.counter_width,
            "Compiler changed the requested coverage counter width"
        );
        std::vector<uint8_t> zeroes(size_t(coverage->getCounterCount()) * buffer_info.elementByteWidth, 0);
        data.coverage->buffer = device->create_buffer(
            BufferDesc{
                .size = zeroes.size(),
                .struct_size = buffer_info.elementByteWidth,
                .usage = BufferUsage::unordered_access | BufferUsage::shader_resource | BufferUsage::copy_source
                    | BufferUsage::copy_destination,
                .label = "shader-coverage",
                .data = zeroes.data(),
                .data_size = zeroes.size(),
            }
        );
        data.coverage->resource_id = info.id;
    }
    auto rhi_desc = desc;
    synthetic_desc.next = desc.next;
    if (!synthetic_resources.empty())
        rhi_desc.next = &synthetic_desc;
    return device->rhi_device()->createShaderProgram(rhi_desc, data.rhi_shader_program.writeRef(), out_diagnostics);
}

ref<Buffer> ShaderProgram::coverage_buffer() const
{
    return m_data->coverage ? m_data->coverage->buffer : nullptr;
}

const std::string& ShaderProgram::coverage_manifest() const
{
    static const std::string empty;
    return m_data->coverage ? m_data->coverage->manifest : empty;
}

void ShaderCoverageProgramData::bind(rhi::IShaderProgram* program, rhi::IShaderObject* root_object) const
{
    if (buffer)
        SLANG_CALL(rhi::bindSyntheticResource(program, root_object, resource_id, rhi::Binding(buffer->rhi_buffer())));
}


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
    return _begin_capture(true, reset).finish();
}

void ShaderCoverageCollector::reset()
{
    _begin_capture(false, true).finish();
}

ShaderCoverageCapture ShaderCoverageCollector::_begin_capture(bool read, bool reset)
{
    return m_device->_begin_shader_coverage_capture(read, reset);
}

ref<ShaderCoverageCollector> Device::shader_coverage()
{
    SGL_CHECK(!m_closed, "Device is closed");
    return make_ref<ShaderCoverageCollector>(ref<Device>(this));
}

ShaderCoverageStateOwner::ShaderCoverageStateOwner() = default;
ShaderCoverageStateOwner::~ShaderCoverageStateOwner() = default;

ShaderCoverageState& ShaderCoverageStateOwner::get_or_create()
{
    if (!m_state) {
        auto candidate = std::make_unique<ShaderCoverageState>();
        std::random_device random;
        SHA1 hash;
        for (int i = 0; i < 8; ++i)
            hash.update(random());
        candidate->collection_id = hash.hex_digest();
        m_state = std::move(candidate);
    }
    return *m_state;
}

void Device::_register_shader_coverage(SlangSessionBuild& build)
{
    SGL_CHECK(!m_closed, "Device is closed");
    auto& state = m_coverage_state.get_or_create();
    bool has_new_programs = false;
    for (const auto& [program, data] : build.programs)
        has_new_programs |= data->coverage && !data->coverage->generation_id;
    if (!has_new_programs)
        return;
    // Keep old generations for pending commands. Never silently evict counts.
    // Stage all records before committing so a failed reload cannot partly register its programs.
    constexpr size_t max_bytes = 256 * 1024 * 1024;
    size_t pending_bytes = 0;
    std::vector<ShaderCoverageState::Program> pending;
    std::vector<ShaderCoverageProgramData*> generations;
    for (const auto& [program, data] : build.programs) {
        if (!data->coverage || data->coverage->generation_id)
            continue;
        auto buffer = data->coverage->buffer ? data->coverage->buffer->rhi_buffer() : nullptr;
        const size_t bytes = (buffer ? buffer->getDesc().size : 0) + data->coverage->manifest.size();
        SGL_CHECK(
            state.programs.size() + pending.size() < 4096 && bytes <= max_bytes
                && state.retained_bytes + pending_bytes <= max_bytes - bytes,
            "Shader coverage retention limit reached (4096 generations or 256 MiB); capture and use a new device"
        );
        pending.push_back(
            {state.next_generation_id + pending.size(),
             Slang::ComPtr<rhi::IBuffer>(buffer),
             program->desc().label,
             data->coverage->manifest,
             data->coverage->counter_width}
        );
        generations.push_back(data->coverage.get());
        pending_bytes += bytes;
    }
    state.programs.reserve(state.programs.size() + pending.size());
    for (auto& program : pending)
        state.programs.push_back(std::move(program));
    for (size_t i = 0; i < generations.size(); ++i)
        generations[i]->generation_id = state.next_generation_id + i;
    state.next_generation_id += generations.size();
    state.retained_bytes += pending_bytes;
}

ShaderCoverageCapture Device::_begin_shader_coverage_capture(bool read, bool reset)
{
    SGL_CHECK(!m_closed, "Device is closed");
    // Collection may be invoked from a different Python thread than device creation.
    // CUDA queue allocation and submission require its context on the calling thread.
    std::optional<cuda::ContextScope> cuda_scope;
    if (type() == DeviceType::cuda)
        cuda_scope.emplace(this);
    auto& state = m_coverage_state.get_or_create();
    ShaderCoverageCapture capture;
    capture.owner = ref<Device>(this);
    capture.device = m_rhi_device;
    auto& result = capture.result;
    result.collection_id = state.collection_id;
    result.capture_id = state.next_capture_id;
    result.interval_id = state.interval_id;
    result.reset_after = reset;

    Slang::ComPtr<rhi::ICommandEncoder> encoder;
    auto& command_buffer = capture.command_buffer;
    auto& fence = capture.fence;
    auto& staging = capture.staging;
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
    ++state.next_capture_id;
    return capture;
}

ShaderCoverageSnapshot ShaderCoverageCapture::finish()
{
    // Only the lifetime-stable diagnostic logger is accessed through owner.
    // Another capture, submission, or close can proceed while these owned buffers are read.
    if (fence) {
        rhi::IFence* fences[] = {fence};
        const uint64_t value = 1;
        SLANG_RHI_CALL(device->waitForFences(1, fences, &value, true, rhi::kTimeoutInfinite), owner.get());
    }
    for (size_t i = 0; i < staging.size(); ++i) {
        if (!staging[i])
            continue;
        void* mapped = nullptr;
        SLANG_RHI_CALL(device->mapBuffer(staging[i], rhi::CpuAccessMode::Read, &mapped), owner.get());
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
        SLANG_RHI_CALL(device->unmapBuffer(staging[i]), owner.get());
    }
    return std::move(result);
}

} // namespace sgl
