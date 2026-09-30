// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/shader.h"
#include "sgl/device/cuda_utils.h"
#include "sgl/device/cuda_interop.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"

#include <array>
#include <fstream>
#include <memory>
#include <type_traits>

using namespace sgl;

TEST_SUITE_BEGIN("device");

namespace {

struct ExecuteCallbackTestState {
    bool called{false};
    NativeHandle callback_handle;
};

struct CUDADriverAPIScope {
    bool loaded{rhiCudaDriverApiInit()};
    ~CUDADriverAPIScope()
    {
        if (loaded)
            rhiCudaDriverApiShutdown();
    }
};

bool is_nvidia_graphics_device(Device* device)
{
    if (device->type() != DeviceType::d3d12 && device->type() != DeviceType::vulkan)
        return false;
    for (const auto& adapter : Device::enumerate_adapters(device->type())) {
        if (adapter.luid == device->info().adapter_luid)
            return adapter.vendor_id == 0x10de;
    }
    return false;
}

void SLANG_MCALL execute_callback_test(
    const ExecuteCallbackContext* context,
    void* user_object,
    const void* user_data,
    size_t user_data_size
)
{
    SGL_UNUSED(user_object);
    REQUIRE(user_data_size == sizeof(ExecuteCallbackTestState*));

    auto state = *static_cast<ExecuteCallbackTestState* const*>(user_data);
    state->called = true;
    state->callback_handle = NativeHandle(context->nativeHandle);
}

void check_execute_callback_native_handle(Device* device, NativeHandle callback_handle)
{
    if (device->type() == DeviceType::d3d12) {
        CHECK(callback_handle.type() == NativeHandleType::D3D12GraphicsCommandList);
        CHECK(callback_handle.value() != 0);
    } else if (device->type() == DeviceType::vulkan) {
        CHECK(callback_handle.type() == NativeHandleType::VkCommandBuffer);
        CHECK(callback_handle.value() != 0);
    }
}

} // namespace

TEST_CASE("enumerate_adapters")
{
    if (!testing::device_tests_enabled())
        SKIP("device tests disabled by -skip-device-tests");

    std::vector<AdapterInfo> adapters = Device::enumerate_adapters();
    CHECK(!adapters.empty());
}

TEST_CASE_GPU("init")
{
    CHECK(ctx.device);
}

TEST_CASE_GPU("invalid_shader_cache_is_disabled_without_deleting_cache")
{
    const std::filesystem::path cache_dir
        = testing::get_case_temp_directory() / std::to_string(static_cast<uint32_t>(ctx.device->type()));
    const std::filesystem::path rhi_cache_dir = cache_dir / "rhi";
    const std::filesystem::path data_path = rhi_cache_dir / "data.mdb";
    const std::filesystem::path marker_path = rhi_cache_dir / "marker";
    std::filesystem::create_directories(rhi_cache_dir);

    std::array<uint8_t, 8192> invalid_data;
    invalid_data.fill(0xff);
    {
        std::ofstream data_file(data_path, std::ios::binary);
        REQUIRE(data_file);
        data_file.write(
            reinterpret_cast<const char*>(invalid_data.data()),
            static_cast<std::streamsize>(invalid_data.size())
        );
        REQUIRE(data_file.good());
    }
    {
        std::ofstream marker_file(marker_path);
        REQUIRE(marker_file);
        marker_file << "keep";
        REQUIRE(marker_file.good());
    }

    DeviceDesc desc = ctx.device->desc();
    desc.shader_cache_path = cache_dir;
    ref<Device> device;
    CHECK_NOTHROW(device = Device::create(desc));
    REQUIRE(device);
    CHECK(device->shader_cache_stats().entry_count == 0);
    CHECK(std::filesystem::exists(marker_path));
    CHECK(std::filesystem::file_size(data_path) == invalid_data.size());
    device->close();
}

TEST_CASE_GPU("close_all_devices_keeps_snapshot_alive")
{
    DeviceDesc desc = ctx.device->desc();
    desc.label = "close-all-devices-snapshot-a";
    ref<Device> device_a = Device::create(desc);
    desc.label = "close-all-devices-snapshot-b";
    ref<Device> device_b = Device::create(desc);

    int close_count_a = 0;
    int close_count_b = 0;

    device_a->register_device_close_callback(
        [&](Device*)
        {
            close_count_a++;
        }
    );
    device_b->register_device_close_callback(
        [&](Device*)
        {
            close_count_b++;
            device_a->close();
            device_a = nullptr;
        }
    );

    Device::close_all_devices();
    testing::release_cached_devices();

    CHECK(close_count_a == 1);
    CHECK(close_count_b == 1);
    CHECK_THROWS(current_device());
}

TEST_CASE_GPU("execute_callback_desc_native_handle")
{
    ExecuteCallbackTestState state;
    ExecuteCallbackTestState* state_ptr = &state;

    ref<CommandEncoder> command_encoder = ctx.device->create_command_encoder();
    command_encoder->execute_callback({
        .callback = execute_callback_test,
        .user_data = &state_ptr,
        .user_data_size = sizeof(state_ptr),
    });

    ref<CommandBuffer> command_buffer = command_encoder->finish();

    ctx.device->submit_command_buffer(command_buffer);
    ctx.device->wait();

    CHECK(state.called);
    check_execute_callback_native_handle(ctx.device, state.callback_handle);
}

TEST_CASE_GPU("execute_callback_lambda_native_handle")
{
    ExecuteCallbackTestState state;

    ref<CommandEncoder> command_encoder = ctx.device->create_command_encoder();
    command_encoder->execute_callback(
        [&](NativeHandle native_handle)
        {
            state.called = true;
            state.callback_handle = native_handle;
        }
    );

    ref<CommandBuffer> command_buffer = command_encoder->finish();

    ctx.device->submit_command_buffer(command_buffer);
    ctx.device->wait();

    CHECK(state.called);
    check_execute_callback_native_handle(ctx.device, state.callback_handle);
}

TEST_CASE_GPU("cuda_close_mapped_buffer_outlives_device")
{
    if (!is_nvidia_graphics_device(ctx.device))
        SKIP("CUDA interop requires an NVIDIA D3D12 or Vulkan adapter");
    CUDADriverAPIScope api;
    if (!api.loaded)
        SKIP("CUDA driver API is unavailable");

    for (bool release_rhi : {false, true}) {
        CAPTURE(release_rhi);
        auto desc = ctx.device->desc();
        desc.adapter_luid = ctx.device->info().adapter_luid;
        desc.enable_cuda_interop = true;
        const size_t device_count = Device::get_created_devices().size();
        auto device = Device::create(desc);
        auto buffer = device->create_buffer({.size = 16, .usage = BufferUsage::unordered_access | BufferUsage::shared});
        void* cuda_memory;
        {
            SGL_CU_SCOPE(device.get());
            cuda_memory = buffer->cuda_memory();
            REQUIRE(cuda_memory != nullptr);
        }
        CUcontext other_context;
        SGL_CU_CHECK(cuCtxCreate(&other_context, 0, device->cuda_device()->device()));
        std::unique_ptr<std::remove_pointer_t<CUcontext>, decltype(cuCtxDestroy)> context_owner(
            other_context,
            cuCtxDestroy
        );
        device->close();
        CHECK(buffer->cuda_memory() == cuda_memory);
        if (release_rhi) {
            device->_release_rhi_resources();
            CHECK(buffer->rhi_buffer() == nullptr);
            CHECK_THROWS_WITH(
                buffer->cuda_memory(),
                doctest::Contains("Cannot import CUDA memory on a closed device.")
            );
        }
        device.reset();
        CHECK(Device::get_created_devices().size() == device_count + 1);
        CHECK(buffer->device()->cuda_device() != nullptr);
        buffer.reset();
        CHECK(Device::get_created_devices().size() == device_count);
        CUcontext current_context;
        SGL_CU_CHECK(cuCtxGetCurrent(&current_context));
        CHECK(current_context == other_context);
    }
}

TEST_CASE_GPU("cuda_close_interop_buffer_outlives_device")
{
    if (!is_nvidia_graphics_device(ctx.device))
        SKIP("CUDA interop requires an NVIDIA D3D12 or Vulkan adapter");
    CUDADriverAPIScope api;
    if (!api.loaded)
        SKIP("CUDA driver API is unavailable");

    auto desc = ctx.device->desc();
    desc.adapter_luid = ctx.device->info().adapter_luid;
    desc.enable_cuda_interop = true;
    const size_t device_count = Device::get_created_devices().size();
    auto device = Device::create(desc);
    auto module = device->load_module_from_source(
        "cuda_close_interop_owner",
        R"(
[shader("compute")]
[numthreads(1, 1, 1)]
void main(RWStructuredBuffer<uint> buffer) { buffer[0] = 1; }
)"
    );
    auto program = device->link_program({module}, {module->entry_point("main")});
    auto root = device->create_root_shader_object(program);
    auto entry_point = root->get_entry_point(0);
    // Construct the internal owner through its binding path; no CUDA copy is submitted.
    ShaderCursor(entry_point)["buffer"].set_cuda_tensor_view(
        {.device_id = device->cuda_device()->device(), .data = nullptr, .size = 16, .stride = 4}
    );
    std::vector<ref<cuda::InteropBuffer>> interop;
    entry_point->get_cuda_interop_buffers(interop);
    REQUIRE(interop.size() == 1);
    {
        SGL_CU_SCOPE(device.get());
        REQUIRE(interop.front()->buffer()->cuda_memory() != nullptr);
    }
    entry_point.reset();
    root.reset();
    program.reset();
    module.reset();
    device->close();
    device->_release_rhi_resources();
    CHECK(interop.front()->buffer()->rhi_buffer() == nullptr);
    device.reset();
    CHECK(Device::get_created_devices().size() == device_count + 1);
    interop.clear();
    CHECK(Device::get_created_devices().size() == device_count);
}

TEST_SUITE_END();
