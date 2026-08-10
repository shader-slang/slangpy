// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/shader.h"

#include <array>
#include <fstream>

using namespace sgl;

TEST_SUITE_BEGIN("device");

namespace {

struct ExecuteCallbackTestState {
    bool called{false};
    NativeHandle callback_handle;
};

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

TEST_SUITE_END();
