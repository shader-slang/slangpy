// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/shader.h"

using namespace sgl;

TEST_SUITE_BEGIN("device");

namespace {

struct ExecuteCallbackTestState {
    bool called{false};
    NativeHandle callback_handle;
};

void SLANG_MCALL execute_callback_test(
    const CommandNativeCallbackContext* context,
    void* user_object,
    const void* user_data,
    size_t user_data_size
)
{
    SGL_UNUSED(user_object);
    CHECK(user_data_size == sizeof(ExecuteCallbackTestState*));

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
