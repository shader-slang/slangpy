// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/device/buffer_ptr.h"
#include "testing.h"
#include "sgl/device/device.h"

#include <sgl/device/kernel.h>
#include <sgl/device/command.h>

using namespace sgl;

TEST_CASE("BufferPtr encoding and null")
{
    using sgl::DeviceType;
    CHECK_FALSE(BufferPtr<>{}.is_valid());
    CHECK_THROWS(BufferPtr<>{}.write_to_cursor(ShaderCursor{}));
    CHECK_THROWS(RWBufferPtr{}.write_to_cursor(BufferElementCursor{}));
    CHECK_FALSE(RWBufferPtr{}.is_valid());
    CHECK_EQ(BufferPtr<>{}.data(), UINT64_MAX);
    CHECK_EQ(detail::encode_buffer_ptr(DeviceType::d3d12, 0, 0), 0);
    CHECK_EQ(detail::encode_buffer_ptr(DeviceType::d3d12, 17, UINT32_MAX), (uint64_t(UINT32_MAX) << 32) | 17);
    CHECK_THROWS(detail::encode_buffer_ptr(DeviceType::d3d12, UINT32_MAX, 0));
    CHECK_THROWS(detail::encode_buffer_ptr(DeviceType::d3d12, 17, uint64_t(UINT32_MAX) + 1));
    CHECK_EQ(detail::encode_buffer_ptr(DeviceType::vulkan, 0x1000, 20), 0x1014);
    CHECK_EQ(detail::encode_buffer_ptr(DeviceType::cuda, 0x1000, 20), 0x1014);
    CHECK_THROWS(detail::encode_buffer_ptr(DeviceType::cuda, UINT64_MAX - 3, 3));
    CHECK_THROWS(detail::encode_buffer_ptr(DeviceType::vulkan, UINT64_MAX - 3, 4));
    CHECK_EQ(detail::encode_buffer_ptr(DeviceType::metal, 0x1000, 20), 0x1014);
    CHECK_THROWS(detail::encode_buffer_ptr(DeviceType::cpu, 0, 0));
    CHECK_THROWS(detail::encode_buffer_ptr(DeviceType::wgpu, 0, 0));
    CHECK_FALSE(to_buffer_ptr({}).is_valid());
    CHECK_THROWS(to_buffer_ptr({}, 4));
    CHECK_THROWS(to_buffer_ptr({}, 0, 3));
}

TEST_CASE_GPU("BufferPtr host construction")
{
    auto buffer = ctx.device->create_buffer(
        {.size = 64, .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access}
    );
    const auto read = to_buffer_ptr(buffer, 16, 16);
    const auto write = to_rw_buffer_ptr(buffer, 16, 16);
    CHECK(read.is_valid());
    CHECK(write.is_valid());
    CHECK_THROWS(to_buffer_ptr(buffer, 2));
    CHECK_THROWS(to_buffer_ptr(buffer, 4, 16));
    CHECK_THROWS(to_buffer_ptr(buffer, 68));
    CHECK(to_buffer_ptr(buffer, 64).is_valid());
    if (ctx.device->type() == sgl::DeviceType::d3d12) {
        CHECK_EQ(uint32_t(read.data() >> 32), 16);
        CHECK_EQ(uint32_t(write.data() >> 32), 16);
        CHECK_EQ(uint32_t(read.data()), buffer->descriptor_handle_ro().value);
        CHECK_EQ(uint32_t(write.data()), buffer->descriptor_handle_rw().value);
    } else {
        CHECK_EQ(read.data(), buffer->device_address() + 16);
        CHECK_EQ(write.data(), read.data());
    }
}

TEST_CASE_GPU("BufferPtr native cursor roundtrip")
{
    auto module = ctx.device->load_module_from_source("buffer_ptr_native_cursor", R"(
import bufferptr;
struct Record { BufferPtr<> ro; RWBufferPtr<> rw; }
[shader("compute")][numthreads(1,1,1)]
void main(BufferPtr<> src, RWBufferPtr<> dst,
          StructuredBuffer<Record> records)
{
    dst.as<uint>()[0] = src.as<uint>()[1];
    records[0].rw.as<uint>()[1] = records[0].ro.as<uint>()[0];
    dst.as<uint>()[2] = records[1].ro.is_valid();
    dst.as<uint>()[3] = records[1].rw.is_valid();
}
)");
    const std::array<uint32_t, 4> values{10, 20, 30, 40};
    auto src = ctx.device->create_buffer(
        {.size = sizeof(values),
         .usage = sgl::BufferUsage::shader_resource,
         .data = values.data(),
         .data_size = sizeof(values)}
    );
    auto dst = ctx.device->create_buffer({.size = 16, .usage = sgl::BufferUsage::unordered_access});
    auto layout = module->layout()
                      ->get_type_layout(module->layout()->find_type_by_name("StructuredBuffer<Record>"))
                      ->element_type_layout();
    CHECK_EQ(layout->size(), 16);
    CHECK_EQ(layout->stride(), 16);
    auto records = ctx.device->create_buffer({.size = 32, .usage = sgl::BufferUsage::shader_resource});
    auto cursor = sgl::make_ref<sgl::BufferCursor>(layout, records);
    (*cursor)[0]["ro"] = to_buffer_ptr(src, 8);
    (*cursor)[0]["rw"] = to_rw_buffer_ptr(dst);
    (*cursor)[1]["ro"] = BufferPtr<>{};
    (*cursor)[1]["rw"] = RWBufferPtr{};
    cursor->apply();
    auto kernel = ctx.device->create_compute_kernel(
        {.program = ctx.device->link_program({module}, {module->entry_point("main")})}
    );
    auto encoder = ctx.device->create_command_encoder();
    encoder->set_buffer_state(src, sgl::ResourceState::shader_resource);
    encoder->set_buffer_state(dst, sgl::ResourceState::unordered_access);
    kernel->dispatch(
        {1, 1, 1},
        [&](sgl::ShaderCursor root)
        {
            auto entry = root.find_entry_point(0);
            entry["src"] = to_buffer_ptr(src, 8);
            entry["dst"] = to_rw_buffer_ptr(dst);
            entry["records"].set_buffer(records);
        },
        encoder
    );
    ctx.device->submit_command_buffer(encoder->finish());
    ctx.device->wait();
    CHECK(dst->get_elements<uint32_t>() == std::vector<uint32_t>{40, 30, 0, 0});
}
