// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "types.h"

#include "sgl/device/buffer_cursor.h"
#include "sgl/device/shader_cursor.h"

#include "sgl/core/error.h"
#include "sgl/core/format.h"

namespace sgl {

void DescriptorHandle::write_to_cursor(const ShaderCursor& cursor, const DescriptorHandle* value)
{
    SGL_CHECK(value, "Cannot write a null descriptor handle pointer to a shader cursor.");
    cursor.set_descriptor_handle(*value);
}

void DescriptorHandle::write_to_cursor(const BufferElementCursor& cursor, const DescriptorHandle* value)
{
    SGL_CHECK(value, "Cannot write a null descriptor handle pointer to a buffer cursor.");
    // CUDA lowers StructuredBuffer.Handle and RWStructuredBuffer.Handle to the
    // inline two-word resource representation from slang-cuda-prelude.h:
    // data pointer followed by element count. Other resource handles remain a
    // single opaque word.
    if (cursor._get_device_type() == DeviceType::cuda
        && (value->type == DescriptorHandleType::buffer || value->type == DescriptorHandleType::rw_buffer)) {
        const uint64_t payload[2] = {value->value, value->auxiliary};
        cursor.set_data(payload, sizeof(payload));
    } else {
        cursor.set_data(&value->value, sizeof(value->value));
    }
}

std::string DescriptorHandle::to_string() const
{
    return fmt::format("DescriptorHandle(type={}, value=0x{:08x})", type, value);
}

std::string Viewport::to_string() const
{
    return fmt::format(
        "Viewport(x={}, y={}, width={}, height={}, min_depth={}, max_depth={})",
        x,
        y,
        width,
        height,
        min_depth,
        max_depth
    );
}

std::string ScissorRect::to_string() const
{
    return fmt::format("ScissorRect(min_x={}, min_y={}, max_x={}, max_y={})", min_x, min_y, max_x, max_y);
}

} // namespace sgl
