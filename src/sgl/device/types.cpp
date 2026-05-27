// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "types.h"

#include "sgl/device/buffer_cursor.h"
#include "sgl/device/shader_cursor.h"

#include "sgl/core/format.h"

namespace sgl {

void DescriptorHandle::write_to_cursor(const ShaderCursor& cursor) const
{
    cursor.set_descriptor_handle(*this);
}

void DescriptorHandle::write_to_cursor(const BufferElementCursor& cursor) const
{
    cursor.set_data(&value, sizeof(value));
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
