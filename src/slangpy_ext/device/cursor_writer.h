// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nanobind.h"

#include "sgl/device/cursor_utils.h"

namespace sgl::slangpy {

struct CursorWriterValue {
    const cursor_utils::CursorWriterTypeInfo* info;
    const void* value;
};

using PythonShaderCursorWriteFunc = std::function<bool(ShaderCursor&, nb::handle)>;
using PythonBufferElementCursorWriteFunc = std::function<bool(BufferElementCursor&, nb::handle)>;
using PythonCursorSignatureWriteFunc = std::function<void(SignatureBuffer&, nb::handle)>;

const cursor_utils::CursorWriterTypeInfo* find_cursor_writer_type_info(nb::handle obj);
std::optional<CursorWriterValue> find_cursor_writer(nb::handle obj);
nb::object get_cursor_writer_type_info(nb::handle obj);
void register_python_cursor_writer_type(
    nb::type_object python_type,
    PythonShaderCursorWriteFunc write_shader_cursor,
    PythonBufferElementCursorWriteFunc write_buffer_cursor,
    PythonCursorSignatureWriteFunc write_signature,
    std::string slang_type_name,
    std::vector<std::string> imports
);
void register_python_cursor_writer_type(
    nb::type_object python_type,
    nb::object write_shader_cursor,
    nb::object write_buffer_cursor,
    nb::object write_signature,
    std::string slang_type_name,
    std::vector<std::string> imports
);
void unregister_python_cursor_writer_types();

using NativeCursorWriterValue = CursorWriterValue;

inline const cursor_utils::CursorWriterTypeInfo* find_native_cursor_writer_type_info(nb::handle obj)
{
    return find_cursor_writer_type_info(obj);
}

inline std::optional<NativeCursorWriterValue> find_native_cursor_writer(nb::handle obj)
{
    return find_cursor_writer(obj);
}

inline nb::object get_native_cursor_writer_type_info(nb::handle obj)
{
    return get_cursor_writer_type_info(obj);
}

} // namespace sgl::slangpy
