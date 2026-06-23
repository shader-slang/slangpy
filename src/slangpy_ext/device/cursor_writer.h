// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <optional>

#include "nanobind.h"

#include "sgl/device/cursor_utils.h"

namespace sgl::slangpy {

struct NativeCursorWriterValue {
    const cursor_utils::CursorWriterTypeInfo* info;
    void* value;
};

const cursor_utils::CursorWriterTypeInfo* find_native_cursor_writer_type_info(nb::handle obj);
std::optional<NativeCursorWriterValue> find_native_cursor_writer(nb::handle obj);
nb::object get_native_cursor_writer_type_info(nb::handle obj);

} // namespace sgl::slangpy
