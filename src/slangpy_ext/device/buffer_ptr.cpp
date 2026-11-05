// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"
#include "sgl/device/buffer_ptr.h"

using namespace nb::literals;

template<typename BufferPtr>
NB_NOINLINE static void bind_buffer_ptr(nb::module_& m, const char* name)
{
    nb::class_<BufferPtr>(m, name, D(BufferPtr))
        .def(nb::init<>())
        .def_prop_ro("data", &BufferPtr::data, D(BufferPtr, data))
        .def_prop_ro("is_valid", &BufferPtr::is_valid, D(BufferPtr, is_valid));
}

SGL_PY_EXPORT(device_buffer_ptr)
{
    using namespace sgl;
    bind_buffer_ptr<BufferPtr<>>(m, "BufferPtr");
    bind_buffer_ptr<RWBufferPtr>(m, "RWBufferPtr");
    m.def(
        "to_buffer_ptr",
        &to_buffer_ptr,
        "buffer"_a,
        "byte_offset"_a = 0,
        "alignment"_a = 4,
        D(BufferPtr, from_buffer)
    );
    m.def(
        "to_rw_buffer_ptr",
        &to_rw_buffer_ptr,
        "buffer"_a,
        "byte_offset"_a = 0,
        "alignment"_a = 4,
        D(BufferPtr, from_buffer)
    );
}
