// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "nanobind.h"
#include "sgl/core/error.h"

namespace sgl::math {

template<class T>
struct PrimitiveType {
    static PyObject* python_type() { SGL_THROW("Unknown primitive type"); }
};
template<>
struct PrimitiveType<float16_t> {
    static PyObject* python_type() { return (PyObject*)&PyFloat_Type; }
};
template<>
struct PrimitiveType<float> {
    static PyObject* python_type() { return (PyObject*)&PyFloat_Type; }
};
template<>
struct PrimitiveType<int> {
    static PyObject* python_type() { return (PyObject*)&PyLong_Type; }
};
template<>
struct PrimitiveType<uint32_t> {
    static PyObject* python_type() { return (PyObject*)&PyLong_Type; }
};
template<>
struct PrimitiveType<bool> {
    static PyObject* python_type() { return (PyObject*)&PyBool_Type; }
};

} // namespace sgl::math
