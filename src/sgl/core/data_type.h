// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/enum.h"

namespace sgl {

enum class DataType {
    void_,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
};

SGL_ENUM_INFO(
    DataType,
    {
        {DataType::void_, "void"},
        {DataType::bool_, "bool"},
        {DataType::int8, "int8"},
        {DataType::int16, "int16"},
        {DataType::int32, "int32"},
        {DataType::int64, "int64"},
        {DataType::uint8, "uint8"},
        {DataType::uint16, "uint16"},
        {DataType::uint32, "uint32"},
        {DataType::uint64, "uint64"},
        {DataType::float16, "float16"},
        {DataType::float32, "float32"},
        {DataType::float64, "float64"},
    }
);
SGL_ENUM_REGISTER(DataType);

/// Get the size of a type in bytes.
inline size_t data_type_size(sgl::DataType type)
{
    using sgl::DataType;
    switch (type) {
    case DataType::int8:
    case DataType::uint8:
        return 1;
    case DataType::int16:
    case DataType::uint16:
    case DataType::float16:
        return 2;
    case DataType::int32:
    case DataType::uint32:
    case DataType::float32:
        return 4;
    case DataType::int64:
    case DataType::uint64:
    case DataType::float64:
        return 8;
    case DataType::void_:
    case DataType::bool_:
        break; // throws
    };
    SGL_THROW("Invalid type.");
}

} // namespace sgl
