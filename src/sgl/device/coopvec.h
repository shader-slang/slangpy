// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"

#include "sgl/core/fwd.h"
#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/core/data_type.h"

#include <vector>
#include <slang-rhi.h>

namespace sgl {


inline rhi::CooperativeVectorComponentType get_rhi_component_type(DataType dtype)
{
    switch (dtype) {
    case DataType::float16:
        return rhi::CooperativeVectorComponentType::Float16;
    case DataType::float32:
        return rhi::CooperativeVectorComponentType::Float32;
    case DataType::float64:
        return rhi::CooperativeVectorComponentType::Float64;
    case DataType::int8:
        return rhi::CooperativeVectorComponentType::Sint8;
    case DataType::int16:
        return rhi::CooperativeVectorComponentType::Sint16;
    case DataType::int32:
        return rhi::CooperativeVectorComponentType::Sint32;
    case DataType::uint8:
        return rhi::CooperativeVectorComponentType::Uint8;
    case DataType::uint16:
        return rhi::CooperativeVectorComponentType::Uint16;
    case DataType::uint32:
        return rhi::CooperativeVectorComponentType::Uint32;
    case DataType::uint64:
        return rhi::CooperativeVectorComponentType::Uint64;
    default:
        SGL_THROW("\"%s\" is not a valid component type for cooperative vector matrix", dtype);
    }
}

inline uint32_t calc_element_stride(uint32_t rows, uint32_t cols, CoopVecMatrixLayout layout)
{
    if (layout == CoopVecMatrixLayout::row_major)
        return cols;
    else if (layout == CoopVecMatrixLayout::column_major)
        return rows;
    return 0ull;
};

inline size_t get_element_size(DataType dtype)
{
    switch (dtype) {
    case DataType::int8:
    case DataType::uint8:
        return 1;
    case DataType::float16:
    case DataType::int16:
    case DataType::uint16:
        return 2;
    case DataType::float32:
    case DataType::int32:
    case DataType::uint32:
        return 4;
    case DataType::float64:
    case DataType::uint64:
        return 8;
    default:
        SGL_THROW("\"%s\" is not a valid component type for cooperative vector matrix", dtype);
    }
}

inline rhi::CooperativeVectorMatrixDesc get_rhi_desc(CoopVecMatrixDesc desc)
{
    rhi::CooperativeVectorMatrixDesc rhi_desc = {};
    rhi_desc.rowCount = desc.rows;
    rhi_desc.colCount = desc.cols;
    rhi_desc.componentType = get_rhi_component_type(desc.element_type);
    rhi_desc.layout = static_cast<rhi::CooperativeVectorMatrixLayout>(desc.layout);
    rhi_desc.size = desc.size;
    rhi_desc.offset = desc.offset;
    rhi_desc.rowColumnStride = desc.row_col_stride;
    return rhi_desc;
}


class SGL_API CoopVec : public Object {
    SGL_OBJECT(CoopVec)
public:
    CoopVec(Device* device);

    static constexpr size_t MATRIX_ALIGNMENT = 64; ///< Minimum byte alignment according to spec.
    static constexpr size_t VECTOR_ALIGHMENT = 16; ///< Minimum byte alignment according to spec.

    size_t query_matrix_size(uint32_t rows, uint32_t cols, CoopVecMatrixLayout layout, DataType element_type);

    // Convenience function for building a matrix desc; calls query_matrix_size internally
    CoopVecMatrixDesc create_matrix_desc(
        uint32_t rows,
        uint32_t cols,
        CoopVecMatrixLayout layout,
        DataType element_type,
        size_t offset = 0
    );

    // Host-to-host conversion
    void convert_matrix_host(
        const void* src,
        size_t src_size,
        CoopVecMatrixDesc src_desc,
        void* dst,
        size_t dst_size,
        CoopVecMatrixDesc dst_desc
    );
    // Device-to-device conversion of single matrix
    void convert_matrix_device(
        const Buffer* src,
        CoopVecMatrixDesc src_desc,
        const Buffer* dst,
        CoopVecMatrixDesc dst_desc,
        CommandEncoder* encoder = nullptr
    );
    // Device-to-device conversion of multiple matrices
    void convert_matrix_device(
        const Buffer* src,
        const std::vector<CoopVecMatrixDesc>& src_desc,
        const Buffer* dst,
        const std::vector<CoopVecMatrixDesc>& dst_desc,
        CommandEncoder* encoder = nullptr
    );
    void convert_matrix_device(
        const Buffer* src,
        const CoopVecMatrixDesc* src_desc,
        const Buffer* dst,
        const CoopVecMatrixDesc* dst_desc,
        uint32_t matrix_count,
        CommandEncoder* encoder = nullptr
    );

    size_t align_matrix_offset(size_t offset)
    {
        return MATRIX_ALIGNMENT * ((offset + MATRIX_ALIGNMENT - 1) / MATRIX_ALIGNMENT);
    }
    size_t align_vector_offset(size_t offset)
    {
        return VECTOR_ALIGHMENT * ((offset + VECTOR_ALIGHMENT - 1) / VECTOR_ALIGHMENT);
    }

private:
    Device* m_device;
};

} // namespace sgl
