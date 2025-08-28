// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/device.h"
#include "sgl/device/reflection.h"
#include "sgl/device/cursor_utils.h"

// TODO: Decide if we want to disable / optimize type checks
// currently can represent 50% of the cost of writes in
// certain situations.
#define SGL_ENABLE_CURSOR_TYPE_CHECKS

namespace sgl {

template<typename BaseCursor, typename TOffset>
class SGL_API CursorWriteWrappers {
    using BaseCursorOffset = TOffset;

    // The array/vector of elements has two special cases where things are not tightly packed
    //
    // First one is bool, with the following options:
    //                cpu_element_size | element_stride | element_size
    // HLSL                          1 |              4 |            4
    // CUDA - array                  1 |              1 |            1
    // CUDA - vector (old)           1 |              4 |            1
    // CUDA - vector (new)           1 |              1 |            1
    //
    // When element_size != cpu_element_size, we need to convert between bool and uint32_t.
    // This is necessary to make sure we do not accidentally ignore bits 8-31 in either read or write.
    //
    // Further caveat is that CUDA - vector (old) says that the element_size == 1, but is actually implemented using int
    // in the backend, which is then cast to bool. But if we use that knowledge (to avoid ignoring bits 8-31),
    // we break future compatibility. So for CUDA vector (old), we are ignoring bits 8-31 in the boolX implementations,
    // and as such CUDA code that reports "true" because the stored value is 256, will report "false" on the CPU.
    //
    //
    // The other case is float4x3, where in HLSL we have a row-stride of 16B in some cases.

    void _set_array_or_vector(
        const void* data,
        size_t size,
        TypeReflection::ScalarType cpu_scalar_type,
        size_t element_count
    ) const
    {
        // CPU is assumed tightly packed, i.e., stride and size are the same value.
        size_t cpu_element_size = cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type);
        size_t element_stride = _get_slang_type_layout()->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
        // CUDA misreports the actual element stride, see https://github.com/shader-slang/slang/issues/7441
        // In the old implementation, bool2-4 are implemented as int2-4, and even though bool1 is implemented as int1,
        // the actual emitted code is bool. So for bool2-4, the element stride is 4, for bool1 it remains at 1.
        // The check for the total size == sizeof(int) * element_count is to disable this on newer Slang implementation,
        // where bool1-4 is implemented as an actual struct of 1-4 bools.
        if (cpu_scalar_type == TypeReflection::ScalarType::bool_ && _get_device_type_internal() == DeviceType::cuda
            && _get_slang_type_layout()->getKind() == slang::TypeReflection::Kind::Vector
            && _get_slang_type_layout()->getSize() == sizeof(int) * element_count) {
            if (element_count > 1)
                element_stride = 4;
        }
        size_t element_size = _get_slang_type_layout()->getElementTypeLayout()->getSize();

        SGL_CHECK(
            element_size <= element_stride,
            "Type `{}` is trying to write stride {} and size {}",
            cpu_scalar_type,
            element_stride,
            element_size
        );

        // The layout matches (both strides are the same), we can just write the data.
        if (element_stride == cpu_element_size) {
            _set_data_internal(_get_offset_internal(), data, size);
            return;
        }

        // The sizes match, but stride does not, so we need to write element-by-element.
        if (element_size == cpu_element_size) {
            auto offset = _get_offset_internal();
            for (size_t i = 0; i < element_count; ++i) {
                _set_data_internal(
                    offset,
                    reinterpret_cast<const uint8_t*>(data) + i * cpu_element_size,
                    cpu_element_size
                );
                offset = _increment_offset_internal(offset, element_stride);
            }
            return;
        }

        check_bool_conversion(_get_slang_type_layout()->getElementTypeLayout(), cpu_scalar_type);
        auto offset = _get_offset_internal();
        for (size_t i = 0; i < element_count; ++i) {
            bool src_value = *(const bool*)(reinterpret_cast<const uint8_t*>(data) + i * cpu_element_size);
            uint32_t dst_value = src_value ? 1u : 0u;
            _set_data_internal(offset, &dst_value, sizeof(dst_value));
            offset = _increment_offset_internal(offset, element_stride);
        }
    }

    void check_bool_conversion(
        slang::TypeLayoutReflection* element_type_layout,
        TypeReflection::ScalarType cpu_scalar_type
    ) const
    {
        size_t cpu_element_size = cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type);
        size_t element_size = element_type_layout->getSize();

        auto sgl_type = sgl::TypeReflection({}, _get_slang_type_layout()->getType());

        SGL_CHECK(
            cpu_element_size == 1 && element_size == 4 && cpu_scalar_type == TypeReflection::ScalarType::bool_,
            "We only support bool conversion from CPU size of 1B to GPU size of 4B, not converting \"{}\" from {} to "
            "{}. Slang layout: {}",
            cpu_scalar_type,
            cpu_element_size,
            element_size,
            sgl_type.full_name()
        );
    }

public:
    void
    _set_array(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, size_t element_count) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_array(_get_slang_type_layout(), size, cpu_scalar_type, element_count);
#endif
        _set_array_or_vector(data, size, cpu_scalar_type, element_count);
    }

    void _set_scalar(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_scalar(_get_slang_type_layout(), size, cpu_scalar_type);
#else
        SGL_UNUSED(scalar_type);
#endif
        size_t cpu_element_size = cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type);
        size_t element_size = _get_slang_type_layout()->getSize();

        if (cpu_element_size == element_size) {
            _set_data_internal(_get_offset_internal(), data, size);
            return;
        }

        check_bool_conversion(_get_slang_type_layout(), cpu_scalar_type);
        bool src_value = *reinterpret_cast<const bool*>(data);
        uint32_t dst_value = src_value ? 1u : 0u;
        _set_data_internal(_get_offset_internal(), &dst_value, sizeof(dst_value));
    }

    void _set_vector(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, int dimension) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_vector(_get_slang_type_layout(), size, cpu_scalar_type, dimension);
#endif

        _set_array_or_vector(data, size, cpu_scalar_type, dimension);
    }

    void
    _set_matrix(const void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, int rows, int cols) const
    {
        // matrix has element type (rows) which has element type (individual cells).
        // we are currently shortcuiting that logic only handling the case where float3x3 is in memory
        // represented as float3x4.
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        SGL_CHECK(
            (_get_slang_type_layout()->getElementTypeLayout()->getElementTypeLayout()->getSize())
                == cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type),
            "Mismatch between CPU ({}B) and GPU ({}B) element size cannot be handled.",
            cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type),
            _get_slang_type_layout()->getElementTypeLayout()->getElementTypeLayout()->getSize()
        );
        cursor_utils::check_matrix(_get_slang_type_layout(), size, cpu_scalar_type, rows, cols);
#else
        SGL_UNUSED(cpu_scalar_type);
        SGL_UNUSED(cols);
#endif

        if (rows > 1) {
            size_t mat_stride = _get_slang_type_layout()->getStride();
            size_t row_stride = mat_stride / rows;
            size_t row_size = size / rows;

            auto offset = _get_offset_internal();
            for (int row = 0; row < rows; ++row) {
                _set_data_internal(offset, reinterpret_cast<const uint8_t*>(data) + row * row_size, row_size);
                offset = _increment_offset_internal(offset, row_stride);
            }
        } else {
            _set_data_internal(_get_offset_internal(), data, size);
        }
    }

protected:
    CursorWriteWrappers() = default;

private:
    void _set_data_internal(BaseCursorOffset offset, const void* data, size_t size) const
    {
        static_cast<const BaseCursor*>(this)->_set_data(offset, data, size);
    }

    BaseCursorOffset _get_offset_internal() const { return static_cast<const BaseCursor*>(this)->_get_offset(); }
    BaseCursorOffset _increment_offset_internal(BaseCursorOffset offset, size_t diff) const
    {
        return BaseCursor::_increment_offset(offset, diff);
    }

    slang::TypeLayoutReflection* _get_slang_type_layout() const
    {
        return static_cast<const BaseCursor*>(this)->slang_type_layout();
    }

    DeviceType _get_device_type_internal() const { return static_cast<const BaseCursor*>(this)->_get_device_type(); }
};

template<typename BaseCursor, typename TOffset>
class SGL_API CursorReadWrappers {
    using BaseCursorOffset = TOffset;

    void _get_array_or_vector(void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, size_t element_count)
        const
    {
        // CPU is assumed tightly packed, i.e., stride and size are the same value.
        size_t cpu_element_size = cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type);
        size_t element_stride = _get_slang_type_layout()->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
        // CUDA misreports the actual element stride, see https://github.com/shader-slang/slang/issues/7441
        // In the old implementation, bool2-4 are implemented as int2-4, and even though bool1 is implemented as int1,
        // the actual emitted code is bool. So for bool2-4, the element stride is 4, for bool1 it remains at 1.
        // The check for the total size == sizeof(int) * element_count is to disable this on newer Slang implementation,
        // where bool1-4 is implemented as an actual struct of 1-4 bools.
        if (cpu_scalar_type == TypeReflection::ScalarType::bool_ && _get_device_type_internal() == DeviceType::cuda
            && _get_slang_type_layout()->getKind() == slang::TypeReflection::Kind::Vector
            && _get_slang_type_layout()->getSize() == sizeof(int) * element_count) {
            if (element_count > 1)
                element_stride = 4;
        }
        size_t element_size = _get_slang_type_layout()->getElementTypeLayout()->getSize();

        SGL_CHECK(
            element_size <= element_stride,
            "Type `{}` is trying to write stride {} and size {}",
            cpu_scalar_type,
            element_stride,
            element_size
        );

        // The layout matches (both strides are the same), we can just write the data.
        if (element_stride == cpu_element_size) {
            _get_data_internal(_get_offset_internal(), data, size);
            return;
        }

        // The sizes match, but stride does not, so we need to write element-by-element.
        if (element_size == cpu_element_size) {
            auto offset = _get_offset_internal();
            for (size_t i = 0; i < element_count; ++i) {
                _get_data_internal(offset, reinterpret_cast<uint8_t*>(data) + i * cpu_element_size, cpu_element_size);
                offset = _increment_offset_internal(offset, element_stride);
            }
            return;
        }

        check_bool_conversion(_get_slang_type_layout()->getElementTypeLayout(), cpu_scalar_type);
        auto offset = _get_offset_internal();
        for (size_t i = 0; i < element_count; ++i) {
            uint32_t src_value;
            _get_data_internal(offset, &src_value, sizeof(src_value));
            offset = _increment_offset_internal(offset, element_stride);

            bool* dst_value = (bool*)(reinterpret_cast<uint8_t*>(data) + i * cpu_element_size);
            *dst_value = src_value != 0;
        }
    }

    void check_bool_conversion(
        slang::TypeLayoutReflection* element_type_layout,
        TypeReflection::ScalarType cpu_scalar_type
    ) const
    {
        size_t cpu_element_size = cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type);
        size_t element_size = element_type_layout->getSize();

        SGL_CHECK(
            cpu_element_size == 1 && element_size == 4 && cpu_scalar_type == TypeReflection::ScalarType::bool_,
            "We only support bool conversion from CPU size of 1B to GPU size of 4B, not converting \"{}\" from {} to "
            "{}.",
            cpu_scalar_type,
            cpu_element_size,
            element_size
        );
    }


public:
    void _get_array(void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, size_t element_count) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_array(_get_slang_type_layout(), size, cpu_scalar_type, element_count);
#endif
        _get_array_or_vector(data, size, cpu_scalar_type, element_count);
    }

    void _get_scalar(void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_scalar(_get_slang_type_layout(), size, cpu_scalar_type);
#endif

        size_t cpu_element_size = cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type);
        size_t element_size = _get_slang_type_layout()->getSize();

        if (cpu_element_size == element_size) {
            _get_data_internal(_get_offset_internal(), data, size);
            return;
        }

        check_bool_conversion(_get_slang_type_layout(), cpu_scalar_type);
        uint32_t src_value;
        _get_data_internal(_get_offset_internal(), &src_value, sizeof(src_value));
        bool* dst_value = reinterpret_cast<bool*>(data);
        *dst_value = src_value != 0;
    }

    void _get_vector(void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, int dimension) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_vector(_get_slang_type_layout(), size, cpu_scalar_type, dimension);
#endif
        _get_array_or_vector(data, size, cpu_scalar_type, dimension);
    }

    void _get_matrix(void* data, size_t size, TypeReflection::ScalarType cpu_scalar_type, int rows, int cols) const
    {
        // matrix has element type (rows) which has element type (individual cells).
        // we are currently shortcuiting that logic only handling the case where float3x3 is in memory
        // represented as float3x4.
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        SGL_CHECK(
            _get_slang_type_layout()->getElementTypeLayout()->getElementTypeLayout()->getSize()
                == cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type),
            "Mismatch between CPU ({}B) and GPU ({}B) element size cannot be handled.",
            cursor_utils::get_scalar_type_cpu_size(cpu_scalar_type),
            _get_slang_type_layout()->getElementTypeLayout()->getElementTypeLayout()->getSize()
        );
        cursor_utils::check_matrix(_get_slang_type_layout(), size, cpu_scalar_type, rows, cols);
#endif

        if (rows > 1) {
            size_t mat_stride = _get_slang_type_layout()->getStride();
            size_t row_stride = mat_stride / rows;
            size_t row_size = size / rows;

            auto offset = _get_offset_internal();
            for (int i = 0; i < rows; ++i) {
                _get_data_internal(offset, reinterpret_cast<uint8_t*>(data) + i * row_size, row_size);
                offset = _increment_offset_internal(offset, row_stride);
            }
        } else {
            _get_data_internal(_get_offset_internal(), data, size);
        }
    }

protected:
    CursorReadWrappers() = default;

private:
    void _get_data_internal(BaseCursorOffset offset, void* data, size_t size) const
    {
        static_cast<const BaseCursor*>(this)->_get_data(offset, data, size);
    }

    BaseCursorOffset _get_offset_internal() const { return static_cast<const BaseCursor*>(this)->_get_offset(); }
    BaseCursorOffset _increment_offset_internal(BaseCursorOffset offset, size_t diff) const
    {
        return BaseCursor::_increment_offset(offset, diff);
    }

    slang::TypeLayoutReflection* _get_slang_type_layout() const
    {
        return static_cast<const BaseCursor*>(this)->slang_type_layout();
    }

    DeviceType _get_device_type_internal() const { return static_cast<const BaseCursor*>(this)->_get_device_type(); }
};


} // namespace sgl
