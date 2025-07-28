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

public:
    void _set_array(const void* data, size_t size, TypeReflection::ScalarType scalar_type, size_t element_count) const
    {
        slang::TypeReflection* element_type = cursor_utils::unwrap_array(_get_slang_type_layout())->getType();
        size_t element_size
            = cursor_utils::get_scalar_type_size((TypeReflection::ScalarType)element_type->getScalarType());

#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_array(_get_slang_type_layout(), size, scalar_type, element_count);
#else
        SGL_UNUSED(scalar_type);
        SGL_UNUSED(element_count);
#endif

        size_t stride = _get_slang_type_layout()->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
        if (element_size == stride) {
            _set_data_internal(_get_offset_internal(), data, size);
        } else {
            auto offset = _get_offset_internal();
            for (size_t i = 0; i < element_count; ++i) {
                _set_data_internal(offset, reinterpret_cast<const uint8_t*>(data) + i * element_size, element_size);
                offset = _increment_offset_internal(offset, stride);
            }
        }
    }

    void _set_scalar(const void* data, size_t size, TypeReflection::ScalarType scalar_type) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_scalar(_get_slang_type_layout(), size, scalar_type);
#else
        SGL_UNUSED(scalar_type);
#endif
        _set_data_internal(_get_offset_internal(), data, size);
    }

    void _set_vector(const void* data, size_t size, TypeReflection::ScalarType scalar_type, int dimension) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_vector(_get_slang_type_layout(), size, scalar_type, dimension);
#else
        SGL_UNUSED(scalar_type);
        SGL_UNUSED(dimension);
#endif
        _set_data_internal(_get_offset_internal(), data, size);
    }

    void _set_matrix(const void* data, size_t size, TypeReflection::ScalarType scalar_type, int rows, int cols) const
    {
#ifdef SGL_ENABLE_CURSOR_TYPE_CHECKS
        cursor_utils::check_matrix(_get_slang_type_layout(), size, scalar_type, rows, cols);
#else
        SGL_UNUSED(scalar_type);
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

    void _set_bool(const bool& value) const
    {
#if SGL_MACOS
        if (_get_device_type_internal() == DeviceType::metal) {
            _set_scalar(&value, sizeof(value), TypeReflection::ScalarType::bool_);
            return;
        }
#endif
        uint32_t v = value ? 1 : 0;
        _set_scalar(&v, sizeof(v), TypeReflection::ScalarType::bool_);
    }

    template<int N>
    void _set_boolN(const sgl::math::vector<bool, N>& value) const
    {
#if SGL_MACOS
        if (_get_device_type_internal() == DeviceType::metal) {
            _set_vector(&value, sizeof(value), TypeReflection::ScalarType::bool_, 1);
            return;
        }
#endif

        sgl::math::vector<uint32_t, N> v;
        for (int i = 0; i < N; ++i)
            v[i] = value[i] ? 1 : 0;
        _set_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, N);
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

public:
    void _get_array(void* data, size_t size, TypeReflection::ScalarType scalar_type, size_t element_count) const
    {
        ref<const TypeReflection> element_type = _get_slang_type_layout()->unwrap_array()->type();
        size_t element_size = cursor_utils::get_scalar_type_size(element_type->scalar_type());

        cursor_utils::check_array(_get_slang_type_layout()->slang_target(), size, scalar_type, element_count);

        size_t stride = _get_slang_type_layout()->element_stride();
        if (element_size == stride) {
            _get_data_internal(_get_offset_internal(), data, size);
        } else {
            auto offset = _get_offset_internal();
            for (size_t i = 0; i < element_count; ++i) {
                read_data(offset, reinterpret_cast<uint8_t*>(data) + i * element_size, element_size);
                offset = _increment_offset_internal(offset, stride);
            }
        }
    }

    void _get_scalar(void* data, size_t size, TypeReflection::ScalarType scalar_type) const
    {
        cursor_utils::check_scalar(_get_slang_type_layout(), size, scalar_type);
        _get_data_internal(_get_offset_internal(), data, size);
    }

    void _get_vector(void* data, size_t size, TypeReflection::ScalarType scalar_type, int dimension) const
    {
        cursor_utils::check_vector(_get_slang_type_layout(), size, scalar_type, dimension);
        _get_data_internal(_get_offset_internal(), data, size);
    }

    void _get_matrix(void* data, size_t size, TypeReflection::ScalarType scalar_type, int rows, int cols) const
    {
        cursor_utils::check_matrix(_get_slang_type_layout(), size, scalar_type, rows, cols);
        size_t stride = _get_slang_type_layout()->getStride();
        if (stride != size) {
            size_t row_stride = stride / rows;
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

    void _get_bool(bool& value) const
    {
#if SGL_MACOS
        if (_get_device_type_internal() == DeviceType::metal) {
            _get_scalar(&value, sizeof(value), TypeReflection::ScalarType::bool_);
            return;
        }
#endif
        uint32_t v;
        _get_scalar(&v, sizeof(v), TypeReflection::ScalarType::bool_);
        value = (v != 0);
    }

    template<int N>
    void _get_boolN(sgl::math::vector<bool, N>& value) const
    {
#if SGL_MACOS
        if (_get_device_type_internal() == DeviceType::metal) {
            _get_vector(&value, sizeof(value), TypeReflection::ScalarType::bool_, N);
            return;
        }
#endif
        sgl::math::vector<uint32_t, N> v;
        _get_vector(&v, sizeof(v), TypeReflection::ScalarType::bool_, N);
        for (int i = 0; i < N; ++i)
            value[i] = (v[i] != 0);
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
