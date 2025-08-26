// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <optional>

#include "nanobind.h"

#include "sgl/device/reflection.h"
#include "sgl/device/cursor_utils.h"

#include "sgl/math/vector_types.h"
#include "sgl/math/matrix_types.h"

#include "utils/slangpypackedarg.h"
#include "utils/slangpystridedbufferview.h"
#include "sgl/device/buffer_cursor.h"

#include <slang.h>

namespace sgl {

/// Helper to convert from numpy type mask to slang scalar type.
inline std::optional<TypeReflection::ScalarType> dtype_to_scalar_type(nb::dlpack::dtype dtype)
{
    switch (dtype.code) {
    case uint8_t(nb::dlpack::dtype_code::Int):
        switch (dtype.bits) {
        case 8:
            return TypeReflection::ScalarType::int8;
        case 16:
            return TypeReflection::ScalarType::int16;
        case 32:
            return TypeReflection::ScalarType::int32;
        case 64:
            return TypeReflection::ScalarType::int64;
        }
        break;
    case uint8_t(nb::dlpack::dtype_code::UInt):
        switch (dtype.bits) {
        case 8:
            return TypeReflection::ScalarType::uint8;
        case 16:
            return TypeReflection::ScalarType::uint16;
        case 32:
            return TypeReflection::ScalarType::uint32;
        case 64:
            return TypeReflection::ScalarType::uint64;
        }
        break;
    case uint8_t(nb::dlpack::dtype_code::Float):
        switch (dtype.bits) {
        case 16:
            return TypeReflection::ScalarType::float16;
        case 32:
            return TypeReflection::ScalarType::float32;
        case 64:
            return TypeReflection::ScalarType::float64;
        }
        break;
    case uint8_t(nb::dlpack::dtype_code::Bool):
        switch (dtype.bits) {
        case 8:
            return TypeReflection::ScalarType::bool_;
        }
        break;
    }
    return {};
}

/// Enforces properties needed for converting to/from vector.
template<typename T>
concept IsSpecializationOfVector = requires {
    T::dimension;
    typename T::value_type;
};

/// Enforces properties needed for converting to/from matrix.
template<typename T>
concept IsSpecializationOfMatrix = requires {
    T::rows;
    T::cols;
    typename T::value_type;
};

#define scalar_case(c_type, scalar_type)                                                                               \
    m_read_scalar[(int)TypeReflection::ScalarType::scalar_type]                                                        \
        = [](const CursorType& self) { return _read_scalar<c_type>(self); };

#define vector_case(c_type, scalar_type)                                                                               \
    m_read_vector[(int)TypeReflection::ScalarType::scalar_type][c_type::dimension]                                     \
        = [](const CursorType& self) { return _read_vector<c_type>(self); };

#define matrix_case(c_type, scalar_type)                                                                               \
    m_read_matrix[(int)TypeReflection::ScalarType::scalar_type][c_type::rows][c_type::cols]                            \
        = [](const CursorType& self) { return _read_matrix<c_type>(self); };

/// Table of converters based on slang scalar type and shape.
template<typename CursorType>
class ReadConverterTable {
public:
    ReadConverterTable()
    {
        // Initialize all entries to an error function that throws an exception.
        auto read_err_func = [](const CursorType&) -> nb::object { SGL_THROW("Unsupported element type"); };
        for (int i = 0; i < (int)TypeReflection::ScalarType::COUNT; i++) {
            m_read_scalar[i] = read_err_func;
            for (int j = 0; j < 5; ++j) {
                m_read_vector[i][j] = read_err_func;
                for (int k = 0; k < 5; ++k) {
                    m_read_matrix[i][j][k] = read_err_func;
                }
            }
        }

        // Register converters for all supported scalar types.
        scalar_case(bool, bool_);
        scalar_case(int8_t, int8);
        scalar_case(uint8_t, uint8);
        scalar_case(int16_t, int16);
        scalar_case(uint16_t, uint16);
        scalar_case(int32_t, int32);
        scalar_case(uint32_t, uint32);
        scalar_case(int64_t, int64);
        scalar_case(uint64_t, uint64);
        scalar_case(float16_t, float16);
        scalar_case(float, float32);
        scalar_case(double, float64);
        scalar_case(intptr_t, intptr);
        scalar_case(uintptr_t, uintptr);

        // Register converters for all supported vector types.
        vector_case(bool1, bool_);
        vector_case(float1, float32);
        vector_case(float16_t1, float16);
        vector_case(int1, int32);
        vector_case(uint1, uint32);
        vector_case(bool2, bool_);
        vector_case(float2, float32);
        vector_case(float16_t2, float16);
        vector_case(int2, int32);
        vector_case(uint2, uint32);
        vector_case(bool3, bool_);
        vector_case(float3, float32);
        vector_case(float16_t3, float16);
        vector_case(int3, int32);
        vector_case(uint3, uint32);
        vector_case(bool4, bool_);
        vector_case(float4, float32);
        vector_case(float16_t4, float16);
        vector_case(int4, int32);
        vector_case(uint4, uint32);

        // Register converters for all supported matrix types.
        matrix_case(float2x2, float32);
        matrix_case(float2x3, float32);
        matrix_case(float2x4, float32);

        matrix_case(float3x2, float32);
        matrix_case(float3x3, float32);
        matrix_case(float3x4, float32);

        matrix_case(float4x2, float32);
        matrix_case(float4x3, float32);
        matrix_case(float4x4, float32);
    }

    /// Read function inspects the slang type and attempts to convert it
    /// to a matching python type. For structs and arrays, generates
    /// a nested dictionary or list and recurses.
    nb::object read(const CursorType& self)
    {
        m_stack.clear();
        try {
            return read_internal(self);
        } catch (const std::exception& err) {
            SGL_THROW("{}: {}", build_error(), err.what());
        }
    }

private:
    std::function<nb::object(const CursorType&)> m_read_scalar[(int)TypeReflection::ScalarType::COUNT];
    std::function<nb::object(const CursorType&)> m_read_vector[(int)TypeReflection::ScalarType::COUNT][5];
    std::function<nb::object(const CursorType&)> m_read_matrix[(int)TypeReflection::ScalarType::COUNT][5][5];
    std::vector<const char*> m_stack;

    std::string build_error() { return fmt::format("{}", fmt::join(m_stack, ".")); }

    nb::object read_internal(const CursorType& self)
    {
        if (!self.is_valid())
            return nb::none();

        auto type = self.type();
        if (type) {
            switch (type->kind()) {
            case TypeReflection::Kind::scalar: {
                return m_read_scalar[(int)type->scalar_type()](self);
            }
            case TypeReflection::Kind::vector: {
                return m_read_vector[(int)type->scalar_type()][type->col_count()](self);
            }
            case TypeReflection::Kind::matrix: {
                return m_read_matrix[(int)type->scalar_type()][type->row_count()][type->col_count()](self);
            }
            case TypeReflection::Kind::struct_: {
                nb::dict res;
                for (uint32_t i = 0; i < type->field_count(); i++) {
                    auto field = type->get_field_by_index(i);
                    const char* name = field->name();
                    m_stack.push_back(name);
                    res[name] = read_internal(self[name]);
                    m_stack.pop_back();
                }
                return res;
            }
            case TypeReflection::Kind::array: {
                nb::list res;
                m_stack.push_back("[]");
                for (uint32_t i = 0; i < type->element_count(); i++) {
                    res.append(read_internal(self[i]));
                }
                m_stack.pop_back();
                return res;
            }
            default:
                break;
            }
        }
        SGL_THROW("Unsupported element type");
    }

    /// Read scalar value from buffer element cursor and convert to Python object.
    template<typename ValType>
    inline static nb::object _read_scalar(const CursorType& self)
    {
        ValType res;
        self.get(res);
        return nb::cast(res);
    }

    /// Read vector value from buffer element cursor and convert to Python object.
    template<typename ValType>
        requires IsSpecializationOfVector<ValType>
    inline static nb::object _read_vector(const CursorType& self)
    {
        ValType res;
        self.get(res);
        return nb::cast(res);
    }

    /// Read matrix value from buffer element cursor and convert to Python object.
    template<typename ValType>
        requires IsSpecializationOfMatrix<ValType>
    inline static nb::object _read_matrix(const CursorType& self)
    {
        ValType res;
        self.get(res);
        return nb::cast(res);
    }
};

#undef scalar_case
#undef vector_case
#undef matrix_case

#define scalar_case(c_type, scalar_type)                                                                               \
    m_write_scalar[(int)TypeReflection::ScalarType::scalar_type]                                                       \
        = [](CursorType& self, nb::object nbval) { _write_scalar<c_type>(self, nbval); };                              \
    m_write_scalar_from_numpy[(int)TypeReflection::ScalarType::scalar_type]                                            \
        = [](CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)                                          \
    { _write_scalar_from_numpy<c_type>(self, nbarray); };

#define vector_case(c_type, scalar_type)                                                                               \
    m_write_vector[(int)TypeReflection::ScalarType::scalar_type][c_type::dimension]                                    \
        = [](CursorType& self, nb::object nbval) { _write_vector<c_type>(self, nbval); };                              \
    m_write_vector_from_numpy[(int)TypeReflection::ScalarType::scalar_type][c_type::dimension]                         \
        = [](CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)                                          \
    { _write_vector_from_numpy<c_type>(self, nbarray); };

#define bool_vector_case(c_type, scalar_type)                                                                          \
    m_write_vector[(int)TypeReflection::ScalarType::scalar_type][c_type::dimension]                                    \
        = [](CursorType& self, nb::object nbval) { _write_bool_vector<c_type>(self, nbval); };                         \
    m_write_vector_from_numpy[(int)TypeReflection::ScalarType::scalar_type][c_type::dimension]                         \
        = [](CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)                                          \
    { _write_bool_vector_from_numpy<c_type>(self, nbarray); };

#define matrix_case(c_type, scalar_type)                                                                               \
    m_write_matrix[(int)TypeReflection::ScalarType::scalar_type][c_type::rows][c_type::cols]                           \
        = [](CursorType& self, nb::object nbval) { _write_matrix<c_type>(self, nbval); };                              \
    m_write_matrix_from_numpy[(int)TypeReflection::ScalarType::scalar_type][c_type::rows][c_type::cols]                \
        = [](CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)                                          \
    { _write_matrix_from_numpy<c_type>(self, nbarray); };

/// Table of converters based on slang scalar type and shape.
template<typename CursorType>
class WriteConverterTable {
public:
    WriteConverterTable()
    {
        // Initialize all entries to an error function that throws an exception.
        auto write_err_func = [](const CursorType&, nb::object) { SGL_THROW("Unsupported element type"); };
        for (int i = 0; i < (int)TypeReflection::ScalarType::COUNT; i++) {
            m_write_scalar[i] = write_err_func;
            for (int j = 0; j < 5; ++j) {
                m_write_vector[i][j] = write_err_func;
                for (int k = 0; k < 5; ++k) {
                    m_write_matrix[i][j][k] = write_err_func;
                }
            }
        }

        // Register converters for all supported scalar types.
        scalar_case(bool, bool_);
        scalar_case(int8_t, int8);
        scalar_case(uint8_t, uint8);
        scalar_case(int16_t, int16);
        scalar_case(uint16_t, uint16);
        scalar_case(int32_t, int32);
        scalar_case(uint32_t, uint32);
        scalar_case(int64_t, int64);
        scalar_case(uint64_t, uint64);
        scalar_case(float16_t, float16);
        scalar_case(float, float32);
        scalar_case(double, float64);
        scalar_case(intptr_t, intptr);
        scalar_case(uintptr_t, uintptr);

        // Register converters for all supported vector types.
        bool_vector_case(bool1, bool_);
        vector_case(float1, float32);
        vector_case(float16_t1, float16);
        vector_case(int1, int32);
        vector_case(uint1, uint32);
        bool_vector_case(bool2, bool_);
        vector_case(float2, float32);
        vector_case(float16_t2, float16);
        vector_case(int2, int32);
        vector_case(uint2, uint32);
        bool_vector_case(bool3, bool_);
        vector_case(float3, float32);
        vector_case(float16_t3, float16);
        vector_case(int3, int32);
        vector_case(uint3, uint32);
        bool_vector_case(bool4, bool_);
        vector_case(float4, float32);
        vector_case(float16_t4, float16);
        vector_case(int4, int32);
        vector_case(uint4, uint32);

        // Register converters for all supported matrix types.
        matrix_case(float2x2, float32);
        matrix_case(float2x3, float32);
        matrix_case(float2x4, float32);

        matrix_case(float3x2, float32);
        matrix_case(float3x3, float32);
        matrix_case(float3x4, float32);

        matrix_case(float4x2, float32);
        matrix_case(float4x3, float32);
        matrix_case(float4x4, float32);
    }

    /// Virtual for writing none-basic value types.
    virtual bool write_value(CursorType& self, nb::object nbval)
    {
        SGL_UNUSED(self);
        SGL_UNUSED(nbval);
        return false;
    }

    /// Write function inspects the slang type and uses it to try
    /// and convert a Python input to the correct c++ type. For structs
    /// and arrays, expects a dict, sequence type or numpy array.
    void write(CursorType& self, nb::object nbval)
    {
        m_stack.clear();
        try {
            write_internal(self, nbval);
        } catch (const std::exception& err) {
            SGL_THROW("{}: {}", build_error(), err.what());
        }
    }

    /// Writes a (dictionary of) ndarrays to the whole buffer.
    /// First dimension of the ndarray is treated as an index into the buffer,
    /// and all the other dimensions are then written to the corresponding
    /// BufferElementCursors.
    /// If unchecked_copy is true, the numpy values are written directly to the backend storage,
    /// with only the basic check that the data fits into the backend store. Writing less is permitted,
    /// (e.g., float3 is 16B on Metal, with the last 4 bytes unused).
    /// Otherwise, the standard _set_type calls are invoked. This is especially dangerous if writing
    /// into bool type which has different size based on the storage.
    void write_from_numpy(BufferCursor& dst, nb::object nbval, bool unchecked_copy = false)
        requires std::same_as<CursorType, BufferElementCursor>
    {
        m_stack.clear();
        try {
            write_from_numpy_internal(dst, dst[0], nbval, unchecked_copy);
        } catch (const std::exception& err) {
            SGL_THROW("{}: {}", build_error(), err.what());
        }
    }

private:
    std::function<void(CursorType&, nb::object)> m_write_scalar[(int)TypeReflection::ScalarType::COUNT];
    std::function<void(CursorType&, nb::object)> m_write_vector[(int)TypeReflection::ScalarType::COUNT][5];
    std::function<void(CursorType&, nb::object)> m_write_matrix[(int)TypeReflection::ScalarType::COUNT][5][5];
    std::function<void(CursorType&, const nb::ndarray<nb::numpy, nb::ro>&)>
        m_write_scalar_from_numpy[(int)TypeReflection::ScalarType::COUNT];
    std::function<void(CursorType&, const nb::ndarray<nb::numpy, nb::ro>&)>
        m_write_vector_from_numpy[(int)TypeReflection::ScalarType::COUNT][5];
    std::function<void(CursorType&, const nb::ndarray<nb::numpy, nb::ro>&)>
        m_write_matrix_from_numpy[(int)TypeReflection::ScalarType::COUNT][5][5];
    std::vector<const char*> m_stack;

    std::string build_error() { return fmt::format("{}", fmt::join(m_stack, ".")); }

    void write_from_numpy_internal(BufferCursor& dst, BufferElementCursor self, nb::object nbval, bool unchecked_copy)
        requires std::same_as<CursorType, BufferElementCursor>
    {
        if (!self.is_valid())
            return;

        slang::TypeLayoutReflection* type_layout = self.slang_type_layout();
        auto kind = (TypeReflection::Kind)type_layout->getKind();

        switch (kind) {
        case TypeReflection::Kind::constant_buffer:
        case TypeReflection::Kind::parameter_block:
        case TypeReflection::Kind::struct_: {
            // Unwrap constant buffers or parameter blocks
            if (kind != TypeReflection::Kind::struct_)
                type_layout = type_layout->getElementTypeLayout();

            // Expect a dict for a slang struct.
            if (nb::isinstance<nb::dict>(nbval)) {
                auto dict = nb::cast<nb::dict>(nbval);
                for (uint32_t i = 0; i < type_layout->getFieldCount(); i++) {
                    auto field = type_layout->getFieldByIndex(i);
                    const char* name = field->getName();
                    auto child = self[name];
                    if (dict.contains(name)) {
                        m_stack.push_back(name);
                        write_from_numpy_internal(dst, child, dict[name], unchecked_copy);
                        m_stack.pop_back();
                    }
                }
                return;
            } else {
                SGL_THROW("Expected dict");
            }
        default:
            break;
        }
        }

        using AcceptedNDArrayType = nb::ndarray<nb::ro, nb::numpy>;
        SGL_CHECK(
            nb::isinstance<AcceptedNDArrayType>(nbval),
            "nbval is not ndarray but {}",
            nb::str(nbval.type()).c_str()
        );
        auto nbarray = nb::cast<AcceptedNDArrayType>(nbval);
        SGL_CHECK(
            nbarray.shape(0) == dst.element_count(),
            "First dimension of ndarray ({}) does not match the size of destination buffer ({})",
            nbarray.shape(0),
            dst.element_count()
        );


        const size_t element_byte_size = nbarray.itemsize();
        auto data = reinterpret_cast<const uint8_t*>(nbarray.data());
        const size_t element_byte_stride = nbarray.stride(0) * element_byte_size;

        if (unchecked_copy) {
            const size_t layout_compatible_size = element_byte_size * (nbarray.ndim() == 2 ? nbarray.shape(1) : 1);
            SGL_CHECK(
                layout_compatible_size <= self.type_layout()->size(),
                "Attempting to write element of size {} into a backing storage of size {} (per element).",
                layout_compatible_size,
                self.type_layout()->size()
            );
            for (size_t index = 0; index < dst.element_count(); ++index, data += element_byte_stride) {
                self.set_data(data, layout_compatible_size);
                self._set_offset(self.offset() + dst.element_type_layout()->stride());
            }
            return;
        }

        /// data for the child ndrrays
        const size_t shape1 = 1;
        size_t subarray_ndim = nbarray.ndim() == 1 ? 1 : nbarray.ndim() - 1;
        const size_t* subarray_shape
            = nbarray.ndim() == 1 ? &shape1 : (reinterpret_cast<const size_t*>(nbarray.shape_ptr()) + 1);

        // Gets the subarray of the current `data` pointer, caller needs to stride the data pointer.
        auto get_subarray = [&]()
        {
            return nb::ndarray<nb::numpy, nb::ro>(
                data,
                subarray_ndim,
                subarray_shape,
                {},
                nbarray.stride_ptr() + 1,
                nbarray.dtype(),
                nbarray.device_type(),
                nbarray.device_id()
            );
        };

        switch (kind) {
        case TypeReflection::Kind::scalar: {
            auto type = type_layout->getType();
            SGL_ASSERT(type);
            for (size_t index = 0; index < dst.element_count(); ++index, data += element_byte_stride) {
                m_write_scalar_from_numpy[(int)type->getScalarType()](self, get_subarray());
                self._set_offset(self.offset() + dst.element_type_layout()->stride());
            }
            break;
        }
        case TypeReflection::Kind::vector: {
            auto type = type_layout->getType();
            SGL_ASSERT(type);
            for (size_t index = 0; index < dst.element_count(); ++index, data += element_byte_stride) {
                m_write_vector_from_numpy[(int)type->getScalarType()][type->getColumnCount()](self, get_subarray());
                self._set_offset(self.offset() + dst.element_type_layout()->stride());
            }
            break;
        }
        case TypeReflection::Kind::matrix: {
            auto type = type_layout->getType();
            SGL_ASSERT(type);
            for (size_t index = 0; index < dst.element_count(); ++index, data += element_byte_stride) {
                m_write_matrix_from_numpy[(int)type->getScalarType()][type->getRowCount()][type->getColumnCount()](
                    self,
                    get_subarray()
                );
                self._set_offset(self.offset() + dst.element_type_layout()->stride());
            }
            break;
        }
        case TypeReflection::Kind::array: {
            for (size_t index = 0; index < dst.element_count(); ++index, data += element_byte_stride) {
                auto subarray = get_subarray();
                SGL_CHECK(subarray.ndim() == 1, "numpy array must have 1 dimension.");
                SGL_CHECK(subarray.shape(0) == type_layout->getElementCount(), "numpy array is the wrong length.");
                auto src_scalar_type = dtype_to_scalar_type(subarray.dtype());
                SGL_CHECK(is_ndarray_contiguous(subarray), "data is not contiguous");
                SGL_CHECK(src_scalar_type.has_value(), "unknown CPU type in numpy.");
                self._set_array(
                    subarray.data(),
                    subarray.nbytes(),
                    *src_scalar_type,
                    narrow_cast<int>(subarray.shape(0))
                );
                self._set_offset(self.offset() + dst.element_type_layout()->stride());
            }
            break;
        }
        default:
            for (size_t index = 0; index < dst.element_count(); ++index, data += element_byte_stride) {
                write_internal(self, get_subarray().cast());
                self._set_offset(self.offset() + dst.element_type_layout()->stride());
            }
            break;
        }
    }

    void write_internal(CursorType& self, nb::object nbval)
    {
        if (!self.is_valid())
            return;

        // Special case for handling DescriptorHandle.
        // These are reflected differently by slang depending on the backend.
        // For example, in D3D12 and Vulkan, they are of type uint2.
        // In Metal and CUDA, they are actual resource types.
        if (nb::isinstance<DescriptorHandle>(nbval)) {
            auto handle = nb::cast<DescriptorHandle>(nbval);
            self.set(handle);
            return;
        }

        // Special case for handling SlangPy PackedArg.
        if (nb::isinstance<sgl::slangpy::NativePackedArg>(nbval)) {
            auto packed_arg = nb::cast<sgl::slangpy::NativePackedArg*>(nbval);
            if constexpr (requires { self.set_object(nullptr); }) {
                self.set_object(packed_arg->shader_object());
            } else {
                SGL_THROW("A SlangPy packed argument can only be used as a function argument or shader cursor field.");
            }
            return;
        }

        // Read uniforms for StridedBufferView
        if (nb::isinstance<sgl::slangpy::StridedBufferView>(nbval)) {
            auto view = nb::cast<sgl::slangpy::StridedBufferView*>(nbval);
            nbval = view->uniforms();
        }

        slang::TypeLayoutReflection* type_layout = self.slang_type_layout();
        auto kind = (TypeReflection::Kind)type_layout->getKind();

        switch (kind) {
        case TypeReflection::Kind::scalar: {
            auto type = type_layout->getType();
            SGL_ASSERT(type);
            return m_write_scalar[(int)type->getScalarType()](self, nbval);
        }
        case TypeReflection::Kind::vector: {
            auto type = type_layout->getType();
            SGL_ASSERT(type);
            return m_write_vector[(int)type->getScalarType()][type->getColumnCount()](self, nbval);
        }
        case TypeReflection::Kind::matrix: {
            auto type = type_layout->getType();
            SGL_ASSERT(type);
            return m_write_matrix[(int)type->getScalarType()][type->getRowCount()][type->getColumnCount()](self, nbval);
        }
        case TypeReflection::Kind::pointer: {
            // Pointers are represented as uint64_t in slang.
            // We can write a pointer value directly.
            uint64_t as_ptr;
            if (nb::try_cast<uint64_t>(nbval, as_ptr)) {
                self.set_pointer(as_ptr);
                return;
            }

            Buffer* buffer;
            if (nb::try_cast<Buffer*>(nbval, buffer)) {
                // If we have a buffer, we can write its device address.
                self.set_pointer(buffer->device_address());
                return;
            }

            BufferView* buffer_view;
            if (nb::try_cast<BufferView*>(nbval, buffer_view)) {
                // If we have a view onto a buffer, write the buffer address plus the offset.
                self.set_pointer(buffer_view->buffer()->device_address() + buffer_view->range().offset);
                return;
            }

            sgl::slangpy::StridedBufferView* sbview;
            if (nb::try_cast<sgl::slangpy::StridedBufferView*>(nbval, sbview)) {
                // If we have a StridedBufferView, write address of storage plus offset.
                self.set_pointer(sbview->storage()->device_address() + sbview->offset());
                return;
            }

            SGL_THROW("Expected dict");
            return;
        }
        case TypeReflection::Kind::constant_buffer:
        case TypeReflection::Kind::parameter_block:
        case TypeReflection::Kind::struct_: {
            // Unwrap constant buffers or parameter blocks
            if (kind != TypeReflection::Kind::struct_)
                type_layout = type_layout->getElementTypeLayout();

            // Handle shader object if possible.
            if constexpr (requires { self.set_object(nullptr); }) {
                if (nb::isinstance<ShaderObject>(nbval)) {
                    self.set_object(nb::cast<ref<ShaderObject>>(nbval));
                    return;
                }
            }

            // Expect a dict for a slang struct.
            if (nb::isinstance<nb::dict>(nbval)) {
                auto dict = nb::cast<nb::dict>(nbval);
                for (uint32_t i = 0; i < type_layout->getFieldCount(); i++) {
                    auto field = type_layout->getFieldByIndex(i);
                    const char* name = field->getName();
                    auto child = self[name];
                    if (dict.contains(name)) {
                        m_stack.push_back(name);
                        write_internal(child, dict[name]);
                        m_stack.pop_back();
                    }
                }
                return;
            } else {
                SGL_THROW("Expected dict");
            }
        }
        case TypeReflection::Kind::array: {
            // Expect numpy array or sequence for a slang array.
            if (nb::isinstance<nb::ndarray<nb::numpy>>(nbval)) {
                // TODO: Should be able to do better job of interpreting nb array values by reading
                // data type and extracting individual elements.
                auto nbarray = nb::cast<nb::ndarray<nb::numpy>>(nbval);
                SGL_CHECK(nbarray.ndim() == 1, "numpy array must have 1 dimension.");
                SGL_CHECK(nbarray.shape(0) == type_layout->getElementCount(), "numpy array is the wrong length.");
                auto src_scalar_type = dtype_to_scalar_type(nbarray.dtype());
                SGL_CHECK(is_ndarray_contiguous(nbarray), "data is not contiguous");
                SGL_CHECK(src_scalar_type.has_value(), "unknown CPU type in numpy.");
                self._set_array(nbarray.data(), nbarray.nbytes(), *src_scalar_type, narrow_cast<int>(nbarray.shape(0)));
                return;
            } else if (nb::isinstance<nb::sequence>(nbval)) {
                auto seq = nb::cast<nb::sequence>(nbval);
                SGL_CHECK(
                    nb::len(seq) == type_layout->getElementCount(),
                    "sequence is the wrong length accessing type {}: {} != {}.",
                    type_layout->getType()->getName(),
                    nb::len(seq),
                    type_layout->getElementCount()
                );
                m_stack.push_back("[]");
                for (uint32_t i = 0; i < type_layout->getElementCount(); i++) {
                    auto child = self[i];
                    write_internal(child, seq[i]);
                }
                m_stack.pop_back();
                return;
            } else {
                SGL_THROW("Expected list");
            }
        }
        default:
            break;
        }

        // In default case call the virtual write_value, and fail if it returns false.
        if (write_value(self, nbval))
            return;

        SGL_THROW("Unsupported element type: {}", kind);
    }

    template<typename ValType>
    inline static void _write_scalar_from_numpy(CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)
    {
        auto val = *reinterpret_cast<const ValType*>(nbarray.data());
        self.set(val);
    }

    /// Write scalar value to buffer element cursor from Python object.
    template<typename ValType>
    inline static void _write_scalar(CursorType& self, nb::object nbval)
    {
        // Avoid warning about converting from numpy array to scalar
        if (nb::isinstance<nb::ndarray<nb::numpy>>(nbval)) {
            auto nbarray = nb::cast<nb::ndarray<nb::numpy>>(nbval);
            auto val = *reinterpret_cast<ValType*>(nbarray.data());
            self.set(val);
            return;
        }
        auto val = nb::cast<ValType>(nbval);
        self.set(val);
    }

    /// Default implementation of write vector from numpy array.
    template<typename ValType>
        requires IsSpecializationOfVector<ValType>
    inline static void _write_vector_from_numpy(CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)
    {
        SGL_CHECK(nbarray.nbytes() == sizeof(ValType), "numpy array has wrong size.");
        auto val = *reinterpret_cast<const ValType*>(nbarray.data());
        self.set(val);
    }

    /// Version of vector write specifically for bool vectors (which are stored as uint32_t)
    /// TODO: This is only special case to avoid the size check in _write_vector_from_numpy,
    /// but that already duplicates work of the checks in cursor_utils
    template<typename ValType>
        requires IsSpecializationOfVector<ValType>
    inline static void _write_bool_vector_from_numpy(CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)
    {
        self._set_vector(nbarray.data(), nbarray.nbytes(), TypeReflection::ScalarType::bool_, ValType::dimension);
    }

    /// Write vector value to buffer element cursor from Python object
    template<typename ValType>
        requires IsSpecializationOfVector<ValType>
    inline static void _write_vector(CursorType& self, nb::object nbval)
    {
        if (nb::isinstance<ValType>(nbval)) {
            // A vector of the correct type - just convert it.
            auto val = nb::cast<ValType>(nbval);
            self.set(val);
        } else if (nb::isinstance<nb::ndarray<nb::numpy>>(nbval)) {
            // A numpy array. Reinterpret numpy memory as vector type.
            nb::ndarray<nb::numpy, nb::ro> nbarray = nb::cast<nb::ndarray<nb::numpy, nb::ro>>(nbval);
            SGL_CHECK(is_ndarray_contiguous(nbarray), "data is not contiguous");
            SGL_CHECK(nbarray.ndim() == 1 || nbarray.ndim() == 2, "numpy array must have 1 or 2 dimensions.");
            size_t dimension = 1;
            for (size_t i = 0; i < nbarray.ndim(); ++i)
                dimension *= nbarray.shape(i);
            SGL_CHECK(dimension == ValType::dimension, "numpy array has wrong dimension.");
            _write_vector_from_numpy<ValType>(self, nbarray);
        } else if (nb::isinstance<nb::sequence>(nbval)) {
            // A list or tuple. Attempt to cast each element of list to element of vector.
            auto seq = nb::cast<nb::sequence>(nbval);
            SGL_CHECK(nb::len(seq) == ValType::dimension, "sequence has wrong dimension.");
            ValType val;
            for (int i = 0; i < ValType::dimension; i++) {
                val[i] = nb::cast<typename ValType::value_type>(seq[i]);
            }
            self.set(val);
        } else {
            SGL_THROW("Expected numpy array or vector");
        }
    }

    /// Bespoke vector implementation for bools.
    template<typename ValType>
        requires IsSpecializationOfVector<ValType>
    inline static void _write_bool_vector(CursorType& self, nb::object nbval)
    {
        if (nb::isinstance<ValType>(nbval)) {
            // A vector of the correct type - just convert it.
            auto val = nb::cast<ValType>(nbval);
            self.set(val);
        } else if (nb::isinstance<nb::ndarray<nb::numpy>>(nbval)) {
            // A numpy array. Reinterpret numpy memory as vector type.
            nb::ndarray<nb::numpy, nb::ro> nbarray = nb::cast<nb::ndarray<nb::numpy, nb::ro>>(nbval);
            SGL_CHECK(is_ndarray_contiguous(nbarray), "data is not contiguous");
            SGL_CHECK(nbarray.ndim() == 1 || nbarray.ndim() == 2, "numpy array must have 1 or 2 dimensions.");
            size_t dimension = 1;
            for (size_t i = 0; i < nbarray.ndim(); ++i)
                dimension *= nbarray.shape(i);
            SGL_CHECK(dimension == ValType::dimension, "numpy array has wrong dimension.");
            _write_bool_vector_from_numpy<ValType>(self, nbarray);
        } else if (nb::isinstance<nb::sequence>(nbval)) {
            // A list or tuple. Attempt to cast each element of list to element of vector.
            auto seq = nb::cast<nb::sequence>(nbval);
            SGL_CHECK(nb::len(seq) == ValType::dimension, "sequence has wrong dimension.");
            ValType val;
            for (int i = 0; i < ValType::dimension; i++) {
                val[i] = nb::cast<typename ValType::value_type>(seq[i]);
            }
            self.set(val);
        } else {
            SGL_THROW("Expected numpy array or vector");
        }
    }

    /// Write matrix value to buffer element cursor from Python object.
    template<typename ValType>
        requires IsSpecializationOfMatrix<ValType>
    inline static void _write_matrix_from_numpy(CursorType& self, const nb::ndarray<nb::numpy, nb::ro>& nbarray)
    {
        SGL_CHECK(nbarray.nbytes() == sizeof(ValType), "numpy array has wrong size.");
        auto val = *reinterpret_cast<const ValType*>(nbarray.data());
        self.set(val);
    }

    /// Write matrix value to buffer element cursor from Python object.
    template<typename ValType>
        requires IsSpecializationOfMatrix<ValType>
    inline static void _write_matrix(CursorType& self, nb::object nbval)
    {
        if (nb::isinstance<ValType>(nbval) || nb::isinstance<nb::ndarray<nb::numpy>>(nbval)) {
            // Matrix of correct type
            auto val = nb::cast<ValType>(nbval);
            self.set(val);
        } else {
            SGL_THROW("Expected numpy array or matrix");
        }
    }
};

#undef scalar_case
#undef vector_case
#undef bool_vector_case
#undef matrix_case

namespace {

    template<typename CursorType>
        requires TraversableCursor<CursorType>
    Py_ssize_t len(CursorType& cursor)
    {
        slang::TypeLayoutReflection* slang_type_layout = cursor.slang_type_layout();
        slang::TypeReflection::Kind kind = slang_type_layout->getKind();
        switch (kind) {
        case slang::TypeReflection::Kind::Array:
            return Py_ssize_t(slang_type_layout->getElementCount());
            break;
        case slang::TypeReflection::Kind::Matrix:
            return Py_ssize_t(slang_type_layout->getRowCount() * slang_type_layout->getColumnCount());
            break;
        case slang::TypeReflection::Kind::Vector:
            return Py_ssize_t(slang_type_layout->getColumnCount());
            break;
        }

        return 0;
    }

} // namespace

template<typename CursorType>
    requires TraversableCursor<CursorType>
inline void bind_traversable_cursor(nanobind::class_<CursorType>& cursor)
{
    cursor //
        .def("is_valid", &CursorType::is_valid, D_NA(CursorType, is_valid))
        .def("find_field", &CursorType::find_field, "name"_a, D_NA(CursorType, find_field))
        .def("find_element", &CursorType::find_element, "index"_a, D_NA(CursorType, find_element))
        .def("has_field", &CursorType::has_field, "name"_a, D_NA(CursorType, has_field))
        .def("has_element", &CursorType::has_element, "index"_a, D_NA(CursorType, has_element))
        .def("__getitem__", [](CursorType& self, std::string_view name) { return self[name]; })
        .def(
            "__getitem__",
            [](CursorType& self, Py_ssize_t index)
            {
                index = detail::sanitize_getitem_index(index, len(self));
                return self[uint32_t(index)];
            }
        )
        // note: __getattr__ should not except if field is not found
        .def("__getattr__", [](CursorType& self, std::string_view name) { return self.find_field(name); });
}

template<typename CursorType>
inline void bind_writable_cursor(WriteConverterTable<CursorType>& table, nanobind::class_<CursorType>& cursor)
{
    // __setitem__ and __setattr__ functions are overloaded to allow direct setting
    // of fields and elements.
    cursor //
        .def(
            "__setattr__",
            [&table](CursorType& self, std::string_view name, nb::object nbval)
            {
                auto child = self[name];
                table.write(child, nbval);
            },
            "name"_a,
            "val"_a,
            D_NA(CursorType, write)
        )
        .def(
            "__setitem__",
            [&table](CursorType& self, std::string_view name, nb::object nbval)
            {
                auto child = self[name];
                table.write(child, nbval);
            },
            "index"_a,
            "val"_a,
            D_NA(CursorType, write)
        )
        .def(
            "__setitem__",
            [&table](CursorType& self, int index, nb::object nbval)
            {
                auto child = self[index];
                table.write(child, nbval);
            },
            "index"_a,
            "val"_a,
            D_NA(CursorType, write)
        )
        .def(
            "set_data",
            [](CursorType& self, nb::ndarray<nb::device::cpu> data)
            {
                SGL_CHECK(is_ndarray_contiguous(data), "data is not contiguous");
                self.set_data(data.data(), data.nbytes());
            },
            "data"_a,
            D(ShaderCursor, set_data)
        )
        .def(
            "write",
            [&table](CursorType& self, nb::object nbval) { table.write(self, nbval); },
            "val"_a,
            D_NA(CursorType, write)
        );
}

template<typename CursorType>
inline void bind_readable_cursor(ReadConverterTable<CursorType>& table, nanobind::class_<CursorType>& cursor)
{
    // __setitem__ and __setattr__ functions are overloaded to allow direct setting
    // of fields and elements.
    cursor //
        .def(
            "read",
            [&table](CursorType& self) { return table.read(self); },
            D_NA(CursorType, read)
        );
}

template<typename CursorType>
    requires WritableCursor<CursorType>
inline void bind_writable_cursor_basic_types(nanobind::class_<CursorType>& cursor)
{
#define def_setter(type)                                                                                               \
    cursor.def("__setitem__", [](CursorType& self, std::string_view name, type value) { self[name] = value; });        \
    cursor.def("__setattr__", [](CursorType& self, std::string_view name, type value) { self[name] = value; });

    def_setter(bool);
    def_setter(bool1);
    def_setter(bool2);
    def_setter(bool3);
    def_setter(bool4);

    def_setter(uint1);
    def_setter(uint2);
    def_setter(uint3);
    def_setter(uint4);

    def_setter(int1);
    def_setter(int2);
    def_setter(int3);
    def_setter(int4);

    def_setter(float1);
    def_setter(float2);
    def_setter(float3);
    def_setter(float4);

    def_setter(float2x2);
    def_setter(float3x3);
    def_setter(float2x4);
    def_setter(float3x4);
    def_setter(float4x4);

    def_setter(float16_t2);
    def_setter(float16_t3);
    def_setter(float16_t4);

#undef def_setter

    auto set_int_field = [](CursorType& self, std::string_view name, nb::int_ value)
    {
        ref<const TypeReflection> type = self[name].type();
        SGL_CHECK(type->kind() == TypeReflection::Kind::scalar, "Field \"{}\" is not a scalar type.", name);
        switch (type->scalar_type()) {
        case TypeReflection::ScalarType::int16:
            self[name] = nb::cast<int16_t>(value);
            break;
        case TypeReflection::ScalarType::int32:
            self[name] = nb::cast<int32_t>(value);
            break;
        case TypeReflection::ScalarType::int64:
            self[name] = nb::cast<int64_t>(value);
            break;
        case TypeReflection::ScalarType::uint16:
            self[name] = nb::cast<uint16_t>(value);
            break;
        case TypeReflection::ScalarType::uint32:
            self[name] = nb::cast<uint32_t>(value);
            break;
        case TypeReflection::ScalarType::uint64:
            self[name] = nb::cast<uint64_t>(value);
            break;
        default:
            SGL_THROW("Field \"{}\" is not an integer type.");
            break;
        }
    };

    auto set_int_element = [](CursorType& self, int index, nb::int_ value)
    {
        ref<const TypeReflection> type = self[index].type();
        SGL_CHECK(type->kind() == TypeReflection::Kind::scalar, "Element {} is not a scalar type.", index);
        switch (type->scalar_type()) {
        case TypeReflection::ScalarType::int16:
            self[index] = nb::cast<int16_t>(value);
            break;
        case TypeReflection::ScalarType::int32:
            self[index] = nb::cast<int32_t>(value);
            break;
        case TypeReflection::ScalarType::int64:
            self[index] = nb::cast<int64_t>(value);
            break;
        case TypeReflection::ScalarType::uint16:
            self[index] = nb::cast<uint16_t>(value);
            break;
        case TypeReflection::ScalarType::uint32:
            self[index] = nb::cast<uint32_t>(value);
            break;
        case TypeReflection::ScalarType::uint64:
            self[index] = nb::cast<uint64_t>(value);
            break;
        default:
            SGL_THROW("Element {} is not an integer type.");
            break;
        }
    };

    cursor.def("__setitem__", set_int_field);
    cursor.def("__setitem__", set_int_element);
    cursor.def("__setattr__", set_int_field);

    auto set_float_field = [](CursorType& self, std::string_view name, nb::float_ value)
    {
        ref<const TypeReflection> type = self[name].type();
        SGL_CHECK(type->kind() == TypeReflection::Kind::scalar, "Field \"{}\" is not a scalar type.", name);
        switch (type->scalar_type()) {
        case TypeReflection::ScalarType::float16:
            self[name] = float16_t(nb::cast<float>(value));
            break;
        case TypeReflection::ScalarType::float32:
            self[name] = nb::cast<float>(value);
            break;
        case TypeReflection::ScalarType::float64:
            self[name] = nb::cast<double>(value);
            break;
        default:
            SGL_THROW("Field \"{}\" is not a floating point type.");
            break;
        }
    };

    auto set_float_element = [](CursorType& self, int index, nb::float_ value)
    {
        ref<const TypeReflection> type = self[index].type();
        SGL_CHECK(type->kind() == TypeReflection::Kind::scalar, "Element {} is not a scalar type.", index);
        switch (type->scalar_type()) {
        case TypeReflection::ScalarType::float16:
            self[index] = float16_t(nb::cast<float>(value));
            break;
        case TypeReflection::ScalarType::float32:
            self[index] = nb::cast<float>(value);
            break;
        case TypeReflection::ScalarType::float64:
            self[index] = nb::cast<double>(value);
            break;
        default:
            SGL_THROW("Element {} is not a floating point type.");
            break;
        }
    };

    cursor.def("__setitem__", set_float_field);
    cursor.def("__setitem__", set_float_element);
    cursor.def("__setattr__", set_float_field);

    auto set_numpy_field = [](CursorType& self, std::string_view name, nb::ndarray<nb::numpy> value)
    {
        ref<const TypeReflection> type = self[name].type();
        auto src_scalar_type = dtype_to_scalar_type(value.dtype());
        SGL_CHECK(src_scalar_type, "numpy array has unsupported dtype.");
        SGL_CHECK(is_ndarray_contiguous(value), "numpy array is not contiguous.");

        switch (type->kind()) {
        case TypeReflection::Kind::array:
            SGL_CHECK(value.ndim() == 1, "numpy array must have 1 dimension.");
            self[name]._set_array(value.data(), value.nbytes(), *src_scalar_type, narrow_cast<int>(value.shape(0)));
            break;
        case TypeReflection::Kind::matrix:
            SGL_CHECK(value.ndim() == 2, "numpy array must have 2 dimensions.");
            self[name]._set_matrix(
                value.data(),
                value.nbytes(),
                *src_scalar_type,
                narrow_cast<int>(value.shape(0)),
                narrow_cast<int>(value.shape(1))
            );
            break;
        case TypeReflection::Kind::vector: {
            SGL_CHECK(value.ndim() == 1 || value.ndim() == 2, "numpy array must have 1 or 2 dimensions.");
            size_t dimension = 1;
            for (size_t i = 0; i < value.ndim(); ++i)
                dimension *= value.shape(i);
            self[name]._set_vector(value.data(), value.nbytes(), *src_scalar_type, narrow_cast<int>(dimension));
            break;
        }
        default:
            SGL_THROW("Field \"{}\" is not a vector, matrix, or array type.", name);
        }
    };

    cursor.def("__setitem__", set_numpy_field);
    cursor.def("__setattr__", set_numpy_field);
}

} // namespace sgl
