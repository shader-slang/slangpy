// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/reflection.h"

#include "sgl/core/macros.h"

namespace sgl {

namespace cursor_utils {
    // Get the CPU size of the scalar types
    inline size_t get_scalar_type_cpu_size(TypeReflection::ScalarType type)
    {
        switch (type) {
        case TypeReflection::ScalarType::bool_:
        case TypeReflection::ScalarType::int8:
        case TypeReflection::ScalarType::uint8:
            static_assert(sizeof(bool) == 1 && sizeof(int8_t) == 1 && sizeof(uint8_t) == 1);
            return 1;
        case TypeReflection::ScalarType::int16:
        case TypeReflection::ScalarType::uint16:
        case TypeReflection::ScalarType::float16:
            static_assert(sizeof(int16_t) == 2 && sizeof(uint16_t) == 2 && sizeof(float16_t) == 2);
            return 2;
        case TypeReflection::ScalarType::int32:
        case TypeReflection::ScalarType::uint32:
        case TypeReflection::ScalarType::float32:
            static_assert(sizeof(int32_t) == 4 && sizeof(uint32_t) == 4 && sizeof(float) == 4);
            return 4;
        case TypeReflection::ScalarType::int64:
        case TypeReflection::ScalarType::uint64:
        case TypeReflection::ScalarType::float64:
            static_assert(sizeof(int64_t) == 8 && sizeof(uint64_t) == 8 && sizeof(double) == 8);
            return 8;
        default:
            SGL_THROW("Unexpected ScalarType \"{}\"", type);
        }
    }

    SGL_API slang::TypeLayoutReflection* unwrap_array(slang::TypeLayoutReflection* layout);

    SGL_API void check_array(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        size_t element_count
    );

    SGL_API void
    check_scalar(slang::TypeLayoutReflection* type_layout, size_t size, TypeReflection::ScalarType scalar_type);

    SGL_API void check_vector(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        int dimension
    );

    SGL_API void check_matrix(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        int rows,
        int cols
    );
} // namespace cursor_utils

/// Dummy type to represent traits of an arbitrary value type usable by cursors
struct _AnyCursorValue { };

/// Concept that defines the requirements for a cursor that can be traversed using
/// field names and element indices. Each traversal function should return a new
/// cursor object that represents the field or element.
template<typename T>
concept TraversableCursor = requires(T obj, std::string_view name_idx, uint32_t el_index) {
    {
        obj[name_idx]
    } -> std::same_as<T>;
    {
        obj[el_index]
    } -> std::same_as<T>;
    {
        obj.find_field(name_idx)
    } -> std::same_as<T>;
    {
        obj.find_element(el_index)
    } -> std::same_as<T>;
    {
        obj.has_field(name_idx)
    } -> std::convertible_to<bool>;
    {
        obj.has_element(el_index)
    } -> std::convertible_to<bool>;
    {
        obj.slang_type_layout()
    } -> std::convertible_to<slang::TypeLayoutReflection*>;
    {
        obj.is_valid()
    } -> std::convertible_to<bool>;
};

/// Concept that defines the requirements for a cursor that can be read from.
template<typename T>
concept ReadableCursor = requires(
    T obj,
    void* data,
    size_t size,
    TypeReflection::ScalarType scalar_type,
    size_t element_count,
    _AnyCursorValue& val
) {
    {
        obj.template get<_AnyCursorValue>(val)
    } -> std::same_as<void>; // Ensure set() method exists
    {
        obj.template as<_AnyCursorValue>()
    } -> std::same_as<_AnyCursorValue>;
    {
        obj._get_array(data, size, scalar_type, element_count)
    } -> std::same_as<void>;
    {
        obj._get_scalar(data, size, scalar_type)
    } -> std::same_as<void>;
    {
        obj._get_vector(data, size, scalar_type, 0)
    } -> std::same_as<void>;
    {
        obj._get_matrix(data, size, scalar_type, 0, 0)
    } -> std::same_as<void>;
};

/// Concept that defines the requirements for a cursor that can be written to.
template<typename T>
concept WritableCursor
    = requires(T obj, void* data, size_t size, TypeReflection::ScalarType scalar_type, size_t element_count) {
          {
              obj.template set<_AnyCursorValue>({})
          } -> std::same_as<void>;
          {
              obj.template operator=<_AnyCursorValue>({})
          } -> std::same_as<void>;
          {
              obj._set_array(data, size, scalar_type, element_count)
          } -> std::same_as<void>;
          {
              obj._set_scalar(data, size, scalar_type)
          } -> std::same_as<void>;
          {
              obj._set_vector(data, size, scalar_type, 0)
          } -> std::same_as<void>;
          {
              obj._set_matrix(data, size, scalar_type, 0, 0)
          } -> std::same_as<void>;
      };

} // namespace sgl
