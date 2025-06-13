// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/reflection.h"

namespace sgl {

namespace cursor_utils {
    slang::TypeLayoutReflection* unwrap_array(slang::TypeLayoutReflection* layout);

    void check_array(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        size_t element_count
    );

    void check_scalar(slang::TypeLayoutReflection* type_layout, size_t size, TypeReflection::ScalarType scalar_type);

    void check_vector(
        slang::TypeLayoutReflection* type_layout,
        size_t size,
        TypeReflection::ScalarType scalar_type,
        int dimension
    );

    void check_matrix(
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
