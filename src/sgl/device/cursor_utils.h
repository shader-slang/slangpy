// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/reflection.h"

#include "sgl/core/macros.h"
#include "sgl/core/signature_buffer.h"

#include <concepts>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sgl {

// Concept to detect if T has write_to_cursor method
template<typename T, typename TCursor>
concept HasWriteToCursor = requires(const T& obj, TCursor& cursor) {
    { obj.write_to_cursor(cursor) };
};

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

    using ShaderCursorObjectWriteFunc = std::function<bool(ShaderCursor&, const void*)>;
    using BufferElementCursorObjectWriteFunc = std::function<bool(BufferElementCursor&, const void*)>;

    struct CursorWriterTypeInfo {
        const std::type_info* type{nullptr};

        ShaderCursorObjectWriteFunc write_shader_cursor;
        BufferElementCursorObjectWriteFunc write_buffer_cursor;

        bool has_functional_metadata{false};
        std::function<std::string_view(const void*)> slang_type_name;
        std::function<void(SignatureBuffer&, const void*)> write_signature;
        std::vector<std::string> imports;
    };

    template<typename T>
    struct CursorWriterTraits { };

    SGL_API void register_cursor_writer_type(CursorWriterTypeInfo info);
    SGL_API std::span<const CursorWriterTypeInfo> cursor_writer_type_infos();
    SGL_API const CursorWriterTypeInfo* find_cursor_writer_type_info(const std::type_info& type);

    SGL_API void register_shader_cursor_object_writer(const std::type_info& type, ShaderCursorObjectWriteFunc write);
    SGL_API void
    register_buffer_element_cursor_object_writer(const std::type_info& type, BufferElementCursorObjectWriteFunc write);

    template<typename T>
    void register_shader_cursor_object_writer(std::function<bool(ShaderCursor&, const T&)> write)
    {
        register_shader_cursor_object_writer(
            typeid(T),
            [write = std::move(write)](ShaderCursor& cursor, const void* value)
            {
                return write(cursor, *static_cast<const T*>(value));
            }
        );
    }

    template<typename T>
    void register_buffer_element_cursor_object_writer(std::function<bool(BufferElementCursor&, const T&)> write)
    {
        register_buffer_element_cursor_object_writer(
            typeid(T),
            [write = std::move(write)](BufferElementCursor& cursor, const void* value)
            {
                return write(cursor, *static_cast<const T*>(value));
            }
        );
    }

    template<typename T>
    void register_shader_cursor_object_writer()
    {
        register_shader_cursor_object_writer<T>(
            [](ShaderCursor& cursor, const T& value)
            {
                cursor = value;
                return true;
            }
        );
    }

    template<typename T>
    void register_buffer_element_cursor_object_writer()
    {
        register_buffer_element_cursor_object_writer<T>(
            [](BufferElementCursor& cursor, const T& value)
            {
                cursor = value;
                return true;
            }
        );
    }

    template<typename T>
    concept HasCursorWriterTraitsSlangTypeName = requires {
        { std::string_view(CursorWriterTraits<T>::slang_type_name) } -> std::same_as<std::string_view>;
    };

    template<typename T>
    concept HasTypeStaticSlangTypeName = requires {
        { std::string_view(T::slang_type_name) } -> std::same_as<std::string_view>;
    };

    template<typename T>
    concept HasValueSlangTypeName = requires(const T& value) {
        { value.slang_type_name() } -> std::convertible_to<std::string_view>;
    };

    template<typename T>
    concept HasCursorWriterTraitsSignature = requires(SignatureBuffer& signature) {
        { CursorWriterTraits<T>::write_slangpy_signature(signature) };
    };

    template<typename T>
    concept HasTypeStaticSignature = requires(SignatureBuffer& signature) {
        { T::write_slangpy_signature(signature) };
    };

    template<typename T>
    concept HasValueSignature = requires(const T& value, SignatureBuffer& signature) {
        { value.write_slangpy_signature(signature) };
    };

    template<typename T>
    concept HasCursorWriterTraitsImports = requires {
        { CursorWriterTraits<T>::slangpy_imports() };
    };

    template<typename T>
    concept HasTypeStaticImports = requires {
        { T::slangpy_imports() };
    };

    template<typename T>
    constexpr bool has_static_slang_type_name_v
        = HasCursorWriterTraitsSlangTypeName<T> || HasTypeStaticSlangTypeName<T>;

    template<typename T>
    std::string_view static_slang_type_name()
        requires(has_static_slang_type_name_v<T>)
    {
        if constexpr (HasCursorWriterTraitsSlangTypeName<T>)
            return std::string_view(CursorWriterTraits<T>::slang_type_name);
        else
            return std::string_view(T::slang_type_name);
    }

    template<typename T>
    void write_static_slangpy_signature(SignatureBuffer& signature)
        requires(has_static_slang_type_name_v<T>)
    {
        if constexpr (HasCursorWriterTraitsSignature<T>)
            CursorWriterTraits<T>::write_slangpy_signature(signature);
        else if constexpr (HasTypeStaticSignature<T>)
            T::write_slangpy_signature(signature);
        else
            signature.add(static_slang_type_name<T>());
    }

    template<typename TImports>
    void append_cursor_writer_imports(std::vector<std::string>& result, TImports&& imports)
    {
        for (auto&& import_path : imports)
            result.emplace_back(std::string_view(import_path));
    }

    template<typename T>
    std::vector<std::string> cursor_writer_imports()
    {
        std::vector<std::string> result;
        if constexpr (HasCursorWriterTraitsImports<T>)
            append_cursor_writer_imports(result, CursorWriterTraits<T>::slangpy_imports());
        else if constexpr (HasTypeStaticImports<T>)
            append_cursor_writer_imports(result, T::slangpy_imports());
        return result;
    }

    template<typename T>
    void register_cursor_writer()
        requires(HasWriteToCursor<T, ShaderCursor> || HasWriteToCursor<T, BufferElementCursor>)
    {
        CursorWriterTypeInfo info;
        info.type = &typeid(T);

        if constexpr (HasWriteToCursor<T, ShaderCursor>) {
            info.write_shader_cursor = [](ShaderCursor& cursor, const void* value)
            {
                static_cast<const T*>(value)->write_to_cursor(cursor);
                return true;
            };
        }

        if constexpr (HasWriteToCursor<T, BufferElementCursor>) {
            info.write_buffer_cursor = [](BufferElementCursor& cursor, const void* value)
            {
                static_cast<const T*>(value)->write_to_cursor(cursor);
                return true;
            };
        }

        if constexpr (has_static_slang_type_name_v<T>) {
            info.has_functional_metadata = true;
            info.slang_type_name = [](const void*) -> std::string_view
            {
                return static_slang_type_name<T>();
            };
            info.write_signature = [](SignatureBuffer& signature, const void*)
            {
                write_static_slangpy_signature<T>(signature);
            };
            info.imports = cursor_writer_imports<T>();
        } else if constexpr (HasValueSlangTypeName<T>) {
            static_assert(
                HasValueSignature<T>,
                "Cursor writer types with value-aware slang_type_name() must provide write_slangpy_signature()."
            );
            info.has_functional_metadata = true;
            info.slang_type_name = [](const void* value) -> std::string_view
            {
                return static_cast<const T*>(value)->slang_type_name();
            };
            info.write_signature = [](SignatureBuffer& signature, const void* value)
            {
                static_cast<const T*>(value)->write_slangpy_signature(signature);
            };
            info.imports = cursor_writer_imports<T>();
        }

        register_cursor_writer_type(std::move(info));
    }
} // namespace cursor_utils

/// Dummy type to represent traits of an arbitrary value type usable by cursors
struct _AnyCursorValue { };

/// Concept that defines the requirements for a cursor that can be traversed using
/// field names and element indices. Each traversal function should return a new
/// cursor object that represents the field or element.
template<typename T>
concept TraversableCursor = requires(T obj, std::string_view name_idx, uint32_t el_index) {
    { obj[name_idx] } -> std::same_as<T>;
    { obj[el_index] } -> std::same_as<T>;
    { obj.find_field(name_idx) } -> std::same_as<T>;
    { obj.find_element(el_index) } -> std::same_as<T>;
    { obj.has_field(name_idx) } -> std::convertible_to<bool>;
    { obj.has_element(el_index) } -> std::convertible_to<bool>;
    { obj.slang_type_layout() } -> std::convertible_to<slang::TypeLayoutReflection*>;
    { obj.is_valid() } -> std::convertible_to<bool>;
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
    { obj.template get<_AnyCursorValue>(val) } -> std::same_as<void>; // Ensure set() method exists
    { obj.template as<_AnyCursorValue>() } -> std::same_as<_AnyCursorValue>;
    { obj._get_array(data, size, scalar_type, element_count) } -> std::same_as<void>;
    { obj._get_scalar(data, size, scalar_type) } -> std::same_as<void>;
    { obj._get_vector(data, size, scalar_type, 0) } -> std::same_as<void>;
    { obj._get_matrix(data, size, scalar_type, 0, 0) } -> std::same_as<void>;
};

/// Concept that defines the requirements for a cursor that can be written to.
template<typename T>
concept WritableCursor
    = requires(T obj, void* data, size_t size, TypeReflection::ScalarType scalar_type, size_t element_count) {
          { obj.template set<_AnyCursorValue>({}) } -> std::same_as<void>;
          { obj.template operator= <_AnyCursorValue>({}) } -> std::same_as<void>;
          { obj._set_array(data, size, scalar_type, element_count) } -> std::same_as<void>;
          { obj._set_scalar(data, size, scalar_type) } -> std::same_as<void>;
          { obj._set_vector(data, size, scalar_type, 0) } -> std::same_as<void>;
          { obj._set_matrix(data, size, scalar_type, 0, 0) } -> std::same_as<void>;
      };

} // namespace sgl
