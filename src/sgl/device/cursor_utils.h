// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/reflection.h"

#include "sgl/core/macros.h"
#include "sgl/core/signature_buffer.h"
#include "sgl/core/type_utils.h"

#include <concepts>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sgl {

namespace detail {
    template<typename T>
    struct CursorWriterOwner {
        using type = std::remove_cvref_t<T>;
    };
} // namespace detail

/// True when T has a class-owned cursor writer for the exact cursor and nullable value pointer.
template<typename T, typename TCursor>
concept HasWriteToCursor = requires(const TCursor& cursor, const typename detail::CursorWriterOwner<T>::type* value) {
    { detail::CursorWriterOwner<T>::type::write_to_cursor(cursor, value) };
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

    /// Erased writer used by ShaderCursor to bind a registered native object.
    using ShaderCursorObjectWriteFunc = std::function<bool(ShaderCursor&, const void*)>;
    /// Erased writer used by BufferElementCursor to bind a registered native object.
    using BufferElementCursorObjectWriteFunc = std::function<bool(BufferElementCursor&, const void*)>;

    /// Native registry entry for one cursor-writable value type.
    ///
    /// The SlangPy cache signature is required, while the simple WriteToCursorMarshall fallback type name is optional.
    /// This lets resource types provide native signatures and direct cursor writes while still using bespoke functional
    /// API marshalls.
    struct CursorWriterTypeInfo {
        /// Native C++ type exposed through nanobind.
        const std::type_info* type{nullptr};

        /// Write an object instance into a ShaderCursor.
        ShaderCursorObjectWriteFunc write_shader_cursor;
        /// Write an object instance into a BufferElementCursor.
        BufferElementCursorObjectWriteFunc write_buffer_cursor;

        /// Static Slang type name supplied by T::slang_type_name for the simple functional fallback.
        std::string slang_type_name;
        /// Writes the cache signature for a concrete value instance when native signature metadata is available.
        std::function<void(SignatureBuffer&, const void*)> write_signature;
        /// Static imports copied from T::slangpy_imports() at registration time for the simple functional fallback.
        std::vector<std::string> imports;
    };

    /// Add a type entry to the cursor-writer registry.
    /// Duplicate registrations for the same native type are rejected.
    SGL_API void register_cursor_writer_type(CursorWriterTypeInfo info);

    /// Return a read-only view of all registered cursor-writer type entries.
    SGL_API std::span<const CursorWriterTypeInfo> cursor_writer_type_infos();

    /// Find the exact native cursor-writer entry for a std::type_info, if one exists.
    SGL_API const CursorWriterTypeInfo* find_cursor_writer_type_info(const std::type_info& type);

    /// Register cursor writers for built-in SGL value types.
    SGL_API void register_cursor_writers();

    template<typename TCursor, typename T>
        requires(HasWriteToCursor<T, TCursor>)
    void write_to_cursor(const TCursor& cursor, const T* value)
    {
        detail::CursorWriterOwner<T>::type::write_to_cursor(cursor, value);
    }

    /// True when T supplies the required static Slang type name metadata.
    template<typename T>
    concept HasTypeStaticSlangTypeName = requires {
        { std::string_view(T::slang_type_name) } -> std::same_as<std::string_view>;
    };

    /// True when T supplies a static string cache signature.
    template<typename T>
    concept HasTypeStaticStringSignature = requires {
        { std::string_view(T::slangpy_signature) } -> std::same_as<std::string_view>;
    };

    /// True when T supplies a pointer-aware static cache signature function.
    template<typename T>
    concept HasStaticValueSignature = requires(SignatureBuffer& signature, const T* value) {
        { T::write_slangpy_signature(signature, value) };
    };

    /// True when T supplies static import metadata copied at registration time.
    template<typename T>
    concept HasTypeStaticImports = requires {
        { T::slangpy_imports() };
    };

    /// Public direct cursor-writer registration contract.
    template<typename T>
    concept CanRegisterCursorWriter = HasWriteToCursor<T, ShaderCursor> || HasWriteToCursor<T, BufferElementCursor>;

    /// True when T can also be used as a simple functional API fallback marshall.
    template<typename T>
    concept CanRegisterFunctionalCursorWriter = CanRegisterCursorWriter<T> && HasTypeStaticSlangTypeName<T>;

    /// Write the SlangPy cache signature for a registered cursor-writer value.
    template<typename T>
    void write_cursor_writer_signature(SignatureBuffer& signature, const void* value)
    {
        if constexpr (HasStaticValueSignature<T>)
            T::write_slangpy_signature(signature, static_cast<const T*>(value));
        else if constexpr (HasTypeStaticStringSignature<T>)
            signature.add(std::string_view(T::slangpy_signature));
        else
            signature.add(detail::type_name<T>());
    }

    /// Register T as a native bindable value for direct cursor writes.
    ///
    /// If T also owns static slang_type_name metadata, it is used as a simple functional API fallback.
    /// Types with bespoke marshalls should omit that metadata and keep the Python/native marshall as owner.
    template<typename T>
    void register_cursor_writer()
        requires(CanRegisterCursorWriter<T>)
    {
        CursorWriterTypeInfo info;
        info.type = &typeid(T);

        if constexpr (HasWriteToCursor<T, ShaderCursor>) {
            info.write_shader_cursor = [](ShaderCursor& cursor, const void* value)
            {
                write_to_cursor(cursor, static_cast<const T*>(value));
                return true;
            };
        }
        if constexpr (HasWriteToCursor<T, BufferElementCursor>) {
            info.write_buffer_cursor = [](BufferElementCursor& cursor, const void* value)
            {
                write_to_cursor(cursor, static_cast<const T*>(value));
                return true;
            };
        }
        info.write_signature = [](SignatureBuffer& signature, const void* value)
        {
            write_cursor_writer_signature<T>(signature, value);
        };

        if constexpr (HasTypeStaticSlangTypeName<T>) {
            info.slang_type_name = std::string(std::string_view(T::slang_type_name));
            if constexpr (HasTypeStaticImports<T>) {
                for (auto&& import_path : T::slangpy_imports())
                    info.imports.emplace_back(std::string_view(import_path));
            }
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
