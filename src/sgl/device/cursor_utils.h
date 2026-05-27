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
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace sgl {

/// True when T can write itself to the exact cursor type that will be passed.
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

    /// Erased writer used by ShaderCursor to bind a registered native object.
    using ShaderCursorObjectWriteFunc = std::function<bool(ShaderCursor&, const void*)>;
    /// Erased writer used by BufferElementCursor to bind a registered native object.
    using BufferElementCursorObjectWriteFunc = std::function<bool(BufferElementCursor&, const void*)>;

    /// Records which class-owned signature hook a cursor writer uses.
    enum class CursorWriterSignatureKind {
        none,
        default_class_name,
        static_string,
        static_function,
        dynamic_function,
    };

    /// Native registry entry for one bindable value type.
    ///
    /// Low-level cursor writers may populate only one cursor callback while migrating existing code.
    /// Public register_cursor_writer<T>() entries always populate both callbacks and functional metadata.
    struct CursorWriterTypeInfo {
        /// Native C++ type exposed through nanobind.
        const std::type_info* type{nullptr};

        /// Write an object instance into a ShaderCursor, if this type supports it.
        ShaderCursorObjectWriteFunc write_shader_cursor;
        /// Write an object instance into a BufferElementCursor, if this type supports it.
        BufferElementCursorObjectWriteFunc write_buffer_cursor;

        /// True when this entry can also build SlangPy functional API marshalling.
        bool has_functional_metadata{false};
        /// Static Slang type name supplied by T::slang_type_name.
        std::string slang_type_name;
        /// Writes the cache signature for a concrete value instance.
        std::function<void(SignatureBuffer&, const void*)> write_signature;
        /// Debug/test aid describing which signature source was selected.
        CursorWriterSignatureKind signature_kind{CursorWriterSignatureKind::none};
        /// Static imports copied from T::slangpy_imports() at registration time.
        std::vector<std::string> imports;
    };

    /// Add or merge a type entry into the combined cursor-writer registry.
    ///
    /// Re-registering the same cursor callback or functional metadata is rejected. Separate
    /// one-sided legacy registrations for the same type are merged into one entry.
    SGL_API void register_cursor_writer_type(CursorWriterTypeInfo info);

    /// Return a read-only view of all registered cursor-writer type entries.
    SGL_API std::span<const CursorWriterTypeInfo> cursor_writer_type_infos();

    /// Find the exact native cursor-writer entry for a std::type_info, if one exists.
    SGL_API const CursorWriterTypeInfo* find_cursor_writer_type_info(const std::type_info& type);

    /// Migration wrapper for registering only a ShaderCursor writer.
    SGL_API void register_shader_cursor_object_writer(const std::type_info& type, ShaderCursorObjectWriteFunc write);
    /// Migration wrapper for registering only a BufferElementCursor writer.
    SGL_API void
    register_buffer_element_cursor_object_writer(const std::type_info& type, BufferElementCursorObjectWriteFunc write);

    /// Migration wrapper that erases a typed ShaderCursor writer into the combined registry.
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

    /// Migration wrapper that erases a typed BufferElementCursor writer into the combined registry.
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

    /// Migration wrapper for types whose ShaderCursor::set(T) implementation should be reused directly.
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

    /// Migration wrapper for types whose BufferElementCursor::set(T) implementation should be reused directly.
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

    namespace detail {

        /// Return the compiler-specific spelling that contains T's name.
        template<typename T>
        constexpr std::string_view wrapped_type_name()
        {
#if SGL_MSVC
            return __FUNCSIG__;
#else
            return __PRETTY_FUNCTION__;
#endif
        }

        /// Remove MSVC's "class " / "struct " prefix from a type name fragment.
        constexpr std::string_view strip_class_key(std::string_view name)
        {
            if (name.starts_with("class "))
                return name.substr(6);
            if (name.starts_with("struct "))
                return name.substr(7);
            return name;
        }

        /// Best-effort default signature name used when T provides no explicit signature.
        template<typename T>
        constexpr std::string_view cursor_writer_class_name()
        {
            constexpr std::string_view wrapped = wrapped_type_name<T>();
#if SGL_MSVC
            constexpr std::string_view marker = "wrapped_type_name<";
            size_t begin = wrapped.find(marker);
            if (begin == std::string_view::npos)
                return wrapped;
            begin += marker.size();
            size_t end = wrapped.rfind(">(void)");
            if (end == std::string_view::npos)
                end = wrapped.rfind('>');
#else
            constexpr std::string_view marker = "T = ";
            size_t begin = wrapped.find(marker);
            if (begin == std::string_view::npos)
                return wrapped;
            begin += marker.size();
            size_t end = wrapped.find(';', begin);
            if (end == std::string_view::npos)
                end = wrapped.find(']', begin);
#endif
            if (end == std::string_view::npos || end <= begin)
                return wrapped;
            return strip_class_key(wrapped.substr(begin, end - begin));
        }

    } // namespace detail

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

    /// True when T supplies a value-aware cache signature member function.
    template<typename T>
    concept HasValueSignature = requires {
        requires std::is_member_function_pointer_v<decltype(&T::write_slangpy_signature)>;
    } && requires(const T& value, SignatureBuffer& signature) {
        { value.write_slangpy_signature(signature) };
    };

    /// True when T supplies a static function cache signature.
    template<typename T>
    concept HasTypeStaticSignature = requires(SignatureBuffer& signature) {
        { T::write_slangpy_signature(signature) };
    };

    /// True when T supplies static import metadata copied at registration time.
    template<typename T>
    concept HasTypeStaticImports = requires {
        { T::slangpy_imports() };
    };

    /// Public cursor-writer registration contract.
    ///
    /// Registered types must provide class-owned Slang metadata and must write through both cursor kinds.
    template<typename T>
    concept CanRegisterCursorWriter = HasTypeStaticSlangTypeName<T> && HasWriteToCursor<T, ShaderCursor>
        && HasWriteToCursor<T, BufferElementCursor>;

    /// Select the signature source using the public contract's priority order.
    template<typename T>
    constexpr CursorWriterSignatureKind cursor_writer_signature_kind()
    {
        if constexpr (HasValueSignature<T>)
            return CursorWriterSignatureKind::dynamic_function;
        else if constexpr (HasTypeStaticSignature<T>)
            return CursorWriterSignatureKind::static_function;
        else if constexpr (HasTypeStaticStringSignature<T>)
            return CursorWriterSignatureKind::static_string;
        else
            return CursorWriterSignatureKind::default_class_name;
    }

    /// Write the SlangPy cache signature for a registered cursor-writer value.
    template<typename T>
    void write_cursor_writer_signature(SignatureBuffer& signature, const void* value)
        requires(HasTypeStaticSlangTypeName<T>)
    {
        if constexpr (HasValueSignature<T>)
            static_cast<const T*>(value)->write_slangpy_signature(signature);
        else if constexpr (HasTypeStaticSignature<T>)
            T::write_slangpy_signature(signature);
        else if constexpr (HasTypeStaticStringSignature<T>)
            signature.add(std::string_view(T::slangpy_signature));
        else
            signature.add(detail::cursor_writer_class_name<T>());
    }

    /// Copy an arbitrary iterable of string-like import paths into owned storage.
    template<typename TImports>
    void append_cursor_writer_imports(std::vector<std::string>& result, TImports&& imports)
    {
        for (auto&& import_path : imports)
            result.emplace_back(std::string_view(import_path));
    }

    /// Collect static imports for T once during registration.
    template<typename T>
    std::vector<std::string> cursor_writer_imports()
    {
        std::vector<std::string> result;
        if constexpr (HasTypeStaticImports<T>)
            append_cursor_writer_imports(result, T::slangpy_imports());
        return result;
    }

    /// Register T as a native bindable value for both direct cursor writes and the functional API.
    ///
    /// T owns all metadata: static slang_type_name is required, signature/import metadata is optional,
    /// and write_to_cursor must compile for both ShaderCursor and BufferElementCursor.
    template<typename T>
    void register_cursor_writer()
        requires(CanRegisterCursorWriter<T>)
    {
        CursorWriterTypeInfo info;
        info.type = &typeid(T);

        info.write_shader_cursor = [](ShaderCursor& cursor, const void* value)
        {
            static_cast<const T*>(value)->write_to_cursor(cursor);
            return true;
        };
        info.write_buffer_cursor = [](BufferElementCursor& cursor, const void* value)
        {
            static_cast<const T*>(value)->write_to_cursor(cursor);
            return true;
        };

        info.has_functional_metadata = true;
        info.slang_type_name = std::string(std::string_view(T::slang_type_name));
        info.write_signature = [](SignatureBuffer& signature, const void* value)
        {
            write_cursor_writer_signature<T>(signature, value);
        };
        info.signature_kind = cursor_writer_signature_kind<T>();
        info.imports = cursor_writer_imports<T>();

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
