// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device/cursor_writer.h"

#include "sgl/core/signature_buffer.h"
#include "sgl/device/buffer_cursor.h"
#include "sgl/device/shader_cursor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace sgl::slangpy {
namespace {

    const std::type_info& native_type_info(const cursor_utils::CursorWriterTypeInfo& info)
    {
        SGL_ASSERT(info.key_kind == cursor_utils::CursorWriterTypeKeyKind::native_type_info);
        SGL_ASSERT(info.type_key != nullptr);
        return *static_cast<const std::type_info*>(info.type_key);
    }

    // Return the underlying C++ pointer when a Python object is backed by the requested nanobind type.
    bool native_cursor_writer_pointer(const std::type_info& type, nb::handle obj, void*& value)
    {
        return nb::detail::nb_type_get(
            &type,
            obj.ptr(),
            static_cast<uint8_t>(nb::detail::cast_flags::manual),
            nullptr,
            &value
        );
    }

    bool python_type_isinstance(nb::handle obj, const cursor_utils::CursorWriterTypeInfo& info)
    {
        SGL_ASSERT(info.key_kind == cursor_utils::CursorWriterTypeKeyKind::python_type);
        auto* python_type = reinterpret_cast<PyTypeObject*>(const_cast<void*>(info.type_key));
        int result = PyObject_IsInstance(obj.ptr(), reinterpret_cast<PyObject*>(python_type));
        if (result < 0)
            nb::raise_python_error();
        return result != 0;
    }

    const cursor_utils::CursorWriterTypeInfo*
    find_cursor_writer_type_info_uncached(nb::handle obj, std::span<const cursor_utils::CursorWriterTypeInfo> infos)
    {
        nb::handle type = obj.type();
        PyTypeObject* python_type = Py_TYPE(obj.ptr());
        const std::type_info* exact_type = nb::type_check(type) ? &nb::type_info(type) : nullptr;

        if (exact_type) {
            if (const auto* info = cursor_utils::find_cursor_writer_type_info(*exact_type))
                return info;
        }

        for (const auto& info : infos) {
            if (info.key_kind != cursor_utils::CursorWriterTypeKeyKind::python_type)
                continue;
            if (info.type_key == python_type)
                return &info;
        }

        for (const auto& info : infos) {
            if (info.key_kind != cursor_utils::CursorWriterTypeKeyKind::native_type_info)
                continue;
            const std::type_info& info_type = native_type_info(info);
            if (exact_type && info_type == *exact_type)
                continue;
            if (!nb::detail::nb_type_isinstance(obj.ptr(), &info_type))
                continue;
            return &info;
        }

        for (const auto& info : infos) {
            if (info.key_kind != cursor_utils::CursorWriterTypeKeyKind::python_type)
                continue;
            if (!python_type_isinstance(obj, info))
                continue;
            return &info;
        }

        return nullptr;
    }

    std::string python_type_debug_name(nb::type_object python_type)
    {
        std::string module = nb::cast<std::string>(python_type.attr("__module__"));
        std::string qualname = nb::cast<std::string>(python_type.attr("__qualname__"));
        return module + "." + qualname;
    }

    nb::handle python_value_handle(const void* value)
    {
        return nb::handle(reinterpret_cast<PyObject*>(const_cast<void*>(value)));
    }

} // namespace

const cursor_utils::CursorWriterTypeInfo* find_cursor_writer_type_info(nb::handle obj)
{
    auto infos = cursor_utils::cursor_writer_type_infos();
    if (infos.empty())
        return nullptr;

    struct Cache {
        uint64_t registry_generation = 0;
        std::unordered_map<PyTypeObject*, const cursor_utils::CursorWriterTypeInfo*> entries;
    };
    static Cache cache;

    const uint64_t registry_generation = cursor_utils::cursor_writer_registry_generation();
    if (cache.registry_generation != registry_generation) {
        cache.entries.clear();
        cache.registry_generation = registry_generation;
    }

    // Cache by Python type, including nullptr misses, to avoid repeating the registered-base scan for
    // ordinary Python values on signature hot paths.
    PyTypeObject* python_type = Py_TYPE(obj.ptr());
    auto it = cache.entries.find(python_type);
    if (it != cache.entries.end())
        return it->second;

    const cursor_utils::CursorWriterTypeInfo* info = find_cursor_writer_type_info_uncached(obj, infos);
    cache.entries.emplace(python_type, info);
    return info;
}

std::optional<CursorWriterValue> find_cursor_writer(nb::handle obj)
{
    const cursor_utils::CursorWriterTypeInfo* info = find_cursor_writer_type_info(obj);
    if (!info)
        return std::nullopt;

    if (info->key_kind == cursor_utils::CursorWriterTypeKeyKind::python_type)
        return CursorWriterValue{info, obj.ptr()};

    void* value = nullptr;
    if (native_cursor_writer_pointer(native_type_info(*info), obj, value) && value != nullptr)
        return CursorWriterValue{info, value};

    return std::nullopt;
}

nb::object get_cursor_writer_type_info(nb::handle obj)
{
    auto writer = find_cursor_writer(obj);
    if (!writer || writer->info->slang_type_name.empty())
        return nb::none();

    SignatureBuffer signature;
    writer->info->write_signature(signature, writer->value);

    nb::list imports;
    for (const auto& import_path : writer->info->imports)
        imports.append(import_path);

    nb::dict result;
    result["slang_type_name"] = writer->info->slang_type_name;
    result["signature"] = std::string(signature.view());
    result["imports"] = nb::tuple(imports);
    return result;
}

void register_python_cursor_writer_type(
    nb::type_object python_type,
    PythonShaderCursorWriteFunc write_shader_cursor,
    PythonBufferElementCursorWriteFunc write_buffer_cursor,
    PythonCursorSignatureWriteFunc write_signature,
    std::string slang_type_name,
    std::vector<std::string> imports
)
{
    const std::string debug_name = python_type_debug_name(python_type);
    nb::object type_keep_alive(python_type);

    cursor_utils::CursorWriterTypeInfo info;
    info.key_kind = cursor_utils::CursorWriterTypeKeyKind::python_type;
    info.type_key = python_type.ptr();
    info.debug_name = debug_name;
    info.slang_type_name = std::move(slang_type_name);
    info.imports = std::move(imports);

    if (write_shader_cursor) {
        info.write_shader_cursor
            = [write_shader_cursor = std::move(write_shader_cursor)](ShaderCursor& cursor, const void* value) -> bool
        {
            return write_shader_cursor(cursor, python_value_handle(value));
        };
    }

    if (write_buffer_cursor) {
        info.write_buffer_cursor
            = [write_buffer_cursor
               = std::move(write_buffer_cursor)](BufferElementCursor& cursor, const void* value) -> bool
        {
            return write_buffer_cursor(cursor, python_value_handle(value));
        };
    }

    if (write_signature) {
        info.write_signature
            = [write_signature = std::move(write_signature), type_keep_alive](SignatureBuffer& out, const void* value)
        {
            (void)type_keep_alive;
            write_signature(out, python_value_handle(value));
        };
    }

    cursor_utils::register_cursor_writer_type(std::move(info));
}

void register_python_cursor_writer_type(
    nb::type_object python_type,
    nb::object write_shader_cursor,
    nb::object write_buffer_cursor,
    nb::object write_signature,
    std::string slang_type_name,
    std::vector<std::string> imports
)
{
    const std::string debug_name = python_type_debug_name(python_type);

    PythonShaderCursorWriteFunc write_shader_cursor_func;
    PythonBufferElementCursorWriteFunc write_buffer_cursor_func;
    PythonCursorSignatureWriteFunc write_signature_func;

    if (!write_shader_cursor.is_none()) {
        SGL_CHECK(
            PyCallable_Check(write_shader_cursor.ptr()),
            "Shader cursor writer for \"{}\" must be callable.",
            debug_name
        );
        write_shader_cursor_func
            = [callback = nb::object(write_shader_cursor)](ShaderCursor& cursor, nb::handle value) -> bool
        {
            // Python cursor-writer callbacks are invoked from Python-entered paths where the GIL is already held.
            callback(cursor, nb::borrow<nb::object>(value));
            return true;
        };
    }

    if (!write_buffer_cursor.is_none()) {
        SGL_CHECK(
            PyCallable_Check(write_buffer_cursor.ptr()),
            "Buffer cursor writer for \"{}\" must be callable.",
            debug_name
        );
        write_buffer_cursor_func
            = [callback = nb::object(write_buffer_cursor)](BufferElementCursor& cursor, nb::handle value) -> bool
        {
            // Python cursor-writer callbacks are invoked from Python-entered paths where the GIL is already held.
            callback(cursor, nb::borrow<nb::object>(value));
            return true;
        };
    }

    if (nb::isinstance<nb::str>(write_signature)) {
        std::string signature = nb::cast<std::string>(write_signature);
        write_signature_func = [signature = std::move(signature)](SignatureBuffer& out, nb::handle value)
        {
            (void)value;
            out.add(signature);
        };
    } else if (!write_signature.is_none()) {
        SGL_CHECK(PyCallable_Check(write_signature.ptr()), "Signature writer for \"{}\" must be callable.", debug_name);
        write_signature_func = [callback = nb::object(write_signature)](SignatureBuffer& out, nb::handle value)
        {
            // Python cursor-writer callbacks are invoked from Python-entered paths where the GIL is already held.
            nb::object result = callback(nb::borrow<nb::object>(value));
            if (!result.is_none())
                out.add(nb::cast<std::string>(result));
        };
    }

    register_python_cursor_writer_type(
        python_type,
        std::move(write_shader_cursor_func),
        std::move(write_buffer_cursor_func),
        std::move(write_signature_func),
        std::move(slang_type_name),
        std::move(imports)
    );
}

void unregister_python_cursor_writer_types()
{
    cursor_utils::unregister_cursor_writer_types(cursor_utils::CursorWriterTypeKeyKind::python_type);
}

} // namespace sgl::slangpy
