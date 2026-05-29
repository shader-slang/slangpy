// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device/cursor_writer.h"

#include "sgl/core/signature_buffer.h"

#include <cstdint>
#include <unordered_map>

namespace sgl::slangpy {
namespace {

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

    const cursor_utils::CursorWriterTypeInfo* find_native_cursor_writer_type_info_uncached(
        nb::handle obj,
        std::span<const cursor_utils::CursorWriterTypeInfo> infos
    )
    {
        nb::handle type = obj.type();
        const std::type_info* exact_type = nb::type_check(type) ? &nb::type_info(type) : nullptr;

        if (exact_type) {
            if (const auto* info = cursor_utils::find_cursor_writer_type_info(*exact_type))
                return info;
        }

        for (const auto& info : infos) {
            if (exact_type && *info.type == *exact_type)
                continue;
            if (!nb::detail::nb_type_isinstance(obj.ptr(), info.type))
                continue;
            return &info;
        }

        return nullptr;
    }

} // namespace

const cursor_utils::CursorWriterTypeInfo* find_native_cursor_writer_type_info(nb::handle obj)
{
    auto infos = cursor_utils::cursor_writer_type_infos();
    if (infos.empty())
        return nullptr;

    struct Cache {
        size_t registry_size = 0;
        std::unordered_map<PyTypeObject*, const cursor_utils::CursorWriterTypeInfo*> entries;
    };
    static Cache cache;

    if (cache.registry_size != infos.size()) {
        cache.entries.clear();
        cache.registry_size = infos.size();
    }

    // Cache by Python type, including nullptr misses, to avoid repeating the registered-base scan for
    // ordinary Python values on signature hot paths.
    PyTypeObject* python_type = Py_TYPE(obj.ptr());
    auto it = cache.entries.find(python_type);
    if (it != cache.entries.end())
        return it->second;

    const cursor_utils::CursorWriterTypeInfo* info = find_native_cursor_writer_type_info_uncached(obj, infos);
    cache.entries.emplace(python_type, info);
    return info;
}

std::optional<NativeCursorWriterValue> find_native_cursor_writer(nb::handle obj)
{
    const cursor_utils::CursorWriterTypeInfo* info = find_native_cursor_writer_type_info(obj);
    if (!info)
        return std::nullopt;

    void* value = nullptr;
    if (native_cursor_writer_pointer(*info->type, obj, value) && value != nullptr)
        return NativeCursorWriterValue{info, value};

    return std::nullopt;
}

nb::object get_native_cursor_writer_type_info(nb::handle obj)
{
    auto writer = find_native_cursor_writer(obj);
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

} // namespace sgl::slangpy
