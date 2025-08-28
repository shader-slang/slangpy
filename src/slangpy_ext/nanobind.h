// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>

#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/core/type_utils.h"
#include "sgl/core/data_struct.h"
#include "sgl/core/data_type.h"
#include "sgl/core/static_vector.h"

#include "sgl/math/float16.h"

#include "sgl/device/cuda_interop.h"

#include "py_doc.h"

#include <span>

namespace nb = nanobind;
using namespace nb::literals;

NAMESPACE_BEGIN(NB_NAMESPACE)

template<>
struct ndarray_traits<sgl::math::float16_t> {
    static constexpr bool is_complex = false;
    static constexpr bool is_float = true;
    static constexpr bool is_bool = false;
    static constexpr bool is_int = false;
    static constexpr bool is_signed = true;
};

NAMESPACE_BEGIN(detail)

/// Type caster for sgl::ref<T>
template<typename T>
struct type_caster<sgl::ref<T>> {
    using Caster = make_caster<T>;
    static constexpr bool IsClass = true;
    NB_TYPE_CASTER(sgl::ref<T>, Caster::Name);

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
    {
        Caster caster;
        if (!caster.from_python(src, flags, cleanup))
            return false;

        value = Value(caster.operator T*());
        return true;
    }

    static handle from_cpp(const sgl::ref<T>& value, rv_policy policy, cleanup_list* cleanup) noexcept
    {
        return Caster::from_cpp(value.get(), policy, cleanup);
    }
};

/// Type caster for std::span<T>
template<typename T>
struct type_caster<std::span<T>> {
    NB_TYPE_CASTER(
        std::span<T>,
        io_name("Sequence", NB_TYPING_LIST) + const_name("[") + make_caster<T>::Name + const_name("]")
    )

    using Caster = make_caster<T>;

    std::vector<T> vec;

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
    {
        size_t size;
        PyObject* temp;
        PyObject** o = seq_get(src.ptr(), &size, &temp);

        vec.clear();
        vec.reserve(size);

        Caster caster;
        bool success = o != nullptr;

        flags = flags_for_local_caster<T>(flags);

        for (size_t i = 0; i < size; ++i) {
            if (!caster.from_python(o[i], flags, cleanup) || !caster.template can_cast<T>()) {
                success = false;
                break;
            }

            vec.push_back(caster.operator cast_t<T>());
        }

        Py_XDECREF(temp);

        value = std::span{vec};

        return success;
    }

    static handle from_cpp(const std::span<T>& src, rv_policy policy, cleanup_list* cleanup) noexcept
    {
        object ret = steal(PyList_New(src.size()));

        if (ret.is_valid()) {
            Py_ssize_t index = 0;

            for (auto&& value : src) {
                handle h = Caster::from_cpp(forward_like_<T>(value), policy, cleanup);
                if (!h.is_valid()) {
                    ret.reset();
                    break;
                }
                NB_LIST_SET_ITEM(ret.ptr(), index++, h.ptr());
            }
        }

        return ret.release();
    }
};

template<typename T, std::size_t N>
struct type_caster<sgl::static_vector<T, N>> : list_caster<sgl::static_vector<T, N>, T> { };


NAMESPACE_END(detail)

template<typename T>
class sgl_enum : public enum_<T> {
public:
    static_assert(::sgl::has_enum_info<T>, "nanobind::sgl_enum<> requires an enumeration type with infos!");

    using Base = enum_<T>;

    template<typename... Extra>
    NB_INLINE sgl_enum(handle scope, const char* name, const Extra&... extra)
        : Base(scope, name, extra...)
    {
        for (const auto& item : ::sgl::EnumInfo<T>::items)
            Base::value(item.second, item.first);
    }
};

template<typename T>
class sgl_enum_flags : public sgl_enum<T> {
public:
    static_assert(
        std::is_same_v<T, decltype(std::declval<T>() & std::declval<T>())>,
        "nanobind::sgl_enum_flags<> requires an enumeration type with bitwise operators!"
    );

    using Base = sgl_enum<T>;

    template<typename... Extra>
    NB_INLINE sgl_enum_flags(handle scope, const char* name, const Extra&... extra)
        : Base(scope, name, nb::is_arithmetic(), nb::is_flag(), extra...)
    {
    }
};


NAMESPACE_END(NB_NAMESPACE)


namespace sgl {

/// Helper class passed to GcHelper::traverse() to visit attributes and child objects.
class GcVisitor {
public:
    /// Visit an attribute with the given name.
    void operator()(const char* name)
    {
        if (result != 0)
            return;
        if (!nb::hasattr(self, name))
            return;
        nb::object value = nb::getattr(self, name);
        if (value.is_valid() && !value.is_none()) {
            result = [&]()
            {
                Py_VISIT(value.ptr());
                return 0;
            }();
        }
    }
    /// Visit a child object.
    void operator()(Object* object)
    {
        if (result != 0)
            return;
        result = [&]()
        {
            Py_VISIT(object->self_py());
            return 0;
        }();
    }

private:
    PyObject* self;
    visitproc visit;
    void* arg;
    int result{0};
    template<typename T>
    friend class GcHelperFunctions;
};

/// Garbage collection helper to be specialized for each object type.
/// The default implementation does nothing.
template<typename T>
struct GcHelper {
    void traverse(T*, GcVisitor&) { }
    void clear(T*) { }
};

/// Implementation of tp_traverse and tp_clear for a given object type.
template<typename T>
class GcHelperFunctions {
public:
    static int tp_traverse(PyObject* self, visitproc visit, void* arg)
    {
        T* object = nb::inst_ptr<T>(self);
        GcVisitor visitor;
        visitor.self = self;
        visitor.visit = visit;
        visitor.arg = arg;
        GcHelper<T>{}.traverse(object, visitor);
        return visitor.result;
    }

    static int tp_clear(PyObject* self)
    {
        T* object = nb::inst_ptr<T>(self);
        GcHelper<T>{}.clear(object);
        return 0;
    }

public:
    static inline PyType_Slot type_slots[] = {
        {Py_tp_traverse, (void*)tp_traverse},
        {Py_tp_clear, (void*)tp_clear},
        {0, nullptr},
    };
};

/// Return Python type slots (tp_traverse, tp_clear) for a given object type.
template<typename T>
nb::type_slots gc_helper_type_slots()
{
    return nb::type_slots(GcHelperFunctions<T>::type_slots);
}

}; // namespace sgl

namespace sgl {

/// Check if ndarray is contiguous from the `first_dimension` (inclusive) onwards
template<typename... Args>
size_t is_ndarray_partially_contiguous(const nb::ndarray<Args...>& array, size_t first_dimension = 0)
{
    if (array.ndim() <= first_dimension)
        return false;
    size_t prod = 1;
    for (size_t i = array.ndim() - 1;;) {
        if (array.stride(i) != narrow_cast<int64_t>(prod))
            return false;
        prod *= array.shape(i);
        if (i == first_dimension)
            break;
        --i;
    }
    return true;
}


template<typename... Args>
size_t is_ndarray_contiguous(const nb::ndarray<Args...>& array)
{
    return is_ndarray_partially_contiguous(array, 0);
}

inline cuda::TensorView ndarray_to_cuda_tensor_view(nb::ndarray<nb::device::cuda> array)
{
    return cuda::TensorView{
        .device_id = array.device_id(),
        .data = array.data(),
        .size = array.nbytes(),
        .stride = 0, // TODO
    };
}

inline nb::dlpack::dtype data_type_to_dtype(DataType type)
{
    switch (type) {
    case DataType::bool_:
        return nb::dtype<bool>();
    case DataType::int8:
        return nb::dtype<int8_t>();
    case DataType::int16:
        return nb::dtype<int16_t>();
    case DataType::int32:
        return nb::dtype<int32_t>();
    case DataType::int64:
        return nb::dtype<int64_t>();
    case DataType::uint8:
        return nb::dtype<uint8_t>();
    case DataType::uint16:
        return nb::dtype<uint16_t>();
    case DataType::uint32:
        return nb::dtype<uint32_t>();
    case DataType::uint64:
        return nb::dtype<uint64_t>();
    case DataType::float16:
        return nb::dtype<math::float16_t>();
    case DataType::float32:
        return nb::dtype<float>();
    case DataType::float64:
        return nb::dtype<double>();
    default:
        break;
    }
    SGL_THROW("Data type is incompatible with DLPack.");
}

inline DataType dtype_to_data_type(nb::dlpack::dtype dtype)
{
    switch ((nb::dlpack::dtype_code)dtype.code) {
    case nb::dlpack::dtype_code::Float:
        switch (dtype.bits) {
        case 16:
            return DataType::float16;
        case 32:
            return DataType::float32;
        case 64:
            return DataType::float64;
        default:
            break;
        }
        break;

    case nb::dlpack::dtype_code::Int:
        switch (dtype.bits) {
        case 8:
            return DataType::int8;
        case 16:
            return DataType::int16;
        case 32:
            return DataType::int32;
        case 64:
            return DataType::int64;
        default:
            break;
        }
        break;

    case nb::dlpack::dtype_code::UInt:
        switch (dtype.bits) {
        case 8:
            return DataType::uint8;
        case 16:
            return DataType::uint16;
        case 32:
            return DataType::uint32;
        case 64:
            return DataType::uint64;
        default:
            break;
        }
        break;

    case nb::dlpack::dtype_code::Bool:
        switch (dtype.bits) {
        case 8:
            return DataType::bool_;
        default:
            break;
        }
        break;

    default:
        break;
    }
    SGL_THROW("Unsupported dtype.");
}

inline std::optional<DataStruct::Type> dtype_to_data_struct_type(nb::dlpack::dtype dtype)
{
    switch (dtype.code) {
    case uint8_t(nb::dlpack::dtype_code::Int):
        switch (dtype.bits) {
        case 8:
            return DataStruct::Type::int8;
        case 16:
            return DataStruct::Type::int16;
        case 32:
            return DataStruct::Type::int32;
        case 64:
            return DataStruct::Type::int64;
        }
        break;
    case uint8_t(nb::dlpack::dtype_code::UInt):
        switch (dtype.bits) {
        case 8:
            return DataStruct::Type::uint8;
        case 16:
            return DataStruct::Type::uint16;
        case 32:
            return DataStruct::Type::uint32;
        case 64:
            return DataStruct::Type::uint64;
        }
        break;
    case uint8_t(nb::dlpack::dtype_code::Float):
        switch (dtype.bits) {
        case 16:
            return DataStruct::Type::float16;
        case 32:
            return DataStruct::Type::float32;
        case 64:
            return DataStruct::Type::float64;
        }
        break;
    }
    return {};
}

} // namespace sgl


namespace sgl::detail {

template<typename TLen>
inline Py_ssize_t sanitize_getitem_index(Py_ssize_t index, TLen len_)
{
    Py_ssize_t len = Py_ssize_t(len_);
    // throwing index_error allows this to be used as a python iterator
    if (index >= len || index < -len)
        throw nb::index_error();

    return (index >= 0) ? index : (index + len);
}

inline constexpr uint64_t const_hash(std::string_view str)
{
    uint64_t hash = 0xcbf29ce484222325;
    constexpr size_t prime = 0x00000100000001b3;
    for (auto c : str) {
        hash ^= static_cast<size_t>(c);
        hash *= prime;
    }
    return hash;
}
} // namespace sgl::detail

#define SGL_DICT_TO_DESC_BEGIN(type)                                                                                   \
    type dict_to_##type(nb::dict dict)                                                                                 \
    {                                                                                                                  \
        type desc = {};                                                                                                \
        for (const auto& [k, v] : dict) {                                                                              \
            std::string_view key = nb::cast<std::string_view>(k);                                                      \
            uint64_t hash = ::sgl::detail::const_hash(key);                                                            \
            switch (hash) {

#define SGL_DICT_TO_DESC_FIELD(name, type)                                                                             \
    case ::sgl::detail::const_hash(#name):                                                                             \
        desc.name = nb::cast<type>(v);                                                                                 \
        break;

#define SGL_DICT_TO_DESC_FIELD_LIST(name, type)                                                                        \
    case ::sgl::detail::const_hash(#name):                                                                             \
        desc.name = {};                                                                                                \
        for (const auto& item : v)                                                                                     \
            desc.name.push_back(nb::cast<type>(item));                                                                 \
        break;

#define SGL_DICT_TO_DESC_FIELD_CUSTOM(name, code)                                                                      \
    case ::sgl::detail::const_hash(#name):                                                                             \
        desc.name = code;                                                                                              \
        break;

#define SGL_DICT_TO_DESC_END()                                                                                         \
    default:                                                                                                           \
        SGL_THROW("Unknown key {}", key);                                                                              \
        }                                                                                                              \
        }                                                                                                              \
        return desc;                                                                                                   \
        }


#define SGL_PY_DECLARE(name) extern void sgl_python_export_##name(nb::module_& m)
#define SGL_PY_EXPORT(name) void sgl_python_export_##name([[maybe_unused]] ::nb::module_& m)
#define SGL_PY_IMPORT(name) sgl_python_export_##name(m)

#define D(...) DOC(sgl, __VA_ARGS__)
#define D_NA(...) "N/A"
