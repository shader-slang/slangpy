// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <initializer_list>
#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/buffer_cursor.h"

#include "utils/slangpybuffer.h"

#include <fmt/format.h>

namespace sgl {

extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);
extern void buffer_copy_from_numpy(Buffer* self, nb::ndarray<nb::numpy> data);

} // namespace sgl

namespace sgl::slangpy {

NativeNDBuffer::NativeNDBuffer(Device* device, NativeNDBufferDesc desc, ref<Buffer> storage)
    : StridedBufferView(device, desc, std::move(storage))
    , m_desc(desc)
{
}

ref<NativeNDBuffer> NativeNDBuffer::view(Shape shape, Shape strides, int offset) const
{
    auto result = make_ref<NativeNDBuffer>(device(), desc(), storage());
    result->view_inplace(shape, strides, offset);
    return result;
}
ref<NativeNDBuffer> NativeNDBuffer::broadcast_to(const Shape& shape) const
{
    auto result = make_ref<NativeNDBuffer>(device(), desc(), storage());
    result->broadcast_to_inplace(shape);
    return result;
}
ref<NativeNDBuffer> NativeNDBuffer::index(nb::object index_arg) const
{
    auto result = make_ref<NativeNDBuffer>(device(), desc(), storage());
    result->index_inplace(index_arg);
    return result;
}

std::string NativeNDBuffer::to_string() const
{
    return fmt::format(
        "NativeNDBuffer(\n"
        "  shape = {},\n"
        "  strides = {},\n"
        "  offset = {},\n"
        "  dtype = \"{}\",\n"
        "  memory_type = {},\n"
        "  usage = {}\n"
        ")",
        shape().to_string(),
        strides().to_string(),
        offset(),
        m_desc.dtype->to_string(),
        static_cast<int>(m_desc.memory_type),
        static_cast<int>(m_desc.usage)
    );
}

Shape NativeNDBufferMarshall::get_shape(nb::object data) const
{
    auto buffer = nb::cast<NativeNDBuffer*>(data);
    return buffer->shape();
}

void NativeNDBufferMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    SGL_UNUSED(read_back);

    // TODO: This function can be a lot more efficient and cleaner
    // optimize in next pass.

    // Cast value to buffer, and get the cursor field to write to.
    auto buffer = nb::cast<NativeNDBuffer*>(value);
    ShaderCursor field = cursor[binding->variable_name()];

    // Write the buffer storage.
    field["_data"] = buffer->storage();

    // Write shape vector as an array of ints (direct access, no temporary allocation).
    const Shape& shape = buffer->shape();
    field["_shape"]._set_array_unsafe(shape.data(), shape.size() * 4, shape.size(), TypeReflection::ScalarType::int32);

    // Generate and write strides vector, clearing strides to 0
    // for dimensions that are broadcast.
    const Shape& strides_shape = buffer->strides();
    const Shape& transform = binding->transform();
    const Shape& call_shape = context->call_shape();

    // Create a mutable copy of strides using Shape instead of std::vector
    Shape strides_copy = strides_shape;
    int* strides_data = strides_copy.data();
    for (size_t i = 0; i < transform.size(); i++) {
        int csidx = transform[i];
        if (call_shape[csidx] != shape[i]) {
            strides_data[i] = 0;
        }
    }

    // Write the strides vector as an array of ints.
    field["_strides"]._set_array_unsafe(
        strides_copy.data(),
        strides_copy.size() * 4,
        strides_copy.size(),
        TypeReflection::ScalarType::int32
    );
    field["_offset"] = buffer->offset();
}

void NativeNDBufferMarshall::read_calldata(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    nb::object data,
    nb::object result
) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);
    SGL_UNUSED(data);
    SGL_UNUSED(result);
}

nb::object NativeNDBufferMarshall::create_output(CallContext* context, NativeBoundVariableRuntime* binding) const
{
    SGL_UNUSED(binding);
    auto buffer = create_buffer(context->device(), context->call_shape());
    return nb::cast(buffer);
}

ref<NativeNDBuffer> NativeNDBufferMarshall::create_buffer(Device* device, const Shape& shape) const
{
    NativeNDBufferDesc desc;
    desc.dtype = m_slang_element_type;
    desc.element_layout = m_element_layout;
    desc.offset = 0;
    desc.shape = shape;
    desc.strides = desc.shape.calc_contiguous_strides();
    desc.usage = BufferUsage::shader_resource | BufferUsage::unordered_access;
    desc.memory_type = MemoryType::device_local;
    return make_ref<NativeNDBuffer>(device, desc);
}

nb::object NativeNDBufferMarshall::create_dispatchdata(nb::object data) const
{
    // Cast value to buffer, and get the cursor field to write to.
    auto buffer = nb::cast<NativeNDBuffer*>(data);
    nb::dict res;
    res["_data"] = buffer->storage();
    res["_shape"] = shape_to_list(buffer->shape());
    res["_offset"] = buffer->offset();
    res["_strides"] = shape_to_list(buffer->strides());
    return res;
}

nb::object
NativeNDBufferMarshall::read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);
    return data;
}

Shape NativeNumpyMarshall::get_shape(nb::object data) const
{
    auto ndarray = nb::cast<nb::ndarray<nb::numpy>>(data);
    const size_t ndim = ndarray.ndim();
    Shape shape(ndim);
    int* shape_data = shape.data();
    for (size_t i = 0; i < ndim; i++) {
        shape_data[i] = static_cast<int>(ndarray.shape(i));
    }
    return shape;
}

void NativeNumpyMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    auto ndarray = nb::cast<nb::ndarray<nb::numpy>>(value);
    SGL_CHECK(ndarray.dtype() == m_dtype, "numpy array dtype does not match the expected dtype");

    const size_t ndim = ndarray.ndim();
    Shape shape(ndim);
    int* shape_data = shape.data();
    for (size_t i = 0; i < ndim; i++) {
        shape_data[i] = static_cast<int>(ndarray.shape(i));
    }

    const Shape& vector_shape = binding->vector_type()->shape();
    for (size_t i = 0; i < vector_shape.size(); i++) {
        int vs_size = vector_shape[vector_shape.size() - i - 1];
        int arr_size = shape[shape.size() - i - 1];

        SGL_CHECK(
            vs_size == arr_size,
            "numpy array shape dim {} does not match the expected shape ({} != {})",
            shape.size() - i - 1,
            vs_size,
            arr_size
        );
    }

    SGL_UNUSED(binding);
    auto buffer = create_buffer(context->device(), shape);
    buffer_copy_from_numpy(buffer->storage().get(), ndarray);

    auto buffer_obj = nb::cast(buffer);
    store_readback(binding, read_back, value, buffer_obj);

    NativeNDBufferMarshall::write_shader_cursor_pre_dispatch(context, binding, cursor, buffer_obj, read_back);
}

void NativeNumpyMarshall::read_calldata(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    nb::object data,
    nb::object result
) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);

    auto ndarray = nb::cast<nb::ndarray<nb::numpy>>(data);
    auto buffer = nb::cast<NativeNDBuffer*>(result);

    size_t numpy_data_size = ndarray.nbytes();
    size_t buffer_data_size = buffer->storage()->size();
    SGL_CHECK(
        numpy_data_size == buffer_data_size,
        "numpy array size does not match the buffer ({} > {})",
        numpy_data_size,
        buffer_data_size
    );

    buffer->storage()->get_data(ndarray.data(), buffer_data_size);
}

nb::object NativeNumpyMarshall::create_output(CallContext* context, NativeBoundVariableRuntime* binding) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);

    Shape shape = context->call_shape() + binding->vector_type()->shape();

    size_t data_size = element_stride() * shape.element_count();
    void* data = new uint8_t[data_size];
    nb::capsule owner(
        data,
        [](void* p) noexcept
        {
            delete[] reinterpret_cast<uint8_t*>(p);
        }
    );

    std::vector<size_t> s;
    s.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        s.push_back(shape[i]);
    }

    auto ndarray = nb::ndarray<nb::numpy>(data, s.size(), s.data(), owner, nullptr, m_dtype);
    return nb::cast(ndarray);
}

nb::object NativeNumpyMarshall::create_dispatchdata(nb::object data) const
{
    SGL_UNUSED(data);
    SGL_THROW("Raw dispatch is not supported for numpy arrays.");
}


} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy_buffer)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<NativeNDBufferDesc, StridedBufferViewDesc>(slangpy, "NativeNDBufferDesc").def(nb::init<>());

    nb::class_<NativeNDBuffer, StridedBufferView>(slangpy, "NativeNDBuffer")
        .def(nb::init<ref<Device>, NativeNDBufferDesc, ref<Buffer>>(), "device"_a, "desc"_a, "buffer"_a = nullptr)
        .def("broadcast_to", &NativeNDBuffer::broadcast_to, "shape"_a)
        .def("view", &NativeNDBuffer::view, "shape"_a, "strides"_a = Shape(), "offset"_a = 0)
        .def("__getitem__", &NativeNDBuffer::index)
        .def("__repr__", &NativeNDBuffer::to_string);


    nb::class_<NativeNDBufferMarshall, NativeMarshall>(slangpy, "NativeNDBufferMarshall") //
        .def(
            "__init__",
            [](NativeNDBufferMarshall& self,
               int dims,
               bool writable,
               ref<NativeSlangType> slang_type,
               ref<NativeSlangType> slang_element_type,
               ref<TypeLayoutReflection> element_layout)
            {
                new (&self) NativeNDBufferMarshall(dims, writable, slang_type, slang_element_type, element_layout);
            },
            "dims"_a,
            "writable"_a,
            "slang_type"_a,
            "slang_element_type"_a,
            "element_layout"_a,
            D_NA(NativeNDBufferMarshall, NativeNDBufferMarshall)
        )
        .def_prop_ro("dims", &sgl::slangpy::NativeNDBufferMarshall::dims)
        .def_prop_ro("writable", &sgl::slangpy::NativeNDBufferMarshall::writable)
        .def_prop_ro("slang_element_type", &sgl::slangpy::NativeNDBufferMarshall::slang_element_type);

    nb::class_<NativeNumpyMarshall, NativeNDBufferMarshall>(slangpy, "NativeNumpyMarshall") //
        .def(
            "__init__",
            [](NativeNumpyMarshall& self,
               int dims,
               ref<NativeSlangType> slang_type,
               ref<NativeSlangType> slang_element_type,
               ref<TypeLayoutReflection> element_layout,
               nb::object numpydtype)
            {
                nb::dlpack::dtype dtype;

                int bytes = nb::cast<int>(numpydtype.attr("itemsize"));
                char kind = nb::cast<char>(numpydtype.attr("kind"));
                dtype.bits = (uint8_t)bytes * 8;
                dtype.lanes = 1;
                switch (kind) {
                case 'i':
                    dtype.code = (uint8_t)nb::dlpack::dtype_code::Int;
                    break;
                case 'u':
                    dtype.code = (uint8_t)nb::dlpack::dtype_code::UInt;
                    break;
                case 'f':
                    dtype.code = (uint8_t)nb::dlpack::dtype_code::Float;
                    break;
                case 'b':
                    dtype.code = (uint8_t)nb::dlpack::dtype_code::Bool;
                    break;
                default:
                    SGL_THROW("Unsupported numpy dtype kind '{}'", kind);
                    break;
                }
                new (&self) NativeNumpyMarshall(dims, slang_type, slang_element_type, element_layout, dtype);
            },
            "dims"_a,
            "slang_type"_a,
            "slang_element_type"_a,
            "element_layout"_a,
            "numpydtype"_a,
            D_NA(NativeNumpyMarshall, NativeNumpyMarshall)
        )
        .def_prop_ro("dtype", &sgl::slangpy::NativeNumpyMarshall::dtype);
}
