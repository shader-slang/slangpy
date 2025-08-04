// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <initializer_list>
#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/buffer_cursor.h"

#include "utils/slangpytensor.h"

namespace sgl {

extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);

} // namespace sgl

namespace sgl::slangpy {

NativeTensor::NativeTensor(
    NativeTensorDesc desc,
    const ref<Buffer>& storage,
    const ref<NativeTensor>& grad_in,
    const ref<NativeTensor>& grad_out
)
    : StridedBufferView(storage->device(), desc, storage)
    , m_desc(desc)
    , m_grad_in(grad_in)
    , m_grad_out(grad_out)
{
}

ref<NativeTensor> NativeTensor::view(Shape shape, Shape strides, int offset) const
{
    auto result = make_ref<NativeTensor>(desc(), storage(), m_grad_in, m_grad_out);
    result->view_inplace(shape, strides, offset);
    return result;
}
ref<NativeTensor> NativeTensor::broadcast_to(const Shape& shape) const
{
    auto result = make_ref<NativeTensor>(desc(), storage(), m_grad_in, m_grad_out);
    result->broadcast_to_inplace(shape);
    return result;
}
ref<NativeTensor> NativeTensor::index(nb::args args) const
{
    auto result = make_ref<NativeTensor>(desc(), storage(), m_grad_in, m_grad_out);
    result->index_inplace(args);
    return result;
}

ref<NativeTensor> NativeTensor::with_grads(ref<NativeTensor> grad_in, ref<NativeTensor> grad_out, bool zero) const
{
    ref<NativeTensor> new_grad_in = std::move(grad_in);
    ref<NativeTensor> new_grad_out = std::move(grad_out);

    // Create new, empty tensor for grads if none specified.
    if (!new_grad_in && !new_grad_out) {

        // Get the derivative type + buffer layout.
        ref<NativeSlangType> dtype = m_desc.dtype->derivative();
        if (!dtype)
            SGL_THROW("No derivative type found for {}", m_desc.dtype->get_type_reflection()->name());
        ref<TypeLayoutReflection> layout = dtype->buffer_type_layout();

        // Create a new structured buffer for storage.
        BufferDesc buffer_desc;
        buffer_desc.usage = BufferUsage::shader_resource | BufferUsage::unordered_access | BufferUsage::shared;
        buffer_desc.struct_size = layout->stride();
        buffer_desc.element_count = element_count();
        ref<Buffer> buffer = device()->create_buffer(buffer_desc);

        NativeTensorDesc desc;
        desc.dtype = dtype;
        desc.element_layout = layout;
        desc.shape = m_desc.shape;
        desc.strides = m_desc.shape.calc_contiguous_strides();
        desc.offset = m_desc.offset;

        // Create new native tensor to hold the grads.
        new_grad_in = make_ref<NativeTensor>(desc, buffer, nullptr, nullptr);
        new_grad_out = new_grad_in;
    }

    // Create a new tensor object that refers to the same data as this one, but with
    // associated grads.
    ref<NativeTensor> result = make_ref<NativeTensor>(m_desc, storage(), new_grad_in, new_grad_out);

    // Optionally clear both.
    if (zero) {
        if (new_grad_in) {
            new_grad_in->clear();
        }
        if (new_grad_out && new_grad_out != new_grad_in) {
            new_grad_out->clear();
        }
    }

    return result;
}

ref<NativeTensor> NativeTensor::detach() const
{
    // Create a new tensor object that refers to the same data as this one, but without
    // associated grads.
    return make_ref<NativeTensor>(m_desc, storage(), nullptr, nullptr);
}

Shape NativeTensorMarshall::get_shape(nb::object data) const
{
    auto buffer = nb::cast<NativeTensor*>(data);
    return buffer->shape();
}

void NativeTensorMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    // The native tensor marshall can be inherited for other types, so
    // we check if we have a native tensor, and if not revert to the
    // base class implementation.
    NativeTensor* primal;
    if (nb::try_cast(value, primal)) {
        ShaderCursor field = cursor[binding->get_variable_name()];

        const ref<NativeTensor>& grad_in = primal->grad_in();
        const ref<NativeTensor>& grad_out = primal->grad_out();

        if (!has_derivative()) {
            write_shader_cursor_fields(context, binding, field, primal, read_back);
        } else {
            write_shader_cursor_fields(context, binding, field["primal"], primal, read_back);
            if (m_d_in) {
                SGL_CHECK(grad_in, "Missing required input gradients");
                write_shader_cursor_fields(context, binding, field["d_in"], grad_in.get(), read_back);
            }
            if (m_d_out) {
                SGL_CHECK(grad_out, "Missing required input gradients");
                write_shader_cursor_fields(context, binding, field["d_out"], grad_out.get(), read_back);
            }
        }

        if (context->call_mode() != CallMode::prim && grad_in && grad_in == grad_out) {
            if (binding->get_access().second == AccessType::readwrite)
                SGL_THROW(
                    "inout parameter gradients need separate buffers for inputs and outputs (see Tensor.with_grads)"
                );
        }
    } else {
        NativeMarshall::write_shader_cursor_pre_dispatch(context, binding, cursor, value, read_back);
    }
}

void NativeTensorMarshall::write_shader_cursor_fields(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor field,
    NativeTensor* buffer,
    nb::list read_back
) const
{
    SGL_UNUSED(read_back);

    // Write the buffer storage.
    field["buffer"] = buffer->storage();

    // Write shape vector as an array of ints.
    const std::vector<int>& shape_vec = buffer->shape().as_vector();
    field["_shape"]
        ._set_array_unsafe(&shape_vec[0], shape_vec.size() * 4, shape_vec.size(), TypeReflection::ScalarType::int32);

    // Generate and write strides vector, clearing strides to 0
    // for dimensions that are broadcast.
    std::vector<int> strides_vec = buffer->strides().as_vector();
    const std::vector<int>& transform = binding->get_transform().as_vector();
    const std::vector<int>& call_shape = context->call_shape().as_vector();
    for (size_t i = 0; i < transform.size(); i++) {
        int csidx = transform[i];
        if (call_shape[csidx] != shape_vec[i]) {
            strides_vec[i] = 0;
        }
    }

    // Write the strides vector as an array of ints.
    auto layout_field = field["layout"];
    layout_field["strides"]._set_array_unsafe(
        &strides_vec[0],
        strides_vec.size() * 4,
        strides_vec.size(),
        TypeReflection::ScalarType::int32
    );
    layout_field["offset"] = buffer->offset();
}

void NativeTensorMarshall::read_calldata(
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

nb::object NativeTensorMarshall::create_output(CallContext* context, NativeBoundVariableRuntime* binding) const
{
    SGL_UNUSED(binding);

    // Get type, buffer layout and shape.
    ref<NativeSlangType> dtype = m_slang_element_type;
    ref<TypeLayoutReflection> layout = m_element_layout;
    auto& shape = context->call_shape();

    // Create a new structured buffer for storage.
    BufferDesc buffer_desc;
    buffer_desc.usage = BufferUsage::shader_resource | BufferUsage::unordered_access | BufferUsage::shared;
    buffer_desc.struct_size = layout->stride();
    buffer_desc.element_count = shape.element_count();
    ref<Buffer> buffer = context->device()->create_buffer(buffer_desc);

    NativeTensorDesc desc;
    desc.dtype = dtype;
    desc.element_layout = layout;
    desc.shape = shape;
    desc.strides = shape.calc_contiguous_strides();
    desc.offset = 0;

    // Create new native tensor to hold the grads.
    return nb::cast(make_ref<NativeTensor>(desc, buffer, nullptr, nullptr));
}

nb::object
NativeTensorMarshall::read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);
    return data;
}

nb::object NativeTensorMarshall::create_dispatchdata(nb::object data) const
{
    // Cast value to buffer, and get the cursor field to write to.
    auto buffer = nb::cast<NativeTensor*>(data);
    nb::dict res;
    res["buffer"] = buffer->storage();
    res["_shape"] = buffer->shape().as_vector();
    nb::dict layout;
    layout["offset"] = buffer->offset();
    layout["strides"] = buffer->strides().as_vector();
    res["layout"] = layout;
    return res;
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy_tensor)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<NativeTensorDesc, StridedBufferViewDesc>(slangpy, "NativeTensorDesc").def(nb::init<>());

    nb::class_<NativeTensor, StridedBufferView>(slangpy, "NativeTensor")
        .def(
            nb::init<NativeTensorDesc, const ref<Buffer>&, const ref<NativeTensor>&, const ref<NativeTensor>&>(),
            "desc"_a,
            "storage"_a,
            "grad_in"_a.none(),
            "grad_out"_a.none()
        )
        .def_prop_rw("grad_in", &NativeTensor::grad_in, &NativeTensor::set_grad_in, nb::none())
        .def_prop_rw("grad_out", &NativeTensor::grad_out, &NativeTensor::set_grad_out, nb::none())
        .def_prop_ro("grad", &NativeTensor::grad)
        .def("broadcast_to", &NativeTensor::broadcast_to, "shape"_a)
        .def("view", &NativeTensor::view, "shape"_a, "strides"_a = Shape(), "offset"_a = 0)
        .def("__getitem__", &NativeTensor::index)
        .def(
            "with_grads",
            &NativeTensor::with_grads,
            "grad_in"_a.none() = nullptr,
            "grad_out"_a.none() = nullptr,
            "zero"_a = false
        )
        .def("detach", &NativeTensor::detach);


    nb::class_<NativeTensorMarshall, PyNativeTensorMarshall, NativeMarshall>(slangpy, "NativeTensorMarshall") //
        .def(
            "__init__",
            [](NativeTensorMarshall& self,
               int dims,
               bool writable,
               ref<NativeSlangType> slang_type,
               ref<NativeSlangType> slang_element_type,
               ref<TypeLayoutReflection> element_layout,
               ref<NativeTensorMarshall> d_in,
               ref<NativeTensorMarshall> d_out) {
                new (&self)
                    PyNativeTensorMarshall(dims, writable, slang_type, slang_element_type, element_layout, d_in, d_out);
            },
            "dims"_a,
            "writable"_a,
            "slang_type"_a,
            "slang_element_type"_a,
            "element_layout"_a,
            "d_in"_a.none(),
            "d_out"_a.none(),
            D_NA(NativeTensorMarshall, NativeTensorMarshall)
        )
        .def_prop_ro("dims", &sgl::slangpy::NativeTensorMarshall::dims)
        .def_prop_ro("writable", &sgl::slangpy::NativeTensorMarshall::writable)
        .def_prop_ro("slang_element_type", &sgl::slangpy::NativeTensorMarshall::slang_element_type)
        .def_prop_ro("d_in", &NativeTensorMarshall::d_in)
        .def_prop_ro("d_out", &NativeTensorMarshall::d_out);
}
