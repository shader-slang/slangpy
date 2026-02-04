// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <initializer_list>
#include "nanobind.h"
#include <fmt/format.h>

#include "sgl/device/device.h"
#include "sgl/device/buffer_cursor.h"

#include "utils/slangpytensor.h"

namespace sgl {

extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);

} // namespace sgl

namespace sgl::slangpy {

namespace {
    /// Helper function to extract shape from PyTorch tensor
    /// Creates Shape directly and populates it - zero allocations for tensors with ≤8 dimensions
    Shape extract_shape(const nb::ndarray<nb::pytorch, nb::device::cuda>& tensor)
    {
        const size_t ndim = tensor.ndim();
        Shape shape(ndim);
        int* shape_data = shape.data(); // Get pointer once
        for (size_t i = 0; i < ndim; i++) {
            shape_data[i] = static_cast<int>(tensor.shape(i));
        }
        return shape;
    }

    /// Helper function to extract strides from PyTorch tensor
    /// Returns element strides directly (PyTorch stride() already returns element strides for nanobind)
    /// Creates Shape directly and populates it - zero allocations for tensors with ≤8 dimensions
    Shape extract_strides(const nb::ndarray<nb::pytorch, nb::device::cuda>& tensor)
    {
        const size_t ndim = tensor.ndim();
        Shape strides(ndim);
        int* strides_data = strides.data(); // Get pointer once
        for (size_t i = 0; i < ndim; i++) {
            // nanobind's tensor.stride() returns element strides, not byte strides
            strides_data[i] = static_cast<int>(tensor.stride(i));
        }
        return strides;
    }

    /// Overload accepting Shape directly and returning Shape (ZERO allocations for small shapes!)
    Shape apply_broadcast_stride_zeroing(
        const Shape& strides,
        const Shape& shape,
        const Shape& transform,
        const Shape& call_shape
    )
    {
        Shape result = strides; // Uses copy constructor (inline if ≤8 dims, one allocation if >8)

        // Get raw pointers once to avoid per-element m_uses_heap branching
        const int* transform_data = transform.data();
        const int* shape_data = shape.data();
        const int* call_shape_data = call_shape.data();
        int* result_data = result.data();
        const size_t count = transform.size();

        for (size_t i = 0; i < count; i++) {
            int csidx = transform_data[i];
            if (call_shape_data[csidx] != shape_data[i]) {
                result_data[i] = 0;
            }
        }
        return result;
    }

    /// Helper for writing single value to base address with offset
    template<typename T>
    void write_value_helper(void* base_address, size_t offset, const T& value)
    {
        T* ptr = reinterpret_cast<T*>(static_cast<uint8_t*>(base_address) + offset);
        *ptr = value;
    }

    /// Helper for writing strided array to base address with offset (raw pointer version)
    template<typename T>
    void write_strided_array_helper(
        void* base_address,
        size_t offset,
        const T* data,
        size_t element_count,
        size_t element_stride
    )
    {
        uint8_t* dest_ptr = static_cast<uint8_t*>(base_address) + offset;
        for (size_t i = 0; i < element_count; i++) {
            T* ptr = reinterpret_cast<T*>(dest_ptr + i * element_stride);
            *ptr = data[i];
        }
    }

    /// Helper for writing strided array from Shape to base address with offset (zero allocation)
    void write_strided_array_helper(void* base_address, size_t offset, const Shape& shape, size_t element_stride)
    {
        uint8_t* dest_ptr = static_cast<uint8_t*>(base_address) + offset;
        const int* shape_data = shape.data(); // Get pointer once - single branch
        const size_t count = shape.size();
        for (size_t i = 0; i < count; i++) {
            int* ptr = reinterpret_cast<int*>(dest_ptr + i * element_stride);
            *ptr = shape_data[i]; // Direct pointer access - no branch
        }
    }

    /// Validate that a tensor's trailing dimensions match the expected vector type shape.
    /// This mirrors the validation in Python's torchtensormarshall.py
    /// Throws nb::value_error to match the Python behavior
    void validate_tensor_shape(const Shape& tensor_shape, const Shape& vector_shape)
    {
        const size_t vector_dims = vector_shape.size();
        if (vector_dims == 0) {
            return; // No vector shape to validate against
        }

        const size_t tensor_dims = tensor_shape.size();
        if (tensor_dims < vector_dims) {
            throw nb::value_error(
                fmt::format(
                    "Tensor shape {} does not match expected shape {}",
                    tensor_shape.to_string(),
                    vector_shape.to_string()
                )
                    .c_str()
            );
        }

        // Get raw pointers once to avoid per-element m_uses_heap branching
        const int* tensor_data = tensor_shape.data();
        const int* vector_data = vector_shape.data();

        // Check trailing dimensions
        for (size_t i = 0; i < vector_dims; i++) {
            int expected = vector_data[vector_dims - 1 - i];
            int actual = tensor_data[tensor_dims - 1 - i];
            // -1 acts as a wildcard (matches any size)
            if (expected != -1 && actual != expected) {
                throw nb::value_error(
                    fmt::format(
                        "Tensor shape {} does not match expected shape {}",
                        tensor_shape.to_string(),
                        vector_shape.to_string()
                    )
                        .c_str()
                );
            }
        }
    }
} // anonymous namespace


NativeTensorMarshall::TensorFieldOffsets NativeTensorMarshall::extract_tensor_field_offsets(ShaderCursor tensor_cursor)
{
    TensorFieldOffsets offsets;

    // TensorView<T> is a magic type with no visible fields.
    // Detect it by type name and use fixed layout (56 bytes written via set_data).
    const char* type_name = tensor_cursor.slang_type_layout()->getType()->getName();
    if (type_name && strcmp(type_name, "TensorView") == 0) {
        offsets.is_tensorview = true;
        offsets.is_valid = true;
        return offsets;
    }

    offsets.data = tensor_cursor["_data"].offset();
    offsets.shape = tensor_cursor["_shape"].offset();
    offsets.strides = tensor_cursor["_strides"].offset();
    offsets.offset = tensor_cursor["_offset"].offset();

    // Extract element_byte_stride offset if present (for AtomicTensor on Metal)
    ShaderCursor ebs_field = tensor_cursor.find_field("_element_byte_stride");
    if (ebs_field.is_valid())
        offsets.element_byte_stride = ebs_field.offset();

    offsets.is_valid = true;
    offsets.array_stride
        = (int)tensor_cursor["_shape"].slang_type_layout()->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
    return offsets;
}

NativeTensorMarshall::CachedOffsets NativeTensorMarshall::extract_offsets(ShaderCursor field)
{
    NativeTensorMarshall::CachedOffsets offsets;

    ShaderCursor primal_field = field.find_field("_primal");
    if (!primal_field.is_valid()) {
        offsets.has_grad_fields = false;
        offsets.primal = extract_tensor_field_offsets(field);
    } else {
        offsets.has_grad_fields = true;
        offsets.primal = extract_tensor_field_offsets(primal_field);

        ShaderCursor grad_in_field = field.find_field("_grad_in");
        if (grad_in_field.is_valid()) {
            offsets.grad_in = extract_tensor_field_offsets(grad_in_field);
        }
        ShaderCursor grad_out_field = field.find_field("_grad_out");
        if (grad_out_field.is_valid()) {
            offsets.grad_out = extract_tensor_field_offsets(grad_out_field);
        }
    }

    offsets.field_offset = field.offset();
    offsets.field_size = (int)field.slang_type_layout()->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);

    return offsets;
}

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
ref<NativeTensor> NativeTensor::index(nb::object index_arg) const
{
    auto result = make_ref<NativeTensor>(desc(), storage(), m_grad_in, m_grad_out);
    result->index_inplace(index_arg);
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
            SGL_THROW("No derivative type found for {}", m_desc.dtype->type_reflection()->name());
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

std::string NativeTensor::to_string() const
{
    return fmt::format(
        "NativeTensor(dtype={}, shape={}, has_grad_in={}, has_grad_out={})",
        m_desc.dtype->to_string(),
        m_desc.shape.to_string(),
        m_grad_in ? "true" : "false",
        m_grad_out ? "true" : "false"
    );
}

Shape NativeTensorMarshall::get_shape(nb::object data) const
{
    auto buffer = nb::cast<NativeTensor*>(data);
    return buffer->shape();
}

void NativeTensorMarshall::ensure_offsets_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const
{
    if (!m_cached_offsets.primal.is_valid) {
        ShaderCursor field = cursor[binding->variable_name()];
        m_cached_offsets = extract_offsets(field);
    }
}

void NativeTensorMarshall::write_native_tensor(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderObject* shader_object,
    void* base_address,
    NativeTensor* primal_tensor,
    nb::list read_back
) const
{
    const ref<NativeTensor>& grad_in = primal_tensor->grad_in();
    const ref<NativeTensor>& grad_out = primal_tensor->grad_out();

    if (!m_cached_offsets.has_grad_fields) {
        // Flat structure - write directly to primal offsets
        write_native_tensor_fields(
            context,
            binding,
            shader_object,
            base_address,
            m_cached_offsets.primal,
            primal_tensor,
            read_back
        );
    } else {
        // Differentiated structure - write to primal, grad_in, grad_out
        write_native_tensor_fields(
            context,
            binding,
            shader_object,
            base_address,
            m_cached_offsets.primal,
            primal_tensor,
            read_back
        );

        if (m_d_in && m_cached_offsets.grad_in.is_valid) {
            SGL_CHECK(grad_in, "Missing required input gradients");
            write_native_tensor_fields(
                context,
                binding,
                shader_object,
                base_address,
                m_cached_offsets.grad_in,
                grad_in.get(),
                read_back
            );
        }

        if (m_d_out && m_cached_offsets.grad_out.is_valid) {
            SGL_CHECK(grad_out, "Missing required output gradients");
            write_native_tensor_fields(
                context,
                binding,
                shader_object,
                base_address,
                m_cached_offsets.grad_out,
                grad_out.get(),
                read_back
            );
        }
    }
}

/**
 * Write tensor data to shader uniforms using pre-cached reflection offsets.
 * This is the optimized path that avoids repeated shader cursor navigation.
 *
 * The offsets are cached on first call and reused for subsequent calls.
 * This assumes the shader structure layout remains constant across calls.
 */
void NativeTensorMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    // Initialize cached offsets on first call
    ensure_offsets_cached(cursor, binding);

#if 0
    // Validate offsets on future calls
    if (m_cached_offsets.primal.is_valid) {
        CachedOffsets offsets = extract_offsets(cursor[binding->variable_name()]);
        SGL_CHECK(
            offsets.primal.data == m_cached_offsets.primal.data &&
                offsets.primal.shape == m_cached_offsets.primal.shape &&
                offsets.primal.strides == m_cached_offsets.primal.strides &&
                offsets.primal.offset == m_cached_offsets.primal.offset,
            "Cached primal tensor offsets do not match current shader cursor offsets"
        );
        if (offsets.grad_in.is_valid) {
                        SGL_CHECK(
                offsets.grad_in.data == m_cached_offsets.grad_in.data &&
                    offsets.grad_in.shape == m_cached_offsets.grad_in.shape &&
                    offsets.grad_in.strides == m_cached_offsets.grad_in.strides &&
                    offsets.grad_in.offset == m_cached_offsets.grad_in.offset,
                "Cached grad_in tensor offsets do not match current shader cursor offsets"
            );
        }
        if (offsets.grad_out.is_valid) {

            SGL_CHECK(
                offsets.grad_out.data == m_cached_offsets.grad_out.data &&
                    offsets.grad_out.shape == m_cached_offsets.grad_out.shape &&
                    offsets.grad_out.strides == m_cached_offsets.grad_out.strides &&
                    offsets.grad_out.offset == m_cached_offsets.grad_out.offset,
                "Cached grad_out tensor offsets do not match current shader cursor offsets"
            );
        }
    }
#endif

    // Try NativeTensor path
    NativeTensor* primal;
    if (nb::try_cast(value, primal)) {
        ShaderObject* shader_object = cursor.shader_object();
        void* base_address = shader_object->reserve_data(m_cached_offsets.field_offset, m_cached_offsets.field_size);

        // Write the differentiated tensor structure
        write_native_tensor(context, binding, shader_object, base_address, primal, read_back);

        // Check for gradient aliasing issues
        const ref<NativeTensor>& grad_in = primal->grad_in();
        const ref<NativeTensor>& grad_out = primal->grad_out();
        if (context->call_mode() != CallMode::prim && grad_in && grad_in == grad_out) {
            if (binding->access().second == AccessType::readwrite)
                SGL_THROW(
                    "inout parameter gradients need separate buffers for inputs and outputs (see Tensor.with_grads)"
                );
        }
        return;
    }

    // Fall back to base class for all other cases
    NativeMarshall::write_shader_cursor_pre_dispatch(context, binding, cursor, value, read_back);
}

void NativeTensorMarshall::write_tensor_fields_from_buffer(
    ShaderObject* shader_object,
    void* base_address,
    const TensorFieldOffsets& offsets,
    const ref<Buffer>& buffer,
    const Shape& shape,
    const Shape& strides,
    int offset
) const
{
    if (offsets.data.binding_range_index == offsets.shape.binding_range_index) {
        write_value_helper(
            base_address,
            offsets.data.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
            buffer->device_address()
        );
    } else {
        shader_object->set_buffer(offsets.data, buffer);
    }

    write_strided_array_helper(
        base_address,
        offsets.shape.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        shape,
        offsets.array_stride
    );

    write_strided_array_helper(
        base_address,
        offsets.strides.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        strides,
        offsets.array_stride
    );

    write_value_helper(
        base_address,
        offsets.offset.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        offset
    );

    // Write element byte stride if field exists (for AtomicTensor on Metal)
    // This is needed because sizeof(T) in shader may differ from buffer stride
    // due to alignment requirements (e.g., sizeof(float3)=12 but Metal buffer stride=16)
    if (offsets.element_byte_stride.is_valid()) {
        write_value_helper(
            base_address,
            offsets.element_byte_stride.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
            static_cast<uint32_t>(buffer->desc().struct_size)
        );
    }
}

void NativeTensorMarshall::write_tensor_fields_from_pointer(
    ShaderObject* shader_object,
    void* base_address,
    const TensorFieldOffsets& offsets,
    void* data_ptr,
    const Shape& shape,
    const Shape& strides,
    int offset
) const
{
    SGL_UNUSED(shader_object);

    // Write device pointer
    DeviceAddress address = reinterpret_cast<DeviceAddress>(data_ptr);
    write_value_helper(
        base_address,
        offsets.data.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        address
    );

    // Write shape and strides using the same mechanism as write_tensor_fields_from_buffer
    write_strided_array_helper(
        base_address,
        offsets.shape.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        shape,
        offsets.array_stride
    );

    write_strided_array_helper(
        base_address,
        offsets.strides.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        strides,
        offsets.array_stride
    );

    write_value_helper(
        base_address,
        offsets.offset.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        offset
    );

    // Note: element_byte_stride is not written here for PyTorch tensors.
    // On CUDA (the only backend PyTorch supports), _element_byte_stride is static const in shader.
    // This field only exists as a runtime field on Metal (for AtomicTensor), and PyTorch doesn't support Metal.
    SGL_CHECK(
        !offsets.element_byte_stride.is_valid(),
        "Unexpected element_byte_stride field for PyTorch tensor - this path should only be used on CUDA"
    );
}

void NativeTensorMarshall::write_native_tensor_fields(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderObject* shader_object,
    void* base_address,
    const TensorFieldOffsets& offsets,
    NativeTensor* tensor,
    nb::list read_back
) const
{
    SGL_UNUSED(read_back);

    const Shape& shape = tensor->shape();
    Shape strides
        = apply_broadcast_stride_zeroing(tensor->strides(), shape, binding->transform(), context->call_shape());

    write_tensor_fields_from_buffer(
        shader_object,
        base_address,
        offsets,
        tensor->storage(),
        shape,
        strides,
        tensor->offset()
    );
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
    res["_data"] = buffer->storage();
    res["_shape"] = shape_to_list(buffer->shape());
    res["_offset"] = buffer->offset();
    res["_strides"] = shape_to_list(buffer->strides());
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
            "zero"_a = true
        )
        .def("detach", &NativeTensor::detach)
        .def("__repr__", &NativeTensor::to_string);


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
               ref<NativeTensorMarshall> d_out)
            {
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
