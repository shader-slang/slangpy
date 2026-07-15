// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <initializer_list>
#include "nanobind.h"
#include <fmt/format.h>

#include "sgl/device/device.h"
#include "sgl/device/command.h"
#include "sgl/device/buffer_cursor.h"
#include "sgl/device/cuda_utils.h"

#include "utils/slangpytensor.h"

namespace sgl {

extern void write_shader_cursor(ShaderCursor& cursor, nb::object value);
extern void buffer_copy_from_numpy(Buffer* self, nb::ndarray<nb::numpy> data);

} // namespace sgl

namespace sgl::slangpy {

namespace {
    /// Overload accepting Shape directly and returning Shape (ZERO allocations for small shapes!)
    Shape apply_broadcast_stride_zeroing(
        const Shape& strides,
        const Shape& shape,
        const Shape& transform,
        const Shape& call_shape
    )
    {
        Shape result = strides; // Uses copy constructor (inline if <=8 dims, one allocation if >8)

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


TensorMarshall::TensorFieldOffsets TensorMarshall::extract_tensor_field_offsets(ShaderCursor tensor_cursor)
{
    TensorFieldOffsets offsets;

    ShaderCursor data_cursor = tensor_cursor.find_field("_data");
    if (!data_cursor.is_valid()) {
        offsets.is_tensorview = true;
        offsets.tensorview_offset = tensor_cursor.offset();
        offsets.is_valid = true;
        return offsets;
    }

    offsets.data = data_cursor.offset();
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

TensorMarshall::CachedBindingInfo TensorMarshall::extract_binding_info(ShaderCursor field)
{
    TensorMarshall::CachedBindingInfo offsets;

    std::string_view type_name = field.slang_type_layout()->getName();
    bool is_diff_tensor_view = type_name.find("DiffTensorView") != std::string_view::npos;
    bool is_diff_tensor = !is_diff_tensor_view && type_name.find("DiffTensor") != std::string_view::npos;

    if (is_diff_tensor) {
        // SlangPy's DiffTensor: _primal/_grad_in/_grad_out pattern
        ShaderCursor primal_field = field.find_field("_primal");
        if (primal_field.is_valid()) {
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
    } else if (is_diff_tensor_view) {
        // Slang's DiffTensorView: primal/diff pattern
        ShaderCursor primal_field = field.find_field("primal");
        ShaderCursor diff_field = field.find_field("diff");
        if (primal_field.is_valid() && diff_field.is_valid()) {
            offsets.has_grad_fields = true;
            offsets.primal = extract_tensor_field_offsets(primal_field);
            offsets.grad_in = extract_tensor_field_offsets(diff_field);
            offsets.grad_out = offsets.grad_in;
        }
    } else {
        // Plain tensor (TensorView or Tensor)
        offsets.has_grad_fields = false;
        offsets.primal = extract_tensor_field_offsets(field);
    }

    offsets.field_offset = field.offset();
    offsets.field_size = (int)field.slang_type_layout()->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);

    return offsets;
}

Shape TensorMarshall::get_shape(nb::object data) const
{
    auto buffer = nb::cast<Tensor*>(data);
    return buffer->shape();
}

void TensorMarshall::ensure_binding_info_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const
{
    if (!m_cached_binding_info.primal.is_valid) {
        ShaderCursor field = cursor[binding->variable_name()];
        m_cached_binding_info = extract_binding_info(field);
    }
}

void TensorMarshall::write_native_tensor(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderObject* shader_object,
    void* base_address,
    Tensor* primal_tensor,
    nb::list read_back
) const
{
    const ref<Tensor>& grad_in = primal_tensor->grad_in();
    const ref<Tensor>& grad_out = primal_tensor->grad_out();

    if (!m_cached_binding_info.has_grad_fields) {
        // Flat structure - write directly to primal offsets
        write_native_tensor_fields(
            context,
            binding,
            shader_object,
            base_address,
            m_cached_binding_info.primal,
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
            m_cached_binding_info.primal,
            primal_tensor,
            read_back
        );

        if (m_d_in && m_cached_binding_info.grad_in.is_valid) {
            SGL_CHECK(grad_in, "Missing required input gradients");
            write_native_tensor_fields(
                context,
                binding,
                shader_object,
                base_address,
                m_cached_binding_info.grad_in,
                grad_in.get(),
                read_back
            );
        }

        if (m_d_out && m_cached_binding_info.grad_out.is_valid) {
            SGL_CHECK(grad_out, "Missing required output gradients");
            write_native_tensor_fields(
                context,
                binding,
                shader_object,
                base_address,
                m_cached_binding_info.grad_out,
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
void TensorMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    // Initialize cached offsets on first call
    ensure_binding_info_cached(cursor, binding);

#if 0
    // Validate offsets on future calls
    if (m_cached_binding_info.primal.is_valid) {
        CachedBindingInfo offsets = extract_binding_info(cursor[binding->variable_name()]);
        SGL_CHECK(
            offsets.primal.data == m_cached_binding_info.primal.data &&
                offsets.primal.shape == m_cached_binding_info.primal.shape &&
                offsets.primal.strides == m_cached_binding_info.primal.strides &&
                offsets.primal.offset == m_cached_binding_info.primal.offset,
            "Cached primal tensor offsets do not match current shader cursor offsets"
        );
        if (offsets.grad_in.is_valid) {
                        SGL_CHECK(
                offsets.grad_in.data == m_cached_binding_info.grad_in.data &&
                    offsets.grad_in.shape == m_cached_binding_info.grad_in.shape &&
                    offsets.grad_in.strides == m_cached_binding_info.grad_in.strides &&
                    offsets.grad_in.offset == m_cached_binding_info.grad_in.offset,
                "Cached grad_in tensor offsets do not match current shader cursor offsets"
            );
        }
        if (offsets.grad_out.is_valid) {

            SGL_CHECK(
                offsets.grad_out.data == m_cached_binding_info.grad_out.data &&
                    offsets.grad_out.shape == m_cached_binding_info.grad_out.shape &&
                    offsets.grad_out.strides == m_cached_binding_info.grad_out.strides &&
                    offsets.grad_out.offset == m_cached_binding_info.grad_out.offset,
                "Cached grad_out tensor offsets do not match current shader cursor offsets"
            );
        }
    }
#endif

    // Try Tensor path
    Tensor* primal;
    if (nb::try_cast(value, primal)) {
        ShaderObject* shader_object = cursor.shader_object();
        void* base_address
            = shader_object->reserve_data(m_cached_binding_info.field_offset, m_cached_binding_info.field_size);

        // Write the differentiated tensor structure
        write_native_tensor(context, binding, shader_object, base_address, primal, read_back);

        // Check for gradient aliasing issues
        const ref<Tensor>& grad_in = primal->grad_in();
        const ref<Tensor>& grad_out = primal->grad_out();
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

void TensorMarshall::write_tensor_fields_from_buffer(
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
            offsets.data.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
            buffer->device_address()
        );
    } else {
        shader_object->set_buffer(offsets.data, buffer);
    }

    write_strided_array_helper(
        base_address,
        offsets.shape.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
        shape,
        offsets.array_stride
    );

    write_strided_array_helper(
        base_address,
        offsets.strides.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
        strides,
        offsets.array_stride
    );

    write_value_helper(
        base_address,
        offsets.offset.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
        offset
    );

    // Write element byte stride if field exists (for AtomicTensor on Metal)
    // This is needed because sizeof(T) in shader may differ from buffer stride
    // due to alignment requirements (e.g., sizeof(float3)=12 but Metal buffer stride=16)
    if (offsets.element_byte_stride.is_valid()) {
        write_value_helper(
            base_address,
            offsets.element_byte_stride.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
            static_cast<uint32_t>(buffer->desc().struct_size)
        );
    }
}

void TensorMarshall::write_tensor_fields_from_pointer(
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
        offsets.data.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
        address
    );

    // Write shape and strides using the same mechanism as write_tensor_fields_from_buffer
    write_strided_array_helper(
        base_address,
        offsets.shape.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
        shape,
        offsets.array_stride
    );

    write_strided_array_helper(
        base_address,
        offsets.strides.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
        strides,
        offsets.array_stride
    );

    write_value_helper(
        base_address,
        offsets.offset.uniform_offset - m_cached_binding_info.field_offset.uniform_offset,
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

void TensorMarshall::write_native_tensor_fields(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderObject* shader_object,
    void* base_address,
    const TensorFieldOffsets& offsets,
    Tensor* tensor,
    nb::list read_back
) const
{
    SGL_UNUSED(read_back);

    const Shape& shape = tensor->shape();
    Shape strides
        = apply_broadcast_stride_zeroing(tensor->strides(), shape, binding->transform(), context->call_shape());

    if (offsets.is_tensorview) {
        // TensorView path: build TensorViewData struct and write via set_data()
        TensorViewData tvd
            = Tensor::make_tensor_view_data(tensor->storage(), shape, strides, tensor->offset(), element_stride());
        shader_object->set_data(offsets.tensorview_offset, &tvd, sizeof(TensorViewData));
        return;
    }

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

void TensorMarshall::read_calldata(
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

nb::object TensorMarshall::create_output(CallContext* context, NativeBoundVariableRuntime* binding) const
{
    SGL_UNUSED(binding);

    // Get type, buffer layout and shape.
    ref<refl::Type> dtype = m_slang_element_type;
    ref<TypeLayoutReflection> layout = m_element_layout;
    auto& shape = context->call_shape();

    // Create a new structured buffer for storage.
    BufferDesc buffer_desc;
    buffer_desc.usage = BufferUsage::shader_resource | BufferUsage::unordered_access | BufferUsage::shared;
    buffer_desc.struct_size = layout->stride();
    buffer_desc.element_count = shape.element_count();
    ref<Buffer> buffer = context->device()->create_buffer(buffer_desc);

    TensorDesc desc;
    desc.dtype = dtype;
    desc.element_layout = layout;
    desc.shape = shape;
    desc.strides = shape.calc_contiguous_strides();
    desc.offset = 0;

    // Create new native tensor to hold the grads.
    return nb::cast(make_ref<Tensor>(desc, buffer, nullptr, nullptr));
}

nb::object TensorMarshall::read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);
    return data;
}

nb::object TensorMarshall::create_dispatchdata(nb::object data) const
{
    // Cast value to buffer, and get the cursor field to write to.
    auto buffer = nb::cast<Tensor*>(data);
    nb::dict res;
    res["_data"] = buffer->storage();
    res["_shape"] = shape_to_list(buffer->shape());
    res["_offset"] = buffer->offset();
    res["_strides"] = shape_to_list(buffer->strides());
    return res;
}

ref<Tensor> NativeNumpyMarshall::create_tensor(Device* device, const Shape& shape) const
{
    BufferDesc buffer_desc;
    buffer_desc.usage = BufferUsage::shader_resource | BufferUsage::unordered_access;
    buffer_desc.struct_size = element_layout()->stride();
    buffer_desc.element_count = shape.element_count();
    ref<Buffer> buffer = device->create_buffer(buffer_desc);

    TensorDesc desc;
    desc.dtype = slang_element_type();
    desc.element_layout = element_layout();
    desc.offset = 0;
    desc.shape = shape;
    desc.strides = shape.calc_contiguous_strides();
    desc.usage = buffer_desc.usage;
    desc.memory_type = buffer_desc.memory_type;
    return make_ref<Tensor>(desc, buffer, nullptr, nullptr);
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

    auto tensor = create_tensor(context->device(), shape);
    buffer_copy_from_numpy(tensor->storage().get(), ndarray);

    auto tensor_obj = nb::cast(tensor);
    store_readback(binding, read_back, value, tensor_obj);

    TensorMarshall::write_shader_cursor_pre_dispatch(context, binding, cursor, tensor_obj, read_back);
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
    auto tensor = nb::cast<Tensor*>(result);

    size_t numpy_data_size = ndarray.nbytes();
    size_t buffer_data_size = tensor->storage()->size();
    SGL_CHECK(
        numpy_data_size == buffer_data_size,
        "numpy array size does not match the tensor storage ({} > {})",
        numpy_data_size,
        buffer_data_size
    );

    tensor->storage()->get_data(ndarray.data(), buffer_data_size);
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

    std::vector<size_t> sizes;
    sizes.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        sizes.push_back(shape[i]);
    }

    auto ndarray = nb::ndarray<nb::numpy>(data, sizes.size(), sizes.data(), owner, nullptr, m_dtype);
    return nb::cast(ndarray);
}

nb::object NativeNumpyMarshall::create_dispatchdata(nb::object data) const
{
    SGL_UNUSED(data);
    SGL_THROW("Raw dispatch is not supported for numpy arrays.");
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy_tensor)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<TensorMarshall, PyTensorMarshall, NativeMarshall>(slangpy, "TensorMarshall") //
        .def(
            "__init__",
            [](TensorMarshall& self,
               int dims,
               bool writable,
               ref<refl::Type> slang_type,
               ref<refl::Type> slang_element_type,
               ref<TypeLayoutReflection> element_layout,
               ref<TensorMarshall> d_in,
               ref<TensorMarshall> d_out)
            {
                new (&self)
                    PyTensorMarshall(dims, writable, slang_type, slang_element_type, element_layout, d_in, d_out);
            },
            "dims"_a,
            "writable"_a,
            "slang_type"_a,
            "slang_element_type"_a,
            "element_layout"_a,
            "d_in"_a.none(),
            "d_out"_a.none(),
            D_NA(TensorMarshall, TensorMarshall)
        )
        .def_prop_ro("dims", &sgl::slangpy::TensorMarshall::dims)
        .def_prop_ro("writable", &sgl::slangpy::TensorMarshall::writable)
        .def_prop_ro("slang_element_type", &sgl::slangpy::TensorMarshall::slang_element_type)
        .def_prop_ro("d_in", &TensorMarshall::d_in)
        .def_prop_ro("d_out", &TensorMarshall::d_out);

    nb::class_<NativeNumpyMarshall, TensorMarshall>(slangpy, "NativeNumpyMarshall") //
        .def(
            "__init__",
            [](NativeNumpyMarshall& self,
               int dims,
               ref<refl::Type> slang_type,
               ref<refl::Type> slang_element_type,
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
