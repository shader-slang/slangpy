// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

} // anonymous namespace


NativeTensorMarshall::TensorFieldOffsets NativeTensorMarshall::extract_tensor_field_offsets(ShaderCursor tensor_cursor)
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

NativeTensorMarshall::CachedBindingInfo NativeTensorMarshall::extract_binding_info(ShaderCursor field)
{
    NativeTensorMarshall::CachedBindingInfo offsets;

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

void NativeTensorMarshall::ensure_binding_info_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const
{
    if (!m_cached_binding_info.primal.is_valid) {
        ShaderCursor field = cursor[binding->variable_name()];
        m_cached_binding_info = extract_binding_info(field);
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
void NativeTensorMarshall::write_shader_cursor_pre_dispatch(
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

    // Try NativeTensor path
    NativeTensor* primal;
    if (nb::try_cast(value, primal)) {
        ShaderObject* shader_object = cursor.shader_object();
        void* base_address
            = shader_object->reserve_data(m_cached_binding_info.field_offset, m_cached_binding_info.field_size);

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

    if (offsets.is_tensorview) {
        // TensorView path: build TensorViewData struct and write via set_data()
        TensorViewData tvd = {};
        // Device address is buffer base + byte offset
        tvd.data = tensor->storage()->device_address() + tensor->offset() * element_stride();

        const int ndim = static_cast<int>(shape.size());
        // TensorView strides are in bytes, NativeTensor strides are in elements
        for (int i = 0; i < ndim && i < kSlangPyTensorViewMaxDim; i++) {
            tvd.strides[i] = static_cast<uint32_t>(strides[i] * element_stride());
            tvd.sizes[i] = static_cast<uint32_t>(shape[i]);
        }
        tvd.dimensionCount = static_cast<uint32_t>(ndim);
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

    // Allow tuples/lists to be passed wherever Shape is expected.
    nb::implicitly_convertible<nb::tuple, Shape>();
    nb::implicitly_convertible<nb::list, Shape>();

    // ---------------------------------------------------------------------------
    // Type resolution helpers (call back to Python reflection system)
    // ---------------------------------------------------------------------------

    /// Resolve dtype from various Python representations to NativeSlangType.
    /// Fast path if dtype is already a NativeSlangType; otherwise calls Python.
    auto resolve_dtype = [](Device* device, nb::object dtype, nb::object program_layout) -> ref<NativeSlangType>
    {
        if (nb::isinstance<NativeSlangType>(dtype))
            return nb::cast<ref<NativeSlangType>>(dtype);
        nb::module_ lookup = nb::module_::import_("slangpy.reflection.lookup");
        nb::object layout = lookup.attr("resolve_program_layout")(device, dtype, program_layout);
        return nb::cast<ref<NativeSlangType>>(lookup.attr("resolve_element_type")(layout, dtype));
    };

    /// Map a numpy dtype to a NativeSlangType, returns nullptr on failure.
    auto resolve_numpy_dtype
        = [](Device* device, nb::object np_dtype, nb::object program_layout) -> ref<NativeSlangType>
    {
        nb::module_ lookup = nb::module_::import_("slangpy.reflection.lookup");
        nb::object result = lookup.attr("numpy_to_slang")(np_dtype, device, program_layout);
        if (result.is_none())
            return nullptr;
        return nb::cast<ref<NativeSlangType>>(result);
    };

    /// Create a NativeTensor from storage buffer + dtype + shape + strides.
    auto make_tensor = [](const ref<Buffer>& storage,
                          const ref<NativeSlangType>& dtype,
                          const Shape& shape,
                          const Shape& strides,
                          int offset = 0) -> ref<NativeTensor>
    {
        NativeTensorDesc desc;
        desc.shape = shape;
        desc.strides = strides;
        desc.offset = offset;
        desc.dtype = dtype;
        desc.element_layout = dtype->buffer_type_layout();
        desc.usage = storage->desc().usage;
        return make_ref<NativeTensor>(desc, storage, nullptr, nullptr);
    };

    /// Shared implementation for empty().
    auto tensor_empty_impl = [&](const ref<Device>& device,
                                 const Shape& shape,
                                 nb::object dtype_obj,
                                 BufferUsage usage,
                                 MemoryType memory_type,
                                 nb::object program_layout) -> ref<NativeTensor>
    {
        ref<NativeSlangType> dtype = resolve_dtype(device, dtype_obj, program_layout);
        if (!dtype)
            throw nb::value_error("Element type (dtype) must be specified");
        if (shape.size() == 0)
            throw nb::value_error("Cannot create a tensor with zero dimensions");

        size_t num_elems = shape.element_count();
        ref<TypeLayoutReflection> layout = dtype->buffer_type_layout();

        BufferDesc buffer_desc;
        buffer_desc.element_count = num_elems;
        buffer_desc.struct_size = layout->stride();
        buffer_desc.usage = usage;
        buffer_desc.memory_type = memory_type;

        ref<Buffer> buffer = device->create_buffer(buffer_desc);
        return make_tensor(buffer, dtype, shape, shape.calc_contiguous_strides());
    };

    // ---------------------------------------------------------------------------
    // NativeTensorDesc
    // ---------------------------------------------------------------------------

    nb::class_<NativeTensorDesc, StridedBufferViewDesc>(slangpy, "NativeTensorDesc").def(nb::init<>());

    // ---------------------------------------------------------------------------
    // Tensor (was NativeTensor)
    // ---------------------------------------------------------------------------

    auto tensor_cls = nb::class_<NativeTensor, StridedBufferView>(slangpy, "Tensor");

    // Internal desc-based constructor (used by marshalls etc.)
    tensor_cls.def(
        nb::init<NativeTensorDesc, const ref<Buffer>&, const ref<NativeTensor>&, const ref<NativeTensor>&>(),
        "desc"_a,
        "storage"_a,
        "grad_in"_a.none(),
        "grad_out"_a.none()
    );

    // User-friendly constructor: Tensor(storage, dtype, shape, strides=None, offset=0, ...)
    tensor_cls.def(
        "__init__",
        [](NativeTensor& self,
           const ref<Buffer>& storage,
           const ref<NativeSlangType>& dtype,
           const Shape& shape,
           std::optional<Shape> strides,
           int offset,
           ref<NativeTensor> grad_in,
           ref<NativeTensor> grad_out)
        {
            Shape actual_strides = strides.has_value() ? *strides : shape.calc_contiguous_strides();
            if (actual_strides.size() != shape.size())
                throw nb::value_error("Number of strides must match number of dimensions");

            NativeTensorDesc desc;
            desc.shape = shape;
            desc.strides = actual_strides;
            desc.offset = offset;
            desc.dtype = dtype;
            desc.element_layout = dtype->buffer_type_layout();
            desc.usage = storage->desc().usage;
            new (&self) NativeTensor(desc, storage, grad_in, grad_out);
        },
        "storage"_a,
        "dtype"_a,
        "shape"_a,
        "strides"_a.none() = nb::none(),
        "offset"_a = 0,
        "grad_in"_a.none() = nb::none(),
        "grad_out"_a.none() = nb::none()
    );

    // Properties & methods inherited from StridedBufferView are already bound.
    tensor_cls //
        .def_prop_rw("grad_in", &NativeTensor::grad_in, &NativeTensor::set_grad_in, nb::none())
        .def_prop_rw("grad_out", &NativeTensor::grad_out, &NativeTensor::set_grad_out, nb::none())
        .def_prop_ro("grad", &NativeTensor::grad)
        .def(
            "broadcast_to",
            &NativeTensor::broadcast_to,
            "shape"_a,
            "Returns a new view of the tensor with the requested shape, following standard broadcasting rules."
        )
        .def(
            "view",
            &NativeTensor::view,
            "shape"_a,
            "strides"_a = Shape(),
            "offset"_a = 0,
            "Returns a new view of the tensor with the requested shape, strides and offset.\n"
            "The offset is in elements (not bytes) and is specified relative to the current offset."
        )
        .def("__getitem__", &NativeTensor::index)
        .def(
            "with_grads",
            &NativeTensor::with_grads,
            "grad_in"_a.none() = nb::none(),
            "grad_out"_a.none() = nb::none(),
            "zero"_a = true,
            "Returns a new tensor view with gradients attached. If called with no arguments, the\n"
            "tensor defaults to attaching a zeros-like initialized gradient tensor for both input and\n"
            "output gradients.\n"
            "\n"
            "Specifying input gradients (grad_in) and/or output gradients (grad_out) allows more precise\n"
            "control over the gradient tensors, and is key when using a function that has inout parameters,\n"
            "so will want to both read and write gradients without causing race conditions.\n"
            "\n"
            "When differentiating a slang call that wrote results to a tensor, gradients of the output will\n"
            "be read from grad_in (if not None). When differentiating a slang call that read inputs from a\n"
            "tensor, input gradients will be written to grad_out (if not None)."
        )
        .def(
            "detach",
            &NativeTensor::detach,
            "Returns a new tensor view with gradients detached. The returned tensor will not have any\n"
            "gradients attached, and will not be differentiable."
        )
        .def("__repr__", &NativeTensor::to_string)
        .def(
            "__str__",
            [](NativeTensor& self)
            {
                return nb::str(nb::cast(self.to_numpy()));
            }
        );

    // ---------------------------------------------------------------------------
    // Static factory methods
    // ---------------------------------------------------------------------------

    tensor_cls.def_static(
        "from_numpy",
        [&](const ref<Device>& device,
            nb::object ndarray_obj,
            BufferUsage usage,
            MemoryType memory_type,
            nb::object program_layout) -> ref<NativeTensor>
        {
            // Cast to ndarray for direct C++ access to shape/strides/data
            nb::ndarray<nb::numpy> arr = nb::cast<nb::ndarray<nb::numpy>>(ndarray_obj);

            // Resolve numpy dtype -> Slang type (requires Python object)
            nb::object np_dtype = ndarray_obj.attr("dtype");
            ref<NativeSlangType> dtype = resolve_numpy_dtype(device, np_dtype, program_layout);
            if (!dtype)
                throw nb::value_error(
                    fmt::format("Unsupported numpy dtype {}", nb::cast<std::string>(nb::str(np_dtype))).c_str()
                );

            // Extract shape and strides directly from ndarray
            // ndarray strides are already in elements (DLPack convention)
            size_t ndim = arr.ndim();
            size_t itemsize = arr.itemsize();
            if (itemsize == 0)
                throw nb::value_error("Unsupported numpy array");

            Shape shape(ndim);
            Shape strides(ndim);
            for (size_t i = 0; i < ndim; i++) {
                shape[i] = static_cast<int>(arr.shape(i));
                strides[i] = static_cast<int>(arr.stride(i));
            }

            // Upload raw memory to GPU buffer
            size_t N = arr.nbytes() / itemsize;
            BufferDesc buffer_desc;
            buffer_desc.struct_size = itemsize;
            buffer_desc.element_count = N;
            buffer_desc.usage = usage;
            buffer_desc.memory_type = memory_type;
            buffer_desc.data = arr.data();
            buffer_desc.data_size = arr.nbytes();

            ref<Buffer> buffer = device->create_buffer(buffer_desc);
            return make_tensor(buffer, dtype, shape, strides);
        },
        "device"_a,
        "ndarray"_a,
        "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
        "memory_type"_a = MemoryType::device_local,
        "program_layout"_a.none() = nb::none(),
        "Creates a new tensor with the same contents, shape and strides as the given numpy array."
    );

    tensor_cls.def_static(
        "empty",
        [&](const ref<Device>& device,
            nb::object shape_obj,
            nb::object dtype_obj,
            BufferUsage usage,
            MemoryType memory_type,
            nb::object program_layout,
            nb::object element_count_obj) -> ref<NativeTensor>
        {
            Shape shape;
            if (!element_count_obj.is_none()) {
                nb::module_ warnings = nb::module_::import_("warnings");
                warnings.attr("warn")(
                    "element_count parameter is deprecated; use shape instead",
                    nb::handle(PyExc_DeprecationWarning),
                    "stacklevel"_a = 2
                );
                shape = Shape({nb::cast<int>(element_count_obj)});
            } else {
                shape = nb::cast<Shape>(shape_obj);
            }
            return tensor_empty_impl(device, shape, dtype_obj, usage, memory_type, program_layout);
        },
        "device"_a,
        "shape"_a = Shape(),
        "dtype"_a = nb::none(),
        "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
        "memory_type"_a = MemoryType::device_local,
        "program_layout"_a.none() = nb::none(),
        "element_count"_a.none() = nb::none(),
        "Creates a tensor with the requested shape and element type without attempting to initialize the data."
    );

    tensor_cls.def_static(
        "zeros",
        [&](const ref<Device>& device,
            const Shape& shape,
            nb::object dtype_obj,
            BufferUsage usage,
            MemoryType memory_type,
            nb::object program_layout) -> ref<NativeTensor>
        {
            auto tensor = tensor_empty_impl(device, shape, dtype_obj, usage, memory_type, program_layout);
            tensor->clear();
            return tensor;
        },
        "device"_a,
        "shape"_a,
        "dtype"_a,
        "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
        "memory_type"_a = MemoryType::device_local,
        "program_layout"_a.none() = nb::none(),
        "Creates a zero-initialized tensor with the requested shape and element type."
    );

    tensor_cls.def_static(
        "empty_like",
        [](const ref<NativeTensor>& other) -> ref<NativeTensor>
        {
            NativeTensorDesc desc;
            desc.shape = other->shape();
            desc.strides = other->shape().calc_contiguous_strides();
            desc.offset = 0;
            desc.dtype = other->dtype();
            desc.element_layout = other->desc().element_layout;
            desc.usage = other->usage();

            BufferDesc buffer_desc;
            buffer_desc.element_count = desc.shape.element_count();
            buffer_desc.struct_size = desc.element_layout->stride();
            buffer_desc.usage = desc.usage;

            ref<Buffer> buffer = other->device()->create_buffer(buffer_desc);
            return make_ref<NativeTensor>(desc, buffer, nullptr, nullptr);
        },
        "other"_a,
        "Creates a new tensor with the same shape and element type as the given tensor, without initializing the data."
    );

    tensor_cls.def_static(
        "zeros_like",
        [](const ref<NativeTensor>& other) -> ref<NativeTensor>
        {
            NativeTensorDesc desc;
            desc.shape = other->shape();
            desc.strides = other->shape().calc_contiguous_strides();
            desc.offset = 0;
            desc.dtype = other->dtype();
            desc.element_layout = other->desc().element_layout;
            desc.usage = other->usage();

            BufferDesc buffer_desc;
            buffer_desc.element_count = desc.shape.element_count();
            buffer_desc.struct_size = desc.element_layout->stride();
            buffer_desc.usage = desc.usage;

            ref<Buffer> buffer = other->device()->create_buffer(buffer_desc);
            auto tensor = make_ref<NativeTensor>(desc, buffer, nullptr, nullptr);
            tensor->clear();
            return tensor;
        },
        "other"_a,
        "Creates a zero-initialized tensor with the same shape and element type as the given tensor."
    );

    tensor_cls.def_static(
        "from_torch",
        [&](const ref<Device>& device,
            nb::object torch_tensor,
            nb::object dtype_obj,
            BufferUsage usage,
            nb::object program_layout) -> ref<NativeTensor>
        {
            ref<NativeSlangType> dtype = resolve_dtype(device, dtype_obj, program_layout);

            size_t struct_stride = dtype->buffer_type_layout()->stride();
            size_t scalar_size = nb::cast<size_t>(torch_tensor.attr("element_size")());

            if (struct_stride == 0 || scalar_size == 0 || struct_stride % scalar_size != 0)
                throw nb::value_error(
                    fmt::format(
                        "Torch element size ({}) is not compatible with Slang type '{}' buffer stride ({})",
                        scalar_size,
                        dtype->type_reflection()->full_name(),
                        struct_stride
                    )
                        .c_str()
                );

            size_t scalars_per_element = struct_stride / scalar_size;
            int dim = nb::cast<int>(torch_tensor.attr("dim")());
            if (dim < 1)
                throw nb::value_error("Tensor must have at least 1 dimension");

            nb::tuple torch_shape = nb::cast<nb::tuple>(torch_tensor.attr("shape"));
            size_t last_dim = nb::cast<size_t>(torch_shape[dim - 1]);
            if (last_dim != scalars_per_element)
                throw nb::value_error(
                    fmt::format(
                        "Last dimension size ({}) does not match the number of scalars per '{}' element ({})",
                        last_dim,
                        dtype->type_reflection()->full_name(),
                        scalars_per_element
                    )
                        .c_str()
                );

            int64_t last_stride = nb::cast<int64_t>(torch_tensor.attr("stride")(dim - 1));
            if (last_stride != 1)
                throw nb::value_error("Last dimension of the tensor must be contiguous");

            // Make contiguous and compute element count
            nb::object contiguous = torch_tensor.attr("contiguous")();
            size_t element_count = nb::cast<size_t>(contiguous.attr("numel")());

            // Create buffer
            BufferDesc buffer_desc;
            buffer_desc.size = element_count * scalar_size;
            buffer_desc.struct_size = struct_stride;
            buffer_desc.usage = usage | BufferUsage::shared;

            ref<Buffer> buffer = device->create_buffer(buffer_desc);

            // Copy data from torch tensor to buffer
            nb::module_ spy_ext = nb::module_::import_("slangpy");
            spy_ext.attr("copy_torch_tensor_to_buffer")(contiguous, nb::cast(buffer));

            // Build outer shape (all dims except last)
            Shape outer_shape(dim - 1);
            nb::tuple cont_shape = nb::cast<nb::tuple>(contiguous.attr("shape"));
            for (int i = 0; i < dim - 1; i++)
                outer_shape[i] = nb::cast<int>(cont_shape[i]);

            return make_tensor(buffer, dtype, outer_shape, outer_shape.calc_contiguous_strides());
        },
        "device"_a,
        "tensor"_a,
        "dtype"_a,
        "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
        "program_layout"_a.none() = nb::none(),
        "Reinterpret a torch.Tensor as a slangpy Tensor with a given element type.\n"
        "\n"
        "The last dimension of the torch tensor is treated as packed storage for one\n"
        "element of ``dtype``. For example, a ``torch.Tensor`` of shape ``(N, 2)`` with\n"
        "``dtype=module.Vec2`` (a struct with two float fields) produces a\n"
        "``Tensor`` of shape ``(N,)`` with element type ``Vec2``.\n"
        "\n"
        ".. warning::\n"
        "\n"
        "    Struct layout may vary between platforms due to padding/alignment.\n"
        "    The caller is responsible for ensuring the torch tensor's memory layout\n"
        "    matches the Slang struct's buffer layout."
    );

    // Backward compatibility alias: NativeTensor = Tensor
    slangpy.attr("NativeTensor") = slangpy.attr("Tensor");


    // ---------------------------------------------------------------------------
    // NativeTensorMarshall
    // ---------------------------------------------------------------------------

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
