// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "slangpytorchtensor.h"

#include "sgl/device/device.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/buffer_cursor.h"

#include <fmt/format.h>

namespace sgl::slangpy {

namespace {

    /// Helper function to create Shape from TensorBridgeInfo
    Shape shape_from_bridge_info(const TensorBridgeInfo& info)
    {
        Shape shape(info.ndim);
        int* data = shape.data();
        for (int i = 0; i < info.ndim; i++) {
            data[i] = static_cast<int>(info.shape[i]);
        }
        return shape;
    }

    /// Helper function to create strides Shape from TensorBridgeInfo
    Shape strides_from_bridge_info(const TensorBridgeInfo& info)
    {
        Shape strides(info.ndim);
        int* data = strides.data();
        for (int i = 0; i < info.ndim; i++) {
            data[i] = static_cast<int>(info.strides[i]);
        }
        return strides;
    }

    /// Apply broadcast stride zeroing
    /// Replicates the logic from slangpytensor.cpp
    Shape apply_broadcast_stride_zeroing(
        const Shape& strides,
        const Shape& shape,
        const Shape& transform,
        const Shape& call_shape
    )
    {
        Shape result = strides;
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

    /// Validate tensor shape against expected vector type shape
    void validate_tensor_shape(const Shape& tensor_shape, const Shape& vector_shape)
    {
        const size_t vector_dims = vector_shape.size();
        if (vector_dims == 0) {
            return;
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

        const int* tensor_data = tensor_shape.data();
        const int* vector_data = vector_shape.data();

        for (size_t i = 0; i < vector_dims; i++) {
            int expected = vector_data[vector_dims - 1 - i];
            int actual = tensor_data[tensor_dims - 1 - i];
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

    /// Helper for writing single value to base address with offset
    template<typename T>
    void write_value_helper(void* base_address, size_t offset, const T& value)
    {
        T* ptr = reinterpret_cast<T*>(static_cast<uint8_t*>(base_address) + offset);
        *ptr = value;
    }

    /// Helper for writing strided array from Shape to base address with offset
    void write_strided_array_helper(void* base_address, size_t offset, const Shape& shape, size_t element_stride)
    {
        uint8_t* dest_ptr = static_cast<uint8_t*>(base_address) + offset;
        const int* shape_data = shape.data();
        const size_t count = shape.size();
        for (size_t i = 0; i < count; i++) {
            int* ptr = reinterpret_cast<int*>(dest_ptr + i * element_stride);
            *ptr = shape_data[i];
        }
    }

} // anonymous namespace


NativeTorchTensorMarshall::NativeTorchTensorMarshall(
    int dims,
    bool writable,
    ref<NativeSlangType> slang_type,
    ref<NativeSlangType> slang_element_type,
    ref<TypeLayoutReflection> element_layout,
    ref<NativeTorchTensorMarshall> d_in,
    ref<NativeTorchTensorMarshall> d_out
)
    : NativeMarshall(slang_type)
    , m_dims(dims)
    , m_writable(writable)
    , m_slang_element_type(slang_element_type)
    , m_element_layout(element_layout)
    , m_d_in(d_in)
    , m_d_out(d_out)
{
}

Shape NativeTorchTensorMarshall::get_shape(nb::object data) const
{
    // Use TorchBridge for fast native shape extraction
    TensorBridgeInfo info;
    if (TorchBridge::instance().extract(data.ptr(), info)) {
        return shape_from_bridge_info(info);
    }

    // Fallback: return unknown shape with all -1 dimensions
    Shape result(m_dims);
    int* result_data = result.data();
    for (int i = 0; i < m_dims; i++) {
        result_data[i] = -1;
    }
    return result;
}

void NativeTorchTensorMarshall::ensure_offsets_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const
{
    if (!m_cached_offsets.primal.is_valid) {
        ShaderCursor field = cursor[binding->variable_name()];
        m_cached_offsets = NativeTensorMarshall::extract_offsets(field);
    }
}

void NativeTorchTensorMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    // Ensure cached offsets are initialized
    ensure_offsets_cached(cursor, binding);

    // Extract tensor info using TorchBridge
    TensorBridgeInfo info;
    if (!TorchBridge::instance().extract(value.ptr(), info)) {
        SGL_THROW("Expected torch.Tensor, got {}", nb::type_name(value.type()).c_str());
    }

    // Validate shape
    Shape tensor_shape = shape_from_bridge_info(info);
    validate_tensor_shape(tensor_shape, binding->vector_type()->shape());

    // Only support CUDA tensors on CUDA devices for now
    if (!info.is_cuda) {
        SGL_THROW("Non-CUDA torch tensors are not yet supported. Tensor must be on CUDA device.");
    }
    if (context->device()->type() != DeviceType::cuda) {
        SGL_THROW("torch.Tensor support currently requires CUDA device backend.");
    }

    ShaderObject* shader_object = cursor.shader_object();
    void* base_address = shader_object->reserve_data(m_cached_offsets.field_offset, m_cached_offsets.field_size);

    // Write tensor fields
    if (!m_cached_offsets.has_grad_fields) {
        // Flat structure - write directly to primal offsets
        write_torch_tensor_fields(context, binding, shader_object, base_address, m_cached_offsets.primal, info);
    } else {
        // Differentiated structure - write primal
        write_torch_tensor_fields(context, binding, shader_object, base_address, m_cached_offsets.primal, info);

        // Gradient tensors for raw torch.Tensor are not yet supported
        if (m_d_in && m_cached_offsets.grad_in.is_valid) {
            SGL_THROW("Gradient tensors not yet supported for raw torch.Tensor");
        }
        if (m_d_out && m_cached_offsets.grad_out.is_valid) {
            SGL_THROW("Gradient tensors not yet supported for raw torch.Tensor");
        }
    }
}

void NativeTorchTensorMarshall::write_torch_tensor_fields(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderObject* shader_object,
    void* base_address,
    const TensorFieldOffsets& offsets,
    const TensorBridgeInfo& info
) const
{
    SGL_UNUSED(shader_object);

    Shape shape = shape_from_bridge_info(info);
    Shape strides = strides_from_bridge_info(info);

    // Apply broadcast stride zeroing
    strides = apply_broadcast_stride_zeroing(strides, shape, binding->transform(), context->call_shape());

    // Write device pointer
    DeviceAddress address = reinterpret_cast<DeviceAddress>(info.data_ptr);
    write_value_helper(
        base_address,
        offsets.data.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        address
    );

    // Write shape
    write_strided_array_helper(
        base_address,
        offsets.shape.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        shape,
        offsets.array_stride
    );

    // Write strides
    write_strided_array_helper(
        base_address,
        offsets.strides.uniform_offset - m_cached_offsets.field_offset.uniform_offset,
        strides,
        offsets.array_stride
    );

    // Write offset (always 0 for raw tensors)
    write_value_helper(base_address, offsets.offset.uniform_offset - m_cached_offsets.field_offset.uniform_offset, 0);
}

void NativeTorchTensorMarshall::read_calldata(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    nb::object data,
    nb::object result
) const
{
    // For CUDA tensors on CUDA devices, no read-back is needed
    // The tensor data is written directly via device pointer
    SGL_UNUSED(context);
    SGL_UNUSED(binding);
    SGL_UNUSED(data);
    SGL_UNUSED(result);
}

nb::object NativeTorchTensorMarshall::create_output(CallContext* context, NativeBoundVariableRuntime* binding) const
{
    SGL_UNUSED(binding);

    // Import torch and create output tensor
    nb::module_ torch = nb::module_::import_("torch");

    // Build shape: call_shape + element type shape
    const Shape& call_shape = context->call_shape();
    const Shape& elem_shape = m_slang_element_type->shape();

    std::vector<int64_t> shape_vec;
    shape_vec.reserve(call_shape.size() + elem_shape.size());
    for (size_t i = 0; i < call_shape.size(); i++) {
        shape_vec.push_back(call_shape[i]);
    }
    for (size_t i = 0; i < elem_shape.size(); i++) {
        shape_vec.push_back(elem_shape[i]);
    }

    // Get torch dtype from slang scalar type
    // Map slang scalar type to torch dtype
    TypeReflection::ScalarType scalar_type = m_slang_element_type->type_reflection()->scalar_type();
    nb::object dtype;
    switch (scalar_type) {
    case TypeReflection::ScalarType::int8:
        dtype = torch.attr("int8");
        break;
    case TypeReflection::ScalarType::int16:
        dtype = torch.attr("int16");
        break;
    case TypeReflection::ScalarType::int32:
        dtype = torch.attr("int32");
        break;
    case TypeReflection::ScalarType::int64:
        dtype = torch.attr("int64");
        break;
    case TypeReflection::ScalarType::uint8:
        dtype = torch.attr("uint8");
        break;
    case TypeReflection::ScalarType::float16:
        dtype = torch.attr("float16");
        break;
    case TypeReflection::ScalarType::float32:
        dtype = torch.attr("float32");
        break;
    case TypeReflection::ScalarType::float64:
        dtype = torch.attr("float64");
        break;
    default:
        SGL_THROW("Unsupported scalar type for torch output tensor");
    }

    return torch.attr("empty")(nb::cast(shape_vec), "dtype"_a = dtype, "device"_a = "cuda");
}

nb::object
NativeTorchTensorMarshall::read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);
    // Tensor is already populated, just return it
    return data;
}

nb::object NativeTorchTensorMarshall::create_dispatchdata(nb::object data) const
{
    // Extract tensor info for Python fallback path
    TensorBridgeInfo info;
    if (!TorchBridge::instance().extract(data.ptr(), info)) {
        SGL_THROW("Expected torch.Tensor for create_dispatchdata");
    }

    Shape shape = shape_from_bridge_info(info);
    Shape strides = strides_from_bridge_info(info);

    nb::dict res;
    res["_data"] = reinterpret_cast<uintptr_t>(info.data_ptr);
    res["_shape"] = shape.as_vector();
    res["_offset"] = 0;
    res["_strides"] = strides.as_vector();
    return res;
}

} // namespace sgl::slangpy


SGL_PY_EXPORT(utils_slangpy_torch_tensor)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<NativeTorchTensorMarshall, PyNativeTorchTensorMarshall, NativeMarshall>(
        slangpy,
        "NativeTorchTensorMarshall"
    )
        .def(
            "__init__",
            [](NativeTorchTensorMarshall& self,
               int dims,
               bool writable,
               ref<NativeSlangType> slang_type,
               ref<NativeSlangType> slang_element_type,
               ref<TypeLayoutReflection> element_layout,
               ref<NativeTorchTensorMarshall> d_in,
               ref<NativeTorchTensorMarshall> d_out)
            {
                new (&self) PyNativeTorchTensorMarshall(
                    dims,
                    writable,
                    slang_type,
                    slang_element_type,
                    element_layout,
                    d_in,
                    d_out
                );
            },
            "dims"_a,
            "writable"_a,
            "slang_type"_a,
            "slang_element_type"_a,
            "element_layout"_a,
            "d_in"_a.none(),
            "d_out"_a.none()
        )
        .def_prop_ro("dims", &NativeTorchTensorMarshall::dims)
        .def_prop_ro("writable", &NativeTorchTensorMarshall::writable)
        .def_prop_ro("slang_element_type", &NativeTorchTensorMarshall::slang_element_type)
        .def_prop_ro("element_layout", &NativeTorchTensorMarshall::element_layout)
        .def_prop_ro("has_derivative", &NativeTorchTensorMarshall::has_derivative)
        .def_prop_ro("d_in", &NativeTorchTensorMarshall::d_in)
        .def_prop_ro("d_out", &NativeTorchTensorMarshall::d_out);
}
