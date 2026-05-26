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
    /// Helper function to extract shape from PyTorch tensor
    /// Creates Shape directly and populates it - zero allocations for tensors with <=8 dimensions
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
    /// Creates Shape directly and populates it - zero allocations for tensors with <=8 dimensions
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

    std::optional<nb::dlpack::dtype> scalartype_to_dtype(TypeReflection::ScalarType scalar_type)
    {
        switch (scalar_type) {
        case TypeReflection::ScalarType::none_:
            return {};
        case TypeReflection::ScalarType::void_:
            return {};
        case TypeReflection::ScalarType::bool_:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Bool, 8, 1};
        case TypeReflection::ScalarType::int32:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 32, 1};
        case TypeReflection::ScalarType::uint32:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 32, 1};
        case TypeReflection::ScalarType::int64:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 64, 1};
        case TypeReflection::ScalarType::uint64:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 64, 1};
        case TypeReflection::ScalarType::float16:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Float, 16, 1};
        case TypeReflection::ScalarType::float32:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Float, 32, 1};
        case TypeReflection::ScalarType::float64:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Float, 64, 1};
        case TypeReflection::ScalarType::int8:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 8, 1};
        case TypeReflection::ScalarType::uint8:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 8, 1};
        case TypeReflection::ScalarType::int16:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, 16, 1};
        case TypeReflection::ScalarType::uint16:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, 16, 1};
        case TypeReflection::ScalarType::intptr:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, sizeof(intptr_t) * 8, 1};
        case TypeReflection::ScalarType::uintptr:
            return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, sizeof(uintptr_t) * 8, 1};
        default:
            return {};
        }
    }

    ref<refl::Type> innermost_type(ref<refl::Type> type)
    {
        ref<refl::Type> result = std::move(type);
        while (true) {
            ref<refl::Type> child = result->element_type();
            if (!child || child == result) {
                break;
            }
            result = child;
        }
        return result;
    }

    template<typename Framework>
    nb::ndarray<Framework> to_ndarray(void* data, nb::handle owner, const NativeTensorDesc& desc)
    {
        size_t dtype_size = desc.element_layout->stride();
        ref<refl::Type> innermost = innermost_type(desc.dtype);
        ref<TypeLayoutReflection> innermost_layout = innermost->buffer_type_layout();

        bool is_scalar = innermost_layout->type()->kind() == TypeReflection::Kind::scalar;
        auto dtype_shape = desc.dtype->shape();
        auto dtype_strides = dtype_shape.calc_contiguous_strides();

        size_t innermost_size = is_scalar ? innermost_layout->stride() : 1;
        TypeReflection::ScalarType scalar_type
            = is_scalar ? innermost_layout->type()->scalar_type() : TypeReflection::ScalarType::uint8;
        auto dlpack_type = scalartype_to_dtype(scalar_type);
        SGL_CHECK(dlpack_type.has_value(), "Unsupported tensor element scalar type");

        std::vector<size_t> sizes;
        std::vector<int64_t> strides;

        for (size_t i = 0; i < desc.shape.size(); ++i) {
            sizes.push_back(desc.shape[i]);
            strides.push_back(desc.strides[i] * dtype_size / innermost_size);
        }
        for (size_t i = 0; i < dtype_shape.size(); ++i) {
            sizes.push_back(dtype_shape[i]);
            strides.push_back(dtype_strides[i]);
        }
        if (!is_scalar) {
            sizes.push_back(innermost_layout->stride());
            strides.push_back(1);
        }

        auto device = Framework::value == nb::pytorch::value ? nb::device::cuda::value : nb::device::cpu::value;
        return nb::ndarray<Framework>(data, sizes.size(), sizes.data(), owner, strides.data(), *dlpack_type, device);
    }

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
    : m_desc(std::move(desc))
    , m_storage(storage)
    , m_grad_in(grad_in)
    , m_grad_out(grad_out)
{
    SGL_CHECK(m_storage, "NativeTensor requires storage");

    set_slangpy_signature(fmt::format("[{},{},{}]", m_desc.dtype->full_name(), m_desc.shape.size(), m_desc.usage));
}

bool NativeTensor::is_contiguous() const
{
    const Shape& shape_ref = shape();
    const Shape& stride_ref = strides();

    int prod = 1;
    for (int i = dims() - 1; i >= 0; --i) {
        if (shape_ref[i] == 1)
            continue;

        if (stride_ref[i] != prod)
            return false;
        prod *= shape_ref[i];
    }

    return true;
}

ref<BufferCursor> NativeTensor::cursor(std::optional<int> start, std::optional<int> count) const
{
    size_t el_stride = m_desc.element_layout->stride();
    size_t size = (count.value_or(element_count())) * el_stride;
    size_t byte_offset = (m_desc.offset + start.value_or(0)) * el_stride;
    return make_ref<BufferCursor>(m_desc.element_layout, m_storage, size, byte_offset);
}

nb::dict NativeTensor::uniforms() const
{
    nb::dict res;
    res["_data"] = storage();
    res["_shape"] = shape_to_list(shape());
    res["_offset"] = offset();
    res["_strides"] = shape_to_list(strides());
    return res;
}

void NativeTensor::view_inplace(Shape shape, Shape strides, int offset)
{
    SGL_CHECK(shape.valid(), "New shape must be valid");

    if (!strides.valid() || strides.size() == 0)
        strides = shape.calc_contiguous_strides();

    for (size_t i = 0; i < strides.size(); ++i) {
        SGL_CHECK(strides[i] >= 0, "Strides must be positive");
    }

    SGL_CHECK(
        shape.size() == strides.size(),
        "Shape dimensions ({}) must match stride dimensions ({})",
        shape.size(),
        strides.size()
    );

    m_desc.shape = shape;
    m_desc.strides = strides;
    m_desc.offset += offset;

    SGL_CHECK(m_desc.offset >= 0, "Tensor view offset is negative");
}

void NativeTensor::broadcast_to_inplace(const Shape& new_shape)
{
    const Shape& curr_shape = shape();
    const Shape& curr_strides = strides();

    int dim_delta = static_cast<int>(new_shape.size()) - static_cast<int>(curr_shape.size());
    if (dim_delta < 0) {
        SGL_THROW("Broadcast shape must be larger than tensor shape");
    }

    for (size_t i = 0; i < curr_shape.size(); ++i) {
        if (curr_shape[i] != new_shape[dim_delta + i] && curr_shape[i] != 1) {
            SGL_THROW(
                "Current dimension {} at index {} must be equal to new dimension {} or 1",
                curr_shape[i],
                i,
                new_shape[dim_delta + i]
            );
        }
    }

    Shape new_strides(new_shape.size());
    int* new_strides_data = new_strides.data();
    for (size_t i = 0; i < new_shape.size(); ++i) {
        new_strides_data[i] = 0;
    }
    for (size_t i = 0; i < curr_shape.size(); ++i) {
        if (curr_shape[i] > 1) {
            new_strides_data[dim_delta + i] = curr_strides[i];
        }
    }

    view_inplace(new_shape, new_strides);
}

void NativeTensor::index_inplace(nb::object index_arg)
{
    std::vector<nb::handle> args;
    if (nb::isinstance<nb::tuple>(index_arg)) {
        nb::tuple t = nb::cast<nb::tuple>(index_arg);
        args.insert(args.end(), t.begin(), t.end());
    } else {
        args.push_back(index_arg);
    }

    int real_dims = 0;
    for (auto v : args) {
        if (nb::isinstance<int>(v) || nb::isinstance<nb::slice>(v))
            real_dims++;
    }
    SGL_CHECK(real_dims <= dims(), "Too many indices for tensor of dimension {}", dims());

    const Shape& cur_shape = shape();
    const Shape& cur_strides = strides();

    int dim = 0;
    int offset = 0;
    std::vector<int> out_shape;
    std::vector<int> out_strides;

    for (size_t i = 0; i < args.size(); ++i) {
        const nb::handle& arg = args[i];

        if (nb::isinstance<int>(arg)) {
            int idx = nb::cast<int>(arg);
            if (idx < -cur_shape[dim] || idx >= cur_shape[dim])
                throw nb::index_error();

            if (idx < 0)
                idx += cur_shape[dim];
            offset += idx * cur_strides[dim];
            dim++;
        } else if (nb::isinstance<nb::slice>(arg)) {
            nb::slice slice = nb::cast<nb::slice>(arg);
            auto adjusted = slice.compute(cur_shape[dim]);
            size_t start = adjusted.get<0>();
            size_t step = adjusted.get<2>();
            size_t slice_length = adjusted.get<3>();

            SGL_CHECK(step > 0, "Slice step must be greater than zero (found stride {} at dimension {})", step, i);

            offset += int(start) * cur_strides[dim];
            out_shape.push_back(int(slice_length));
            out_strides.push_back(int(step) * cur_strides[dim]);
            dim++;
        } else if (nb::isinstance<nb::ellipsis>(arg)) {
            int skipped_dims = dims() - real_dims;
            for (int j = 0; j < skipped_dims; ++j) {
                out_shape.push_back(cur_shape[dim + j]);
                out_strides.push_back(cur_strides[dim + j]);
            }
            dim += skipped_dims;
        } else if (arg.is_none()) {
            out_shape.push_back(1);
            out_strides.push_back(0);
        } else {
            auto type_name = nb::str(arg.type());
            SGL_THROW(
                "Illegal argument at dimension {}: Allowed are int, slice, ..., or None; found {} instead",
                i,
                type_name.c_str()
            );
        }
    }

    int remaining = dims() - dim;
    for (int j = 0; j < remaining; ++j) {
        out_shape.push_back(cur_shape[dim + j]);
        out_strides.push_back(cur_strides[dim + j]);
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
        out_strides.push_back(1);
    }

    view_inplace(Shape(out_shape), Shape(out_strides), offset);
}

void NativeTensor::clear(CommandEncoder* cmd)
{
    if (cmd) {
        cmd->clear_buffer(m_storage);
    } else {
        ref<CommandEncoder> temp_cmd = device()->create_command_encoder();
        temp_cmd->clear_buffer(m_storage);
        device()->submit_command_buffer(temp_cmd->finish());
    }
}

nb::ndarray<nb::numpy> NativeTensor::to_numpy() const
{
    size_t dtype_size = m_desc.element_layout->stride();
    size_t byte_offset = m_desc.offset * dtype_size;
    size_t data_size = m_storage->size() - byte_offset;
    void* data = new uint8_t[data_size];
    m_storage->get_data(data, data_size, byte_offset);
    nb::capsule owner(
        data,
        [](void* p) noexcept
        {
            delete[] reinterpret_cast<uint8_t*>(p);
        }
    );

    return to_ndarray<nb::numpy>(data, owner, m_desc);
}

nb::ndarray<nb::pytorch> NativeTensor::to_torch() const
{
    size_t dtype_size = m_desc.element_layout->stride();
    size_t byte_offset = m_desc.offset * dtype_size;
    void* data = reinterpret_cast<uint8_t*>(m_storage->cuda_memory()) + byte_offset;
    auto owner = nb::find(this);

    return to_ndarray<nb::pytorch>(data, owner, m_desc);
}

bool NativeTensor::maybe_pad_data(nb::ndarray<nb::numpy> data, size_t dtype_size, size_t byte_offset)
{
    if (m_desc.element_layout->kind() == TypeReflection::Kind::vector) {
        size_t scalar_size = m_desc.element_layout->element_type_layout()->size();
        size_t required_element_num = dtype_size / scalar_size;
        size_t actual_element_num = data.shape(data.ndim() - 1);
        if (actual_element_num < required_element_num) {
            nb::object np = nb::module_::import_("numpy");
            size_t padding_element_num = required_element_num - actual_element_num;
            nb::list pad_width;
            for (size_t i = 0; i < data.ndim() - 1; i++) {
                pad_width.append(nb::make_tuple(0, 0));
            }
            pad_width.append(nb::make_tuple(0, padding_element_num));

            nb::object arr_obj = nb::cast(data);
            nb::object padding_data = np.attr("pad")(arr_obj, pad_width, "constant_values"_a = 0);

            nb::ndarray<nb::numpy> out = nb::cast<nb::ndarray<nb::numpy>>(padding_data);
            size_t data_size = out.nbytes();
            m_storage->set_data(out.data(), data_size, byte_offset);
            return true;
        }
    }
    return false;
}

void NativeTensor::copy_from_numpy(nb::ndarray<nb::numpy> data)
{
    SGL_CHECK(is_ndarray_contiguous(data), "Source Numpy array must be contiguous");
    SGL_CHECK(is_contiguous(), "Destination tensor must be contiguous");

    size_t dtype_size = m_desc.element_layout->stride();
    size_t byte_offset = m_desc.offset * dtype_size;
    size_t data_size = data.nbytes();
    size_t buffer_size = m_storage->size() - byte_offset;
    SGL_CHECK(
        data_size <= buffer_size,
        "Numpy array is larger than the tensor storage ({} > {})",
        data_size,
        buffer_size
    );

    if (maybe_pad_data(data, dtype_size, byte_offset)) {
        return;
    }

    m_storage->set_data(data.data(), data_size, byte_offset);
}

void NativeTensor::copy_from_torch(nb::object tensor)
{
    bool is_cuda = nb::cast<bool>(tensor.attr("is_cuda"));
    bool has_cuda_memory = m_storage->cuda_memory() != nullptr;

    if (is_cuda && has_cuda_memory) {
        SGL_CHECK(is_contiguous(), "Destination tensor must be contiguous");

        nb::object contiguous_tensor = tensor.attr("contiguous")();
        nb::object data_ptr = contiguous_tensor.attr("data_ptr")();
        void* src_data = reinterpret_cast<void*>(nb::cast<uintptr_t>(data_ptr));

        size_t tensor_bytes = nb::cast<size_t>(contiguous_tensor.attr("numel")())
            * nb::cast<size_t>(contiguous_tensor.attr("element_size")());

        size_t dtype_size = m_desc.element_layout->stride();
        size_t byte_offset = m_desc.offset * dtype_size;
        size_t buffer_size = m_storage->size() - byte_offset;
        SGL_CHECK(
            tensor_bytes <= buffer_size,
            "Tensor is larger than the tensor storage ({} > {})",
            tensor_bytes,
            buffer_size
        );

        SGL_CU_SCOPE(m_storage->device());

        void* dst_data = reinterpret_cast<uint8_t*>(m_storage->cuda_memory()) + byte_offset;
        sgl::cuda::memcpy_device_to_device(dst_data, src_data, tensor_bytes);
    } else {
        nb::object numpy_array = tensor.attr("cpu")().attr("numpy")();
        nb::ndarray<nb::numpy> numpy_data = nb::cast<nb::ndarray<nb::numpy>>(numpy_array);
        copy_from_numpy(numpy_data);
    }
}

void NativeTensor::point_to(ref<NativeTensor> target)
{
    SGL_CHECK(shape() == target->shape(), "Shape of existing and new tensor view must match");
    SGL_CHECK(usage() == target->usage(), "Usage flags of existing and new tensor storage must match");
    SGL_CHECK(memory_type() == target->memory_type(), "Memory type of existing and new tensor storage must match");
    SGL_CHECK(element_stride() == target->element_stride(), "Element size of new and existing data type must match");

    m_desc.offset = target->offset();
    m_desc.strides = target->strides();
    m_storage = target->m_storage;
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
        ref<refl::Type> dtype = m_desc.dtype->derivative();
        if (!dtype)
            SGL_THROW("No derivative type found for {}", m_desc.dtype->name());
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
    ref<refl::Type> dtype = m_slang_element_type;
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

ref<NativeTensor> NativeNumpyMarshall::create_tensor(Device* device, const Shape& shape) const
{
    BufferDesc buffer_desc;
    buffer_desc.usage = BufferUsage::shader_resource | BufferUsage::unordered_access;
    buffer_desc.struct_size = element_layout()->stride();
    buffer_desc.element_count = shape.element_count();
    ref<Buffer> buffer = device->create_buffer(buffer_desc);

    NativeTensorDesc desc;
    desc.dtype = slang_element_type();
    desc.element_layout = element_layout();
    desc.offset = 0;
    desc.shape = shape;
    desc.strides = shape.calc_contiguous_strides();
    desc.usage = buffer_desc.usage;
    desc.memory_type = buffer_desc.memory_type;
    return make_ref<NativeTensor>(desc, buffer, nullptr, nullptr);
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

    NativeTensorMarshall::write_shader_cursor_pre_dispatch(context, binding, cursor, tensor_obj, read_back);
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
    auto tensor = nb::cast<NativeTensor*>(result);

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

    nb::class_<NativeTensorDesc>(slangpy, "NativeTensorDesc")
        .def(nb::init<>())
        .def_rw("dtype", &NativeTensorDesc::dtype)
        .def_rw("element_layout", &NativeTensorDesc::element_layout)
        .def_rw("offset", &NativeTensorDesc::offset)
        .def_rw("shape", &NativeTensorDesc::shape)
        .def_rw("strides", &NativeTensorDesc::strides)
        .def_rw("usage", &NativeTensorDesc::usage)
        .def_rw("memory_type", &NativeTensorDesc::memory_type);

    nb::class_<NativeTensor, NativeObject>(slangpy, "NativeTensor")
        .def(
            nb::init<NativeTensorDesc, const ref<Buffer>&, const ref<NativeTensor>&, const ref<NativeTensor>&>(),
            "desc"_a,
            "storage"_a,
            "grad_in"_a.none(),
            "grad_out"_a.none()
        )
        .def_prop_ro("device", &NativeTensor::device)
        .def_prop_ro("dtype", &NativeTensor::dtype)
        .def_prop_ro("offset", &NativeTensor::offset)
        .def_prop_ro("shape", &NativeTensor::shape)
        .def_prop_ro("strides", &NativeTensor::strides)
        .def_prop_ro("element_count", &NativeTensor::element_count)
        .def_prop_ro("usage", &NativeTensor::usage)
        .def_prop_ro("memory_type", &NativeTensor::memory_type)
        .def_prop_ro("storage", &NativeTensor::storage)
        .def_prop_rw("grad_in", &NativeTensor::grad_in, &NativeTensor::set_grad_in, nb::none())
        .def_prop_rw("grad_out", &NativeTensor::grad_out, &NativeTensor::set_grad_out, nb::none())
        .def_prop_ro("grad", &NativeTensor::grad)
        .def("clear", &NativeTensor::clear, "cmd"_a.none() = nullptr)
        .def("cursor", &NativeTensor::cursor, "start"_a.none() = std::nullopt, "count"_a.none() = std::nullopt)
        .def("uniforms", &NativeTensor::uniforms)
        .def("to_numpy", &NativeTensor::to_numpy)
        .def("to_torch", &NativeTensor::to_torch)
        .def("copy_from_numpy", &NativeTensor::copy_from_numpy, "data"_a)
        .def("copy_from_torch", &NativeTensor::copy_from_torch, "tensor"_a)
        .def("is_contiguous", &NativeTensor::is_contiguous)
        .def("point_to", &NativeTensor::point_to, "target"_a)
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
               ref<refl::Type> slang_type,
               ref<refl::Type> slang_element_type,
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

    nb::class_<NativeNumpyMarshall, NativeTensorMarshall>(slangpy, "NativeNumpyMarshall") //
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
