// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <initializer_list>
#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"
#include "sgl/device/buffer_cursor.h"
#include "sgl/device/cuda_utils.h"

#include "utils/slangpybuffer.h"

namespace sgl::slangpy {

inline std::optional<nb::dlpack::dtype> scalartype_to_dtype(TypeReflection::ScalarType scalar_type)
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

ref<NativeSlangType> innermost_type(ref<NativeSlangType> type)
{
    ref<NativeSlangType> result = type;
    while (true) {
        ref<NativeSlangType> child = result->element_type();
        if (!child || child == result) {
            break;
        }
        result = child;
    }
    return result;
}

StridedBufferView::StridedBufferView(Device* device, const StridedBufferViewDesc& desc, ref<Buffer> storage)
{
    if (!storage) {
        BufferDesc buffer_desc;
        buffer_desc.element_count = desc.offset + desc.shape.element_count();
        buffer_desc.struct_size = desc.element_layout->stride();
        buffer_desc.usage = desc.usage;
        buffer_desc.memory_type = desc.memory_type;
        storage = device->create_buffer(buffer_desc);
    }
    m_storage = std::move(storage);

    set_slangpy_signature(
        fmt::format("[{},{},{}]", desc.dtype->get_type_reflection()->full_name(), desc.shape.size(), desc.usage)
    );
}

bool StridedBufferView::is_contiguous() const
{
    const auto& shape_vec = shape().as_vector();
    const auto& stride_vec = strides().as_vector();

    int prod = 1;
    for (int i = dims() - 1; i >= 0; --i) {
        // Ignore strides of singleton dimensions
        if (shape_vec[i] == 1)
            continue;

        if (stride_vec[i] != prod)
            return false;
        prod *= shape_vec[i];
    }

    return true;
}

ref<BufferCursor> StridedBufferView::cursor(std::optional<int> start, std::optional<int> count) const
{
    size_t el_stride = desc().element_layout->stride();
    size_t size = (count.value_or(element_count())) * el_stride;
    size_t offset = (desc().offset + start.value_or(0)) * el_stride;
    return make_ref<BufferCursor>(desc().element_layout, m_storage, size, offset);
}

nb::dict StridedBufferView::uniforms() const
{
    nb::dict res;
    res["buffer"] = storage();
    res["_shape"] = shape().as_vector();
    nb::dict layout;
    layout["offset"] = offset();
    layout["strides"] = strides().as_vector();
    res["layout"] = layout;
    return res;
}

void StridedBufferView::view_inplace(Shape shape, Shape strides, int offset)
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

    desc().shape = shape;
    desc().strides = strides;
    desc().offset += offset;

    SGL_CHECK(desc().offset >= 0, "Buffer view offset is negative!");
}

void StridedBufferView::broadcast_to_inplace(const Shape& new_shape)
{
    // This 'broadcasts' the buffer view to a new shape, i.e.
    // - Prepend extra dimensions of the new shape to the front our shape
    // - Expand our singleton dimensions to the new shape
    auto& curr_shape_vec = this->shape().as_vector();
    auto& new_shape_vec = new_shape.as_vector();

    int D = (int)new_shape_vec.size() - (int)curr_shape_vec.size();
    if (D < 0) {
        SGL_THROW("Broadcast shape must be larger than tensor shape");
    }

    for (size_t i = 0; i < curr_shape_vec.size(); ++i) {
        if (curr_shape_vec[i] != new_shape_vec[D + i] && curr_shape_vec[i] != 1) {
            SGL_THROW(
                "Current dimension {} at index {} must be equal to new dimension {} or 1",
                curr_shape_vec[i],
                i,
                new_shape_vec[D + i]
            );
        }
    }

    auto& curr_strides_vec = this->strides().as_vector();
    std::vector<int> new_strides(new_shape.size(), 0);
    for (size_t i = 0; i < curr_strides_vec.size(); ++i) {
        if (curr_shape_vec[i] > 1) {
            new_strides[D + i] = curr_strides_vec[i];
        }
    }

    view_inplace(new_shape, Shape(new_strides));
}

void StridedBufferView::index_inplace(nb::object index_arg)
{
    // This implements python indexing (i.e. __getitem__)
    // Like numpy or torch, this supports a number of different ways of indexing:
    // - Indexing with a positive index (e.g. buffer[3, 2])
    // - Indexing with a negative index, for 'from the end' indexing (e.g. buffer[-1])
    // - Indexing with a slice (e.g. buffer[3:], buffer[:-3], buffer[::2])
    // - Inserting singleton dimensions (e.g. buffer[3, None, 2])
    // - Skipping dimensions with ellipsis (e.g. buffer[..., 3])
    //
    // A buffer may be partially indexed. E.g. for a 2D buffer of shape (64, 32),
    // doing buffer[5] is valid and will return a 1D buffer of shape (32, ) that is
    // the 1D slice of the full 2D buffer at index 5

    // We might receive a single argument (e.g. tensor[4]) or a tuple (e.g. tensor[3, 4])
    // in case of multiple indices. Unpack into a consistent argument vector.
    std::vector<nb::handle> args;
    if (nb::isinstance<nb::tuple>(index_arg)) {
        nb::tuple t = nb::cast<nb::tuple>(index_arg);
        args.insert(args.end(), t.begin(), t.end());
    } else {
        args.push_back(index_arg);
    }


    // Step 1: Figure out the number of 'real' indices, i.e. indices that
    // access an existing dimension, as opposed to inserting/skipping them
    // This applies to integers and slices
    int real_dims = 0;
    for (auto v : args) {
        if (nb::isinstance<int>(v) || nb::isinstance<nb::slice>(v))
            real_dims++;
    }
    SGL_CHECK(real_dims <= dims(), "Too many indices for buffer of dimension {}", dims());

    auto cur_shape = shape().as_vector();
    auto cur_strides = strides().as_vector();

    // This is the next dimension to be indexed by a 'real' index
    int dim = 0;
    // Offset (in elements) to be applied by the indexing operation
    int offset = 0;
    // shape and strides of the output of the indexing operation
    std::vector<int> shape, strides;

    for (size_t i = 0; i < args.size(); ++i) {
        const nb::handle& arg = args[i];

        if (nb::isinstance<int>(arg)) {
            // Integer index
            int idx = nb::cast<int>(arg);
            // First, do bounds checking
            if (idx < -cur_shape[dim] || idx >= cur_shape[dim])
                throw nb::index_error();

            // Next, wrap around negative indices
            if (idx < 0)
                idx += cur_shape[dim];
            // Finally, move offset forward by the index
            offset += idx * cur_strides[dim];
            // We indexed this dimension, so advance to the next one
            dim++;
        } else if (nb::isinstance<nb::slice>(arg)) {
            // Slice index
            nb::slice slice = nb::cast<nb::slice>(arg);

            // First, use .compute to apply slice to size of current dimension
            auto adjusted = slice.compute(cur_shape[dim]);
            size_t start = adjusted.get<0>();
            size_t stop = adjusted.get<1>();
            size_t step = adjusted.get<2>();
            size_t slice_length = adjusted.get<3>();

            // We only support positive steps
            SGL_CHECK(step > 0, "Slice step must be greater than zero (found stride {} at dimension {})", step, i);

            // Move offset by start of the slice
            offset += int(start) * cur_strides[dim];
            // Adjust shape by the computed slice length
            shape.push_back(int(slice_length));
            // Finally, adjust strides to account for the slice step
            strides.push_back(int(step) * cur_strides[dim]);
            // We indexed this dimension, so advance to the next one
            dim++;
        } else if (nb::isinstance<nb::ellipsis>(arg)) {
            // The ellipsis (...) skips past all unindexed dimensions
            // This is the number of dimensions of this buffer, minus the
            // number of dimensions indexed
            int eta = dims() - real_dims;
            // The skipped dimensions are directly appended to the output shape/strides
            for (int j = 0; j < eta; ++j) {
                shape.push_back(cur_shape[dim + j]);
                strides.push_back(cur_strides[dim + j]);
            }
            // Advance past the skipped dimensions
            dim += eta;
        } else if (arg.is_none()) {
            // Singleton dimensions are just dimensions of size 1 and stride 0.
            // Insert it to the output
            shape.push_back(1);
            strides.push_back(0);
        } else {
            auto type_name = nb::str(arg.type());
            SGL_THROW(
                "Illegal argument at dimension {}: Allowed are int, slice, ..., or None; found {} instead",
                i,
                type_name.c_str()
            );
        }
    }

    // Any remaining unindexed dimensions can now be appended to the output
    int remaining = dims() - dim;
    for (int j = 0; j < remaining; ++j) {
        shape.push_back(cur_shape[dim + j]);
        strides.push_back(cur_strides[dim + j]);
    }

    if (shape.empty()) {
        // A fully indexed buffer technically returns a 0D buffer
        // This is not really compatible with the rest of the machinery,
        // so turn it into a 1D buffer with 1 element instead
        shape.push_back(1);
        strides.push_back(1);
    }

    // Finally, change our view to the new shape/strides/offset
    view_inplace(Shape(shape), Shape(strides), offset);
}

void StridedBufferView::clear(CommandEncoder* cmd)
{
    if (cmd) {
        cmd->clear_buffer(m_storage);
    } else {
        ref<CommandEncoder> temp_cmd = device()->create_command_encoder();
        temp_cmd->clear_buffer(m_storage);
        device()->submit_command_buffer(temp_cmd->finish());
    }
}

template<typename Framework>
static nb::ndarray<Framework> to_ndarray(void* data, nb::handle owner, const StridedBufferViewDesc& desc)
{
    // Get dlpack type from scalar type.
    size_t dtype_size = desc.element_layout->stride();
    ref<NativeSlangType> innermost = innermost_type(desc.dtype);
    ref<TypeLayoutReflection> innermost_layout = innermost->buffer_type_layout();

    // If the buffer data type (after unwrapping all arrays/vectors) is a scalar,
    // we can map directly to an array of that scalar type.
    // Otherwise, we turn it into an array of bytes and add one more dimension
    // to index the bytes of the element.
    // Examples:
    //      Buffer with shape (4, 5) of float3 -> ndarray of shape (4, 5, 3) and dtype float32
    //      Buffer with shape (5, ) of struct Foo { ... } -> ndarray of shape (5, sizeof(Foo)) and dtype uint8
    bool is_scalar = innermost_layout->type()->kind() == TypeReflection::Kind::scalar;
    auto dtype_shape = desc.dtype->get_shape();
    auto dtype_strides = dtype_shape.calc_contiguous_strides();

    size_t innermost_size = is_scalar ? innermost_layout->stride() : 1;
    TypeReflection::ScalarType scalar_type
        = is_scalar ? innermost_layout->type()->scalar_type() : TypeReflection::ScalarType::uint8;
    auto dlpack_type = scalartype_to_dtype(scalar_type);

    // Build sizes/strides arrays in form numpy wants them.
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
    // If the innermost dtype is not a scalar, add one innermost dimension over
    // the bytes of the element
    if (!is_scalar) {
        sizes.push_back(innermost_layout->stride());
        strides.push_back(1);
    }

    auto device = Framework::value == nb::pytorch::value ? nb::device::cuda::value : nb::device::cpu::value;

    // Return numpy array.
    return nb::ndarray<Framework>(data, sizes.size(), sizes.data(), owner, strides.data(), *dlpack_type, device);
}

nb::ndarray<nb::numpy> StridedBufferView::to_numpy() const
{
    // Create data and nanobind capsule to contain the data.
    size_t dtype_size = desc().element_layout->stride();
    size_t byte_offset = desc().offset * dtype_size;
    size_t data_size = m_storage->size() - byte_offset;
    void* data = new uint8_t[data_size];
    m_storage->get_data(data, data_size, byte_offset);
    nb::capsule owner(data, [](void* p) noexcept { delete[] reinterpret_cast<uint8_t*>(p); });

    return to_ndarray<nb::numpy>(data, owner, desc());
}

nb::ndarray<nb::pytorch> StridedBufferView::to_torch() const
{
    // Map CUDA memory and pass to nanobind ndarray
    size_t dtype_size = desc().element_layout->stride();
    size_t byte_offset = desc().offset * dtype_size;
    void* data = reinterpret_cast<uint8_t*>(m_storage->cuda_memory()) + byte_offset;

    // TODO: We would ideally use m_storage itself as the owner of the ndarray data
    // However, if the buffer was created in cpp rather than python (which is true for NDBuffer),
    // nb::find won't work on it. For now, use the buffer view as the owner of the data,
    // which will keep m_storage alive
    // auto owner = nb::find(m_storage.get());
    auto owner = nb::find(this);

    return to_ndarray<nb::pytorch>(data, owner, desc());
}

bool StridedBufferView::maybe_pad_data(nb::ndarray<nb::numpy> data, size_t dtype_size, size_t byte_offset)
{
    // for vector types, we need special handling, because if the stride is not the same as the GPU requirement,
    // we will need the padding for each element.
    if (desc().element_layout->kind() == TypeReflection::Kind::vector) {
        // scalar_size is the size of the element of the vector type
        // dtype_size is the size of the aligned vector type
        size_t scalar_size = desc().element_layout->element_type_layout()->size();
        size_t required_element_num = dtype_size / scalar_size;
        size_t actual_element_num = data.shape(data.ndim() - 1);
        // If the actual element number is less than the required element number, then we need to pad the data
        // with zeros.
        // For simplicity, we use nanobind API to pad the data instead of writing our own padding logic.
        if (actual_element_num < required_element_num) {
            nb::object np = nb::module_::import_("numpy");
            size_t padding_element_num = required_element_num - actual_element_num;
            nb::list pad_width;
            // construct the pad_width list to specify the padding for each dimension, we only need to pad the
            // last dimension.
            for (size_t i = 0; i < data.ndim() - 1; i++) {
                pad_width.append(nb::make_tuple(0, 0)); // no padding for non-last dimension
            }
            // padding for last dimension
            pad_width.append(nb::make_tuple(0, padding_element_num));

            // pad the data with zeros
            nb::object arr_obj = nb::cast(data);
            nb::object padding_data = np.attr("pad")(arr_obj, pad_width, "constant_values"_a = 0);

            nb::ndarray<nb::numpy> out = nb::cast<nb::ndarray<nb::numpy>>(padding_data);
            size_t data_size = out.nbytes();
            m_storage->set_data(out.data(), data_size, byte_offset);
            return true;
        }
    }
    // TODO: handle the matrix case
    return false;
}

void StridedBufferView::copy_from_numpy(nb::ndarray<nb::numpy> data)
{
    SGL_CHECK(is_ndarray_contiguous(data), "Source Numpy array must be contiguous");
    SGL_CHECK(is_contiguous(), "Destination buffer view must be contiguous");

    size_t dtype_size = desc().element_layout->stride();
    size_t byte_offset = desc().offset * dtype_size;
    size_t data_size = data.nbytes();
    size_t buffer_size = m_storage->size() - byte_offset;
    SGL_CHECK(data_size <= buffer_size, "Numpy array is larger than the buffer ({} > {})", data_size, buffer_size);

    if (maybe_pad_data(data, dtype_size, byte_offset)) {
        return;
    }

    m_storage->set_data(data.data(), data_size, byte_offset);
}

void StridedBufferView::copy_from_torch(nb::object tensor)
{
    // Check if tensor is on CUDA and buffer supports CUDA interop
    bool is_cuda = nb::cast<bool>(tensor.attr("is_cuda"));
    bool has_cuda_memory = m_storage->cuda_memory() != nullptr;

    if (is_cuda && has_cuda_memory) {
        // Add the same error checks as copy_from_numpy for consistency
        SGL_CHECK(is_contiguous(), "Destination buffer view must be contiguous");

        // Establish proper CUDA context scope
        SGL_CU_SCOPE(m_storage->device());

        // Extract tensor data
        nb::object contiguous_tensor = tensor.attr("contiguous")();
        nb::object data_ptr = contiguous_tensor.attr("data_ptr")();
        void* src_data = reinterpret_cast<void*>(nb::cast<uintptr_t>(data_ptr));

        size_t tensor_bytes = nb::cast<size_t>(contiguous_tensor.attr("numel")())
            * nb::cast<size_t>(contiguous_tensor.attr("element_size")());

        // Access buffer descriptor and calculate offsets
        const auto& buffer_desc = desc();
        size_t dtype_size = buffer_desc.element_layout->stride();
        size_t byte_offset = buffer_desc.offset * dtype_size;

        void* dst_data = reinterpret_cast<uint8_t*>(m_storage->cuda_memory()) + byte_offset;

        // Validate memory bounds - use accurate error message for direct tensor operations
        size_t buffer_size = m_storage->size() - byte_offset;
        SGL_CHECK(tensor_bytes <= buffer_size, "Tensor is larger than the buffer ({} > {})", tensor_bytes, buffer_size);

        // Use proper CUDA device-to-device memory copy
        sgl::cuda::memcpy_device_to_device(dst_data, src_data, tensor_bytes);
    } else {
        // CPU fallback for non-CUDA tensors or non-CUDA buffers
        nb::object numpy_array = tensor.attr("cpu")().attr("numpy")();
        nb::ndarray<nb::numpy> numpy_data = nb::cast<nb::ndarray<nb::numpy>>(numpy_array);
        copy_from_numpy(numpy_data);
    }
}

void StridedBufferView::point_to(ref<StridedBufferView> target)
{
    SGL_CHECK(shape() == target->shape(), "Shape of existing and new view must match");
    SGL_CHECK(usage() == target->usage(), "Usage flags of existing and new buffer must match");
    SGL_CHECK(memory_type() == target->memory_type(), "Memory type of existing and new buffer must match");
    SGL_CHECK(element_stride() == target->element_stride(), "Element size of new and existing data type must match");

    desc().offset = target->offset();
    desc().strides = target->strides();
    m_storage = target->m_storage;
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy_strided_buffer_view)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::class_<StridedBufferViewDesc>(slangpy, "StridedBufferViewDesc")
        .def(nb::init<>())
        .def_rw("dtype", &StridedBufferViewDesc::dtype)
        .def_rw("element_layout", &StridedBufferViewDesc::element_layout)
        .def_rw("offset", &StridedBufferViewDesc::offset)
        .def_rw("shape", &StridedBufferViewDesc::shape)
        .def_rw("strides", &StridedBufferViewDesc::strides)
        .def_rw("usage", &StridedBufferViewDesc::usage)
        .def_rw("memory_type", &StridedBufferViewDesc::memory_type);

    nb::class_<StridedBufferView, NativeObject>(slangpy, "StridedBufferView")
        .def(nb::init<ref<Device>, StridedBufferViewDesc, ref<Buffer>>())
        .def_prop_ro("device", &StridedBufferView::device)
        .def_prop_ro("dtype", &StridedBufferView::dtype)
        .def_prop_ro("offset", &StridedBufferView::offset)
        .def_prop_ro("shape", &StridedBufferView::shape)
        .def_prop_ro("strides", &StridedBufferView::strides)
        .def_prop_ro("element_count", &StridedBufferView::element_count)
        .def_prop_ro("usage", &StridedBufferView::usage)
        .def_prop_ro("memory_type", &StridedBufferView::memory_type)
        .def_prop_ro("storage", &StridedBufferView::storage)
        .def("clear", &StridedBufferView::clear, "cmd"_a.none() = nullptr)
        .def("cursor", &StridedBufferView::cursor, "start"_a.none() = std::nullopt, "count"_a.none() = std::nullopt)
        .def("uniforms", &StridedBufferView::uniforms)
        .def("to_numpy", &StridedBufferView::to_numpy, D_NA(StridedBufferView, to_numpy))
        .def("to_torch", &StridedBufferView::to_torch, D_NA(StridedBufferView, to_torch))
        .def("copy_from_numpy", &StridedBufferView::copy_from_numpy, "data"_a, D_NA(StridedBufferView, copy_from_numpy))
        .def("copy_from_torch", &StridedBufferView::copy_from_torch, "tensor"_a)
        .def("is_contiguous", &StridedBufferView::is_contiguous, D_NA(&StridedBufferView, is_contiguous))
        .def("point_to", &StridedBufferView::point_to, "target"_a, D_NA(&StridedBufferView, point_to));
}
