// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/buffer_cursor.h"
#include "sgl/device/command.h"
#include "sgl/device/cuda_utils.h"
#include "sgl/device/device.h"
#include "sgl/func/base_struct.h"
#include "sgl/refl/lookup.h"
#include "utils/slangpytensor.h"

#include <fmt/format.h>

namespace sgl::slangpy {
namespace {

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
    nb::ndarray<Framework> to_ndarray(void* data, nb::handle owner, const TensorDesc& desc)
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

    bool numpy_flag(nb::object array, const char* name)
    {
        return nb::cast<bool>(array.attr("flags").attr("__getitem__")(name));
    }

    ref<refl::Layout> resolve_program_layout(Device* device, nb::object dtype, const ref<refl::Layout>& program_layout)
    {
        refl::Layout* explicit_layout = program_layout.get();

        refl::Type* type = nullptr;
        if (!dtype.is_none() && nb::try_cast<refl::Type*>(dtype, type))
            return refl::resolve_layout(device, type, explicit_layout);

        func::BaseStruct* struct_type = nullptr;
        if (!dtype.is_none() && nb::try_cast<func::BaseStruct*>(dtype, struct_type))
            return refl::resolve_layout(device, struct_type, explicit_layout);

        return refl::resolve_layout(device, static_cast<const refl::Type*>(nullptr), explicit_layout);
    }

    nb::object layout_or_none(const ref<refl::Layout>& layout)
    {
        return layout ? nb::cast(layout) : nb::object(nb::none());
    }

    ref<refl::Type> resolve_dtype(Device* device, nb::object dtype, const ref<refl::Layout>& program_layout)
    {
        if (dtype.is_none())
            return nullptr;

        ref<refl::Layout> layout = resolve_program_layout(device, dtype, program_layout);

        refl::Type* type = nullptr;
        if (nb::try_cast<refl::Type*>(dtype, type))
            return refl::resolve_element_type(layout.get(), type);

        const TypeReflection* type_reflection = nullptr;
        if (nb::try_cast<const TypeReflection*>(dtype, type_reflection))
            return refl::resolve_element_type(layout.get(), type_reflection);

        const TypeLayoutReflection* type_layout_reflection = nullptr;
        if (nb::try_cast<const TypeLayoutReflection*>(dtype, type_layout_reflection))
            return refl::resolve_element_type(layout.get(), type_layout_reflection);

        func::BaseStruct* struct_type = nullptr;
        if (nb::try_cast<func::BaseStruct*>(dtype, struct_type))
            return refl::resolve_element_type(layout.get(), struct_type);

        nb::str name;
        if (nb::try_cast<nb::str>(dtype, name))
            return refl::resolve_element_type(layout.get(), nb::cast<std::string>(name));

        nb::object lookup = nb::module_::import_("slangpy.reflection.lookup");
        return nb::cast<ref<refl::Type>>(lookup.attr("resolve_element_type")(nb::cast(layout), dtype));
    }

    ref<refl::Type> numpy_to_slang(nb::object dtype, Device* device, const ref<refl::Layout>& program_layout)
    {
        nb::object lookup = nb::module_::import_("slangpy.reflection.lookup");
        nb::object result = lookup.attr("numpy_to_slang")(dtype, nb::cast(device), layout_or_none(program_layout));
        if (result.is_none())
            return nullptr;
        return nb::cast<ref<refl::Type>>(result);
    }

    TensorDesc make_tensor_desc(ref<Buffer> storage, ref<refl::Type> dtype, Shape shape, Shape strides, int offset)
    {
        if (!strides.valid() || strides.size() == 0)
            strides = shape.calc_contiguous_strides();
        if (strides.size() != shape.size())
            throw nb::value_error("Number of strides must match number of dimensions");

        TensorDesc desc;
        desc.shape = std::move(shape);
        desc.strides = std::move(strides);
        desc.offset = offset;
        desc.dtype = std::move(dtype);
        desc.element_layout = desc.dtype->buffer_type_layout();
        desc.usage = storage->desc().usage;
        desc.memory_type = storage->desc().memory_type;
        return desc;
    }

    nb::dict tensor_uniforms_impl(const Tensor& tensor)
    {
        nb::dict res;
        res["_data"] = tensor.storage();
        res["_shape"] = shape_to_list(tensor.shape());
        res["_offset"] = tensor.offset();
        res["_strides"] = shape_to_list(tensor.strides());
        return res;
    }

    nb::ndarray<nb::numpy> tensor_to_numpy(const Tensor& tensor)
    {
        const TensorDesc& desc = tensor.desc();
        size_t dtype_size = desc.element_layout->stride();
        size_t byte_offset = desc.offset * dtype_size;
        size_t data_size = tensor.storage()->size() - byte_offset;
        void* data = new uint8_t[data_size];
        tensor.storage()->get_data(data, data_size, byte_offset);
        nb::capsule owner(
            data,
            [](void* p) noexcept
            {
                delete[] reinterpret_cast<uint8_t*>(p);
            }
        );

        return to_ndarray<nb::numpy>(data, owner, desc);
    }

    nb::ndarray<nb::pytorch> tensor_to_torch(const Tensor& tensor)
    {
        const TensorDesc& desc = tensor.desc();
        size_t dtype_size = desc.element_layout->stride();
        size_t byte_offset = desc.offset * dtype_size;
        void* data = reinterpret_cast<uint8_t*>(tensor.storage()->cuda_memory()) + byte_offset;
        nb::handle owner = nb::find(&tensor);

        return to_ndarray<nb::pytorch>(data, owner, desc);
    }

    bool tensor_maybe_pad_data(Tensor& tensor, nb::ndarray<nb::numpy> data, size_t dtype_size, size_t byte_offset)
    {
        const TensorDesc& desc = tensor.desc();
        if (desc.element_layout->kind() == TypeReflection::Kind::vector) {
            size_t scalar_size = desc.element_layout->element_type_layout()->size();
            size_t required_element_num = dtype_size / scalar_size;
            size_t actual_element_num = data.shape(data.ndim() - 1);
            if (actual_element_num < required_element_num) {
                nb::object np = nb::module_::import_("numpy");
                size_t padding_element_num = required_element_num - actual_element_num;
                nb::list pad_width;
                for (size_t i = 0; i < data.ndim() - 1; i++)
                    pad_width.append(nb::make_tuple(0, 0));
                pad_width.append(nb::make_tuple(0, padding_element_num));

                nb::object padding_data = np.attr("pad")(nb::cast(data), pad_width, "constant_values"_a = 0);
                nb::ndarray<nb::numpy> out = nb::cast<nb::ndarray<nb::numpy>>(padding_data);
                tensor.storage()->set_data(out.data(), out.nbytes(), byte_offset);
                return true;
            }
        }
        return false;
    }

    void tensor_copy_from_numpy(Tensor& tensor, nb::ndarray<nb::numpy> data)
    {
        SGL_CHECK(is_ndarray_contiguous(data), "Source Numpy array must be contiguous");
        SGL_CHECK(tensor.is_contiguous(), "Destination tensor must be contiguous");

        const TensorDesc& desc = tensor.desc();
        size_t dtype_size = desc.element_layout->stride();
        size_t byte_offset = desc.offset * dtype_size;
        size_t data_size = data.nbytes();
        size_t buffer_size = tensor.storage()->size() - byte_offset;
        SGL_CHECK(
            data_size <= buffer_size,
            "Numpy array is larger than the tensor storage ({} > {})",
            data_size,
            buffer_size
        );

        if (tensor_maybe_pad_data(tensor, data, dtype_size, byte_offset))
            return;

        tensor.storage()->set_data(data.data(), data_size, byte_offset);
    }

    void tensor_copy_from_torch(Tensor& tensor, nb::object torch_tensor)
    {
        bool is_cuda = nb::cast<bool>(torch_tensor.attr("is_cuda"));
        bool has_cuda_memory = tensor.storage()->cuda_memory() != nullptr;

        if (is_cuda && has_cuda_memory) {
            SGL_CHECK(tensor.is_contiguous(), "Destination tensor must be contiguous");

            nb::object contiguous_tensor = torch_tensor.attr("contiguous")();
            nb::object data_ptr = contiguous_tensor.attr("data_ptr")();
            void* src_data = reinterpret_cast<void*>(nb::cast<uintptr_t>(data_ptr));

            size_t tensor_bytes = nb::cast<size_t>(contiguous_tensor.attr("numel")())
                * nb::cast<size_t>(contiguous_tensor.attr("element_size")());

            const TensorDesc& desc = tensor.desc();
            size_t dtype_size = desc.element_layout->stride();
            size_t byte_offset = desc.offset * dtype_size;
            size_t buffer_size = tensor.storage()->size() - byte_offset;
            SGL_CHECK(
                tensor_bytes <= buffer_size,
                "Tensor is larger than the tensor storage ({} > {})",
                tensor_bytes,
                buffer_size
            );

            SGL_CU_SCOPE(tensor.storage()->device());

            void* dst_data = reinterpret_cast<uint8_t*>(tensor.storage()->cuda_memory()) + byte_offset;
            sgl::cuda::memcpy_device_to_device(dst_data, src_data, tensor_bytes);
        } else {
            nb::object numpy_array = torch_tensor.attr("cpu")().attr("numpy")();
            tensor_copy_from_numpy(tensor, nb::cast<nb::ndarray<nb::numpy>>(numpy_array));
        }
    }

    ref<Tensor> tensor_index(const ref<Tensor>& tensor, nb::object index_arg)
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
        SGL_CHECK(real_dims <= tensor->dims(), "Too many indices for tensor of dimension {}", tensor->dims());

        const Shape& cur_shape = tensor->shape();
        const Shape& cur_strides = tensor->strides();

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
                int skipped_dims = tensor->dims() - real_dims;
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

        int remaining = tensor->dims() - dim;
        for (int j = 0; j < remaining; ++j) {
            out_shape.push_back(cur_shape[dim + j]);
            out_strides.push_back(cur_strides[dim + j]);
        }

        if (out_shape.empty()) {
            out_shape.push_back(1);
            out_strides.push_back(1);
        }

        return tensor->view(Shape(out_shape), Shape(out_strides), offset);
    }

    ref<Tensor> tensor_empty(
        Device* device,
        Shape shape,
        nb::object dtype_obj,
        BufferUsage usage,
        MemoryType memory_type,
        ref<refl::Layout> program_layout
    )
    {
        ref<refl::Type> dtype = resolve_dtype(device, dtype_obj, program_layout);
        if (!dtype)
            throw nb::value_error("Element type (dtype) must be specified");
        if (shape.size() == 0)
            throw nb::value_error("Cannot create a tensor with zero dimensions");

        BufferDesc buffer_desc;
        buffer_desc.element_count = shape.element_count();
        buffer_desc.struct_size = dtype->buffer_layout()->stride();
        buffer_desc.usage = usage;
        buffer_desc.memory_type = memory_type;
        ref<Buffer> buffer = device->create_buffer(buffer_desc);

        TensorDesc desc;
        desc.dtype = dtype;
        desc.element_layout = dtype->buffer_type_layout();
        desc.shape = shape;
        desc.strides = shape.calc_contiguous_strides();
        desc.offset = 0;
        desc.usage = usage;
        desc.memory_type = memory_type;
        return make_ref<Tensor>(desc, buffer, nullptr, nullptr);
    }

    ref<Tensor> tensor_zeros(
        Device* device,
        Shape shape,
        nb::object dtype,
        BufferUsage usage,
        MemoryType memory_type,
        ref<refl::Layout> program_layout
    )
    {
        ref<Tensor> tensor
            = tensor_empty(device, std::move(shape), dtype, usage, memory_type, std::move(program_layout));
        tensor->clear();
        return tensor;
    }

    ref<Tensor> tensor_empty_like(const Tensor& other)
    {
        return tensor_empty(
            other.storage()->device(),
            other.shape(),
            nb::cast(other.dtype()),
            other.usage(),
            other.memory_type(),
            nullptr
        );
    }

    ref<Tensor> tensor_zeros_like(const Tensor& other)
    {
        return tensor_zeros(
            other.storage()->device(),
            other.shape(),
            nb::cast(other.dtype()),
            other.usage(),
            other.memory_type(),
            nullptr
        );
    }

    ref<Tensor> tensor_from_numpy(
        Device* device,
        nb::object ndarray_obj,
        BufferUsage usage,
        MemoryType memory_type,
        ref<refl::Layout> program_layout,
        nb::object target_slang_dtype
    )
    {
        nb::object dtype_obj = ndarray_obj.attr("dtype");
        bool is_structured = !dtype_obj.attr("names").is_none();
        size_t item_size = nb::cast<size_t>(dtype_obj.attr("itemsize"));
        size_t nbytes = nb::cast<size_t>(ndarray_obj.attr("nbytes"));
        nb::object shape_obj = ndarray_obj.attr("shape");
        nb::object strides_obj = ndarray_obj.attr("strides");
        size_t ndim = nb::len(shape_obj);

        ref<refl::Type> dtype;
        if (target_slang_dtype.is_none()) {
            if (is_structured) {
                throw nb::value_error(
                    fmt::format(
                        "Structured numpy dtype {} cannot be automatically mapped to a Slang type. Please provide "
                        "an explicit target_slang_dtype, e.g.:\n  Tensor.from_numpy(device, data, "
                        "target_slang_dtype=module.MyStructType)",
                        nb::cast<std::string>(nb::str(dtype_obj))
                    )
                        .c_str()
                );
            }
            dtype = numpy_to_slang(dtype_obj, device, program_layout);
            if (!dtype)
                throw nb::value_error(
                    fmt::format("Unsupported numpy dtype {}", nb::cast<std::string>(nb::str(dtype_obj))).c_str()
                );
        } else {
            dtype = resolve_dtype(device, target_slang_dtype, program_layout);
        }

        nb::object data_obj;
        if (is_structured) {
            if (!numpy_flag(ndarray_obj, "C_CONTIGUOUS")) {
                throw nb::value_error(
                    "Structured numpy arrays must be C-contiguous for Tensor.from_numpy. "
                    "Call numpy.ascontiguousarray(data) first."
                );
            }
            size_t slang_stride = dtype->buffer_layout()->stride();
            if (item_size != slang_stride) {
                throw nb::value_error(
                    fmt::format(
                        "Numpy structured dtype itemsize ({}) does not match Slang type '{}' buffer stride ({}). "
                        "Ensure the numpy dtype layout matches the Slang struct layout.",
                        item_size,
                        dtype->full_name(),
                        slang_stride
                    )
                        .c_str()
                );
            }
            nb::object np = nb::module_::import_("numpy");
            data_obj = np.attr("frombuffer")(ndarray_obj, "dtype"_a = np.attr("uint8"));
        } else {
            if ((nbytes % item_size) != 0)
                throw nb::value_error("Unsupported numpy array");
            for (size_t i = 0; i < ndim; ++i) {
                int64_t stride_bytes = nb::cast<int64_t>(strides_obj.attr("__getitem__")(i));
                if ((stride_bytes % static_cast<int64_t>(item_size)) != 0)
                    throw nb::value_error("Unsupported numpy array");
            }
            nb::object np = nb::module_::import_("numpy");
            data_obj
                = np.attr("lib")
                      .attr("stride_tricks")
                      .attr("as_strided")(ndarray_obj, nb::make_tuple(nbytes / item_size), nb::make_tuple(item_size));
        }

        nb::ndarray<nb::numpy> data = nb::cast<nb::ndarray<nb::numpy>>(data_obj);
        size_t element_count = nbytes / item_size;

        BufferDesc buffer_desc;
        buffer_desc.struct_size = item_size;
        buffer_desc.element_count = element_count;
        buffer_desc.usage = usage;
        buffer_desc.memory_type = memory_type;
        buffer_desc.data = data.data();
        buffer_desc.data_size = data.nbytes();
        ref<Buffer> buffer = device->create_buffer(buffer_desc);

        Shape shape(ndim);
        Shape strides(ndim);
        int* shape_data = shape.data();
        int* strides_data = strides.data();
        for (size_t i = 0; i < ndim; ++i) {
            shape_data[i] = nb::cast<int>(shape_obj.attr("__getitem__")(i));
            strides_data[i] = static_cast<int>(
                nb::cast<int64_t>(strides_obj.attr("__getitem__")(i)) / static_cast<int64_t>(item_size)
            );
        }

        TensorDesc desc;
        desc.dtype = dtype;
        desc.element_layout = dtype->buffer_type_layout();
        desc.shape = shape;
        desc.strides = strides;
        desc.offset = 0;
        desc.usage = usage;
        desc.memory_type = memory_type;
        return make_ref<Tensor>(desc, buffer, nullptr, nullptr);
    }

    ref<Tensor> tensor_from_torch(
        Device* device,
        nb::object torch_tensor,
        nb::object dtype_obj,
        BufferUsage usage,
        ref<refl::Layout> program_layout
    )
    {
        ref<refl::Type> dtype = resolve_dtype(device, dtype_obj, program_layout);
        size_t struct_stride = dtype->buffer_layout()->stride();
        size_t scalar_size = nb::cast<size_t>(torch_tensor.attr("element_size")());
        if (struct_stride == 0 || scalar_size == 0 || struct_stride % scalar_size != 0) {
            throw nb::value_error(
                fmt::format(
                    "Torch element size ({}) is not compatible with Slang type '{}' buffer stride ({})",
                    scalar_size,
                    dtype->full_name(),
                    struct_stride
                )
                    .c_str()
            );
        }

        size_t scalars_per_element = struct_stride / scalar_size;
        int dims = nb::cast<int>(torch_tensor.attr("dim")());
        if (dims < 1)
            throw nb::value_error("Tensor must have at least 1 dimension");
        nb::object torch_shape = torch_tensor.attr("shape");
        size_t last_dim_size = nb::cast<size_t>(torch_shape.attr("__getitem__")(dims - 1));
        if (last_dim_size != scalars_per_element) {
            throw nb::value_error(
                fmt::format(
                    "Last dimension size ({}) does not match the number of scalars per '{}' element ({})",
                    last_dim_size,
                    dtype->full_name(),
                    scalars_per_element
                )
                    .c_str()
            );
        }
        if (nb::cast<int64_t>(torch_tensor.attr("stride")(dims - 1)) != 1)
            throw nb::value_error("Last dimension of the tensor must be contiguous");

        nb::object contiguous = torch_tensor.attr("contiguous")();
        size_t element_count = nb::cast<size_t>(contiguous.attr("numel")());

        BufferDesc buffer_desc;
        buffer_desc.size = element_count * scalar_size;
        buffer_desc.struct_size = struct_stride;
        buffer_desc.usage = usage | BufferUsage::shared;
        ref<Buffer> buffer = device->create_buffer(buffer_desc);

        nb::module_::import_("slangpy").attr("copy_torch_tensor_to_buffer")(contiguous, buffer);

        Shape shape(dims - 1);
        int* shape_data = shape.data();
        for (int i = 0; i < dims - 1; ++i)
            shape_data[i] = nb::cast<int>(contiguous.attr("shape").attr("__getitem__")(i));

        TensorDesc desc;
        desc.dtype = dtype;
        desc.element_layout = dtype->buffer_type_layout();
        desc.shape = shape;
        desc.strides = shape.calc_contiguous_strides();
        desc.offset = 0;
        desc.usage = buffer_desc.usage;
        desc.memory_type = buffer_desc.memory_type;
        return make_ref<Tensor>(desc, buffer, nullptr, nullptr);
    }

    ref<Tensor> tensor_load_from_image(
        Device* device,
        nb::object path,
        bool flip_y,
        bool linearize,
        float scale,
        float offset,
        bool grayscale
    )
    {
        nb::object data_obj
            = nb::module_::import_("slangpy.types.common")
                  .attr("load_buffer_data_from_image")(path, flip_y, linearize, scale, offset, grayscale);
        nb::ndarray<nb::numpy> data = nb::cast<nb::ndarray<nb::numpy>>(data_obj);

        std::string dtype;
        if (data.ndim() == 2 || data.shape(2) == 1)
            dtype = "float";
        else if (data.shape(2) == 2)
            dtype = "float2";
        else if (data.shape(2) == 3)
            dtype = "float3";
        else if (data.shape(2) == 4)
            dtype = "float4";
        else
            throw nb::value_error(fmt::format("Unsupported number of channels: {}", data.shape(2)).c_str());

        Shape shape({static_cast<int>(data.shape(0)), static_cast<int>(data.shape(1))});
        ref<Tensor> tensor = tensor_empty(
            device,
            std::move(shape),
            nb::cast(dtype),
            BufferUsage::shader_resource | BufferUsage::unordered_access,
            MemoryType::device_local,
            nullptr
        );
        tensor_copy_from_numpy(*tensor, data);
        return tensor;
    }

} // namespace

nb::dict tensor_uniforms(const Tensor& tensor)
{
    return tensor_uniforms_impl(tensor);
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(func_tensor)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ native_func = nb::module_::import_("slangpy.native_func");

    nb::class_<TensorDesc> tensor_desc(native_func, "TensorDesc");
    tensor_desc.def(nb::init<>())
        .def_rw("dtype", &TensorDesc::dtype)
        .def_rw("element_layout", &TensorDesc::element_layout)
        .def_rw("offset", &TensorDesc::offset)
        .def_rw("shape", &TensorDesc::shape)
        .def_rw("strides", &TensorDesc::strides)
        .def_rw("usage", &TensorDesc::usage)
        .def_rw("memory_type", &TensorDesc::memory_type);

    nb::class_<Tensor, Object> tensor(native_func, "Tensor");
    tensor
        .def(
            "__init__",
            [](Tensor* self,
               ref<Buffer> storage,
               ref<refl::Type> dtype,
               Shape shape,
               Shape strides,
               int offset,
               ref<Tensor> grad_in,
               ref<Tensor> grad_out)
            {
                TensorDesc desc
                    = make_tensor_desc(storage, std::move(dtype), std::move(shape), std::move(strides), offset);
                new (self) Tensor(std::move(desc), std::move(storage), std::move(grad_in), std::move(grad_out));
            },
            "storage"_a,
            "dtype"_a,
            "shape"_a,
            "strides"_a = Shape(),
            "offset"_a = 0,
            "grad_in"_a.none() = nullptr,
            "grad_out"_a.none() = nullptr
        )
        .def(
            "__init__",
            [](Tensor* self, TensorDesc desc, ref<Buffer> storage, ref<Tensor> grad_in, ref<Tensor> grad_out)
            {
                new (self) Tensor(std::move(desc), std::move(storage), std::move(grad_in), std::move(grad_out));
            },
            "desc"_a,
            "storage"_a,
            "grad_in"_a.none() = nullptr,
            "grad_out"_a.none() = nullptr
        )
        .def_prop_ro("device", &Tensor::device)
        .def_prop_ro("dtype", &Tensor::dtype)
        .def_prop_ro("offset", &Tensor::offset)
        .def_prop_ro("shape", &Tensor::shape)
        .def_prop_ro("strides", &Tensor::strides)
        .def_prop_ro("element_count", &Tensor::element_count)
        .def_prop_ro("usage", &Tensor::usage)
        .def_prop_ro("memory_type", &Tensor::memory_type)
        .def_prop_ro("storage", &Tensor::storage)
        .def_prop_rw("grad_in", &Tensor::grad_in, &Tensor::set_grad_in, nb::none())
        .def_prop_rw("grad_out", &Tensor::grad_out, &Tensor::set_grad_out, nb::none())
        .def_prop_ro("grad", &Tensor::grad)
        .def("clear", &Tensor::clear, "cmd"_a.none() = nullptr)
        .def("cursor", &Tensor::cursor, "start"_a.none() = std::nullopt, "count"_a.none() = std::nullopt)
        .def("uniforms", &tensor_uniforms)
        .def("to_numpy", &tensor_to_numpy)
        .def("to_torch", &tensor_to_torch)
        .def("copy_from_numpy", &tensor_copy_from_numpy, "data"_a)
        .def("copy_from_torch", &tensor_copy_from_torch, "tensor"_a)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("point_to", &Tensor::point_to, "target"_a)
        .def("broadcast_to", &Tensor::broadcast_to, "shape"_a)
        .def("view", &Tensor::view, "shape"_a, "strides"_a = Shape(), "offset"_a = 0)
        .def("__getitem__", &tensor_index)
        .def(
            "with_grads",
            &Tensor::with_grads,
            "grad_in"_a.none() = nullptr,
            "grad_out"_a.none() = nullptr,
            "zero"_a = true
        )
        .def("detach", &Tensor::detach)
        .def(
            "__str__",
            [](const Tensor& self)
            {
                return nb::str(nb::cast(tensor_to_numpy(self)));
            }
        )
        .def("__repr__", &Tensor::to_string)
        .def_static(
            "numpy",
            [](Device* device, nb::object ndarray)
            {
                nb::module_::import_("warnings")
                    .attr("warn")(
                        "Tensor.numpy is deprecated. Use Tensor.from_numpy instead.",
                        nb::module_::import_("builtins").attr("DeprecationWarning"),
                        "stacklevel"_a = 2
                    );
                return tensor_from_numpy(
                    device,
                    ndarray,
                    BufferUsage::shader_resource | BufferUsage::unordered_access,
                    MemoryType::device_local,
                    nullptr,
                    nb::none()
                );
            },
            "device"_a,
            "ndarray"_a
        )
        .def_static(
            "from_numpy",
            &tensor_from_numpy,
            "device"_a,
            "ndarray"_a,
            "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
            "memory_type"_a = MemoryType::device_local,
            "program_layout"_a.none() = nullptr,
            "target_slang_dtype"_a.none() = nb::none()
        )
        .def_static(
            "empty",
            &tensor_empty,
            "device"_a,
            "shape"_a,
            "dtype"_a.none() = nb::none(),
            "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
            "memory_type"_a = MemoryType::device_local,
            "program_layout"_a.none() = nullptr
        )
        .def_static(
            "zeros",
            &tensor_zeros,
            "device"_a,
            "shape"_a,
            "dtype"_a,
            "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
            "memory_type"_a = MemoryType::device_local,
            "program_layout"_a.none() = nullptr
        )
        .def_static("empty_like", &tensor_empty_like, "other"_a)
        .def_static("zeros_like", &tensor_zeros_like, "other"_a)
        .def_static(
            "from_torch",
            &tensor_from_torch,
            "device"_a,
            "tensor"_a,
            "dtype"_a,
            "usage"_a = BufferUsage::shader_resource | BufferUsage::unordered_access,
            "program_layout"_a.none() = nullptr
        )
        .def_static(
            "load_from_image",
            &tensor_load_from_image,
            "device"_a,
            "path"_a,
            "flip_y"_a = false,
            "linearize"_a = false,
            "scale"_a = 1.0f,
            "offset"_a = 0.0f,
            "grayscale"_a = false
        );
}
