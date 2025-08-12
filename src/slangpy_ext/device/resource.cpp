// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/resource.h"
#include "sgl/device/device.h"
#include "sgl/device/formats.h"
#include "sgl/device/command.h"

#include "sgl/core/bitmap.h"

namespace sgl {

SGL_DICT_TO_DESC_BEGIN(BufferDesc)
SGL_DICT_TO_DESC_FIELD(size, size_t)
SGL_DICT_TO_DESC_FIELD(struct_size, size_t)
SGL_DICT_TO_DESC_FIELD(format, Format)
SGL_DICT_TO_DESC_FIELD(memory_type, MemoryType)
SGL_DICT_TO_DESC_FIELD(usage, BufferUsage)
SGL_DICT_TO_DESC_FIELD(default_state, ResourceState)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(buffer, Buffer*)
SGL_DICT_TO_DESC_FIELD(offset, DeviceOffset)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(TextureDesc)
SGL_DICT_TO_DESC_FIELD(type, TextureType)
SGL_DICT_TO_DESC_FIELD(format, Format)
SGL_DICT_TO_DESC_FIELD(width, uint32_t)
SGL_DICT_TO_DESC_FIELD(height, uint32_t)
SGL_DICT_TO_DESC_FIELD(depth, uint32_t)
SGL_DICT_TO_DESC_FIELD(array_length, uint32_t)
SGL_DICT_TO_DESC_FIELD(mip_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(sample_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(sample_quality, uint32_t)
SGL_DICT_TO_DESC_FIELD(memory_type, MemoryType)
SGL_DICT_TO_DESC_FIELD(usage, TextureUsage)
SGL_DICT_TO_DESC_FIELD(default_state, ResourceState)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(SubresourceRange)
SGL_DICT_TO_DESC_FIELD(layer, uint32_t)
SGL_DICT_TO_DESC_FIELD(layer_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(mip, uint32_t)
SGL_DICT_TO_DESC_FIELD(mip_count, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(TextureViewDesc)
SGL_DICT_TO_DESC_FIELD(format, Format)
SGL_DICT_TO_DESC_FIELD(aspect, TextureAspect)
SGL_DICT_TO_DESC_FIELD(subresource_range, SubresourceRange)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

inline std::optional<nb::dlpack::dtype> resource_format_to_dtype(Format format)
{
    const auto& info = get_format_info(format);

    // Undefined and compressed formats are not supported.
    if (format == Format::undefined || info.is_compressed)
        return {};

    // Formats with different bits per channel are not supported.
    if (!info.has_equal_channel_bits())
        return {};

    // Only formats with 8, 16, 32, or 64 bits per channel are supported.
    uint32_t channel_bit_count = info.channel_bit_count[0];
    if (channel_bit_count != 8 && channel_bit_count != 16 && channel_bit_count != 32 && channel_bit_count != 64)
        return {};

    switch (info.type) {
    case FormatType::float_:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Float, (uint8_t)channel_bit_count, 1};
    case FormatType::uint:
    case FormatType::unorm:
    case FormatType::unorm_srgb:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::UInt, (uint8_t)channel_bit_count, 1};
    case FormatType::sint:
    case FormatType::snorm:
        return nb::dlpack::dtype{(uint8_t)nb::dlpack::dtype_code::Int, (uint8_t)channel_bit_count, 1};
    default:
        return {};
    }
}

static const char* __doc_sgl_buffer_to_numpy = R"doc()doc";

nb::ndarray<nb::numpy> buffer_to_numpy(Buffer* self)
{
    size_t data_size = self->size();
    void* data = new uint8_t[data_size];

    self->get_data(data, data_size);

    nb::capsule owner(data, [](void* p) noexcept { delete[] reinterpret_cast<uint8_t*>(p); });

    if (auto dtype = resource_format_to_dtype(self->format())) {
        const FormatInfo& format_info = get_format_info(self->format());
        size_t channel_count = format_info.channel_count;
        size_t element_count = self->size() / format_info.bytes_per_block;
        if (channel_count == 1) {
            size_t shape[1] = {element_count};
            return nb::ndarray<nb::numpy>(data, 1, shape, owner, nullptr, *dtype, nb::device::cpu::value);
        } else {
            size_t shape[2] = {element_count, channel_count};
            return nb::ndarray<nb::numpy>(data, 2, shape, owner, nullptr, *dtype, nb::device::cpu::value);
        }
    } else {
        size_t shape[1] = {data_size};
        return nb::ndarray<nb::numpy>(data, 1, shape, owner, nullptr, nb::dtype<uint8_t>(), nb::device::cpu::value);
    }
}

static const char* __doc_sgl_buffer_from_numpy = R"doc()doc";

void buffer_copy_from_numpy(Buffer* self, nb::ndarray<nb::numpy> data)
{
    SGL_CHECK(is_ndarray_contiguous(data), "numpy array is not contiguous");

    size_t buffer_size = self->size();
    size_t data_size = data.nbytes();
    SGL_CHECK(data_size <= buffer_size, "numpy array is larger than the buffer ({} > {})", data_size, buffer_size);

    self->set_data(data.data(), data_size);
}

static const char* __doc_sgl_buffer_to_torch = R"doc()doc";

nb::ndarray<nb::pytorch, nb::device::cuda>
buffer_to_torch(Buffer* self, DataType type, std::vector<size_t> shape, std::vector<int64_t> strides, size_t offset)
{
    SGL_CHECK(offset < self->size(), "offset is out of bounds");
    void* data = reinterpret_cast<uint8_t*>(self->cuda_memory()) + offset;
    size_t data_size = self->size() - offset;

    nb::dlpack::dtype dtype = data_type_to_dtype(type);
    size_t element_size = dtype.bits / 8;

    if (shape.empty())
        shape.push_back(data_size / element_size);
    int64_t* strides_ptr = nullptr;
    if (!strides.empty()) {
        SGL_CHECK(strides.size() == shape.size(), "strides must have the same length as the shape");
        strides_ptr = strides.data();
    }

    size_t required_size = 1;
    for (size_t dim : shape)
        required_size *= dim;
    SGL_CHECK(data_size >= required_size * element_size, "requested shape exceeds the buffer size");

    return nb::ndarray<nb::pytorch, nb::device::cuda>(
        data,
        shape.size(),
        shape.data(),
        nb::find(self),
        strides.empty() ? nullptr : strides.data(),
        dtype
    );
}

static const char* __doc_sgl_texture_to_numpy = R"doc()doc";

/**
 * Python binding wrapper for returning the content of a texture as a numpy array.
 */
nb::ndarray<nb::numpy> texture_to_numpy(Texture* self, uint32_t layer, uint32_t mip)
{
    SGL_CHECK_LT(layer, self->layer_count());
    SGL_CHECK_LT(mip, self->mip_count());

    // Get the subresource data and corresponding layout. Depending on platform, these
    // may not be tightly packed.
    OwnedSubresourceData subresource_data = self->get_subresource_data(layer, mip);
    SubresourceLayout src_layout = self->get_subresource_layout(mip);
    SGL_ASSERT(subresource_data.size == src_layout.size_in_bytes);
    SGL_ASSERT(subresource_data.row_pitch == src_layout.row_pitch);
    SGL_ASSERT(subresource_data.slice_pitch == src_layout.slice_pitch);

    // Get tightly packed layout for numpy array.
    SubresourceLayout dst_layout = self->get_subresource_layout(mip, 1);

    // Double check assumptions about column (aka block) size and dimensions
    // being consistent regardless of packing are correct.
    SGL_ASSERT(src_layout.col_pitch == dst_layout.col_pitch);
    SGL_ASSERT(src_layout.row_count == dst_layout.row_count);
    SGL_ASSERT(src_layout.size.x == dst_layout.size.x);
    SGL_ASSERT(src_layout.size.y == dst_layout.size.y);
    SGL_ASSERT(src_layout.size.z == dst_layout.size.z);

    // Dest is tightly packed, so pitch must be <= source pitch.
    SGL_ASSERT(src_layout.row_pitch >= dst_layout.row_pitch);

    // TODO: Could have fast path here if layouts are the same, but for now its
    // better to stress-test the full copy operation.
    uint8_t* src_data = (uint8_t*)subresource_data.data;
    uint8_t* dst_data = new uint8_t[dst_layout.size_in_bytes];
    for (uint32_t slice_idx = 0; slice_idx < src_layout.size.z; slice_idx++) {
        uint8_t* src_slice = src_data + slice_idx * src_layout.slice_pitch;
        uint8_t* dst_slice = dst_data + slice_idx * dst_layout.slice_pitch;
        for (uint32_t row_idx = 0; row_idx < src_layout.row_count; row_idx++) {
            uint8_t* src_row = src_slice + row_idx * src_layout.row_pitch;
            uint8_t* dst_row = dst_slice + row_idx * dst_layout.row_pitch;
            std::memcpy(dst_row, src_row, dst_layout.row_pitch);
        }
    }

    size_t size = dst_layout.size_in_bytes;
    void* data = dst_data;
    uint3 mip_size = dst_layout.size;

    nb::capsule owner(data, [](void* p) noexcept { delete[] reinterpret_cast<uint8_t*>(p); });

    if (auto dtype = resource_format_to_dtype(self->format())) {

        // Select shape based on texture type.
        std::vector<size_t> shape;
        switch (self->type()) {
        case TextureType::texture_1d:
        case TextureType::texture_1d_array:
            shape = {size_t(mip_size.x)};
            break;
        case TextureType::texture_2d:
        case TextureType::texture_2d_array:
        case TextureType::texture_2d_ms:
        case TextureType::texture_2d_ms_array:
        case TextureType::texture_cube:
        case TextureType::texture_cube_array:
            shape = {size_t(mip_size.y), size_t(mip_size.x)};
            break;
        case TextureType::texture_3d:
            shape = {size_t(mip_size.z), size_t(mip_size.y), size_t(mip_size.x)};
            break;
        }

        // Get format info and check assumptions about matching format/layout.
        const FormatInfo& format_info = get_format_info(self->format());
        SGL_ASSERT(format_info.bytes_per_block == dst_layout.col_pitch);
        SGL_ASSERT(format_info.block_width == 1);
        SGL_ASSERT(format_info.block_height == 1);

        // Add extra dimension for multi-channel textures.
        uint32_t channel_count = format_info.channel_count;
        if (channel_count > 1)
            shape.push_back(channel_count);

        // Calculate contiguous strides.
        std::vector<int64_t> strides(shape.size());
        int64_t stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }

        return nb::ndarray<
            nb::numpy>(data, shape.size(), shape.data(), owner, strides.data(), *dtype, nb::device::cpu::value);
    } else {
        size_t shape[1] = {size};
        return nb::ndarray<nb::numpy>(data, 1, shape, owner, nullptr, nb::dtype<uint8_t>(), nb::device::cpu::value);
    }
}

static const char* __doc_sgl_texture_from_numpy = R"doc()doc";

SubresourceData
texture_build_subresource_data_for_upload(Texture* self, nb::ndarray<nb::numpy> data, uint32_t layer, uint32_t mip)
{
    SGL_CHECK(is_ndarray_contiguous(data), "numpy array is not contiguous");
    SGL_CHECK_LT(layer, self->layer_count());
    SGL_CHECK_LT(mip, self->mip_count());

    // Get the sub resource layout with 1B alignment so rows/slices are tightly packed.
    SubresourceLayout subresource_layout = self->get_subresource_layout(mip, 1);

    // Setup subresource data from the numpy array information.
    SubresourceData subresource_data{
        .data = data.data(),
        .size = data.nbytes(),
        .row_pitch = subresource_layout.row_pitch,
        .slice_pitch = subresource_layout.slice_pitch,
    };
    uint3 mip_size = subresource_layout.size;

    // Validate the numpy array shape.
    if (auto dtype = resource_format_to_dtype(self->format())) {
        std::vector<size_t> expected_shape;
        switch (self->type()) {
        case TextureType::texture_1d:
        case TextureType::texture_1d_array:
            expected_shape = {size_t(mip_size.x)};
            break;
        case TextureType::texture_2d:
        case TextureType::texture_2d_array:
        case TextureType::texture_2d_ms:
        case TextureType::texture_2d_ms_array:
        case TextureType::texture_cube:
        case TextureType::texture_cube_array:
            expected_shape = {size_t(mip_size.y), size_t(mip_size.x)};
            break;
        case TextureType::texture_3d:
            expected_shape = {size_t(mip_size.z), size_t(mip_size.y), size_t(mip_size.x)};
            break;
        }
        uint32_t channel_count = get_format_info(self->format()).channel_count;
        if (channel_count > 1)
            expected_shape.push_back(channel_count);

        SGL_CHECK(
            data.ndim() == expected_shape.size(),
            "numpy array has wrong number of dimensions (expected {})",
            expected_shape.size()
        );
        for (size_t i = 0; i < expected_shape.size(); ++i)
            SGL_CHECK(data.shape(i) == expected_shape[i], "numpy array has wrong shape (expected {})", expected_shape);
    }

    // Check numpy array size in bytes matches the sub resource layotu size in bytes.
    SGL_CHECK(data.nbytes() == subresource_layout.size_in_bytes, "numpy array doesn't match the subresource size");

    return subresource_data;
}

inline void texture_from_numpy(Texture* self, nb::ndarray<nb::numpy> data, uint32_t layer, uint32_t mip)
{
    // Validate and set up SubresourceData argument to point to the numpy array data.
    SubresourceData subresource_data = texture_build_subresource_data_for_upload(self, data, layer, mip);

    // Write numpy data to the texture.
    self->set_subresource_data(layer, mip, subresource_data);
}

} // namespace sgl

SGL_PY_EXPORT(device_resource)
{
    using namespace sgl;

    m.attr("ALL_LAYERS") = ALL_LAYERS;
    m.attr("ALL_MIPS") = ALL_MIPS;

    nb::sgl_enum<ResourceState>(m, "ResourceState");
    nb::sgl_enum_flags<BufferUsage>(m, "BufferUsage");
    nb::sgl_enum_flags<TextureUsage>(m, "TextureUsage");
    nb::sgl_enum<TextureType>(m, "TextureType");
    nb::sgl_enum<MemoryType>(m, "MemoryType");

    nb::class_<Resource, DeviceResource>(m, "Resource", D(Resource)) //
        .def_prop_ro("native_handle", &Resource::native_handle, D(Resource, native_handle));

    nb::sgl_enum<TextureAspect>(m, "TextureAspect");

    nb::class_<BufferRange>(m, "BufferRange", D(BufferRange))
        .def(nb::init<>())
        .def_ro("offset", &BufferRange::offset, D(BufferRange, offset))
        .def_ro("size", &BufferRange::size, D(BufferRange, size))
        .def("__repr__", &BufferRange::to_string, D(BufferRange, to_string));

    nb::class_<SubresourceRange>(m, "SubresourceRange", D(SubresourceRange))
        .def(nb::init<>())
        .def(
            "__init__",
            [](SubresourceRange* self, nb::dict dict) { new (self) SubresourceRange(dict_to_SubresourceRange(dict)); }
        )
        .def_rw("layer", &SubresourceRange::layer, D(SubresourceRange, layer))
        .def_rw("layer_count", &SubresourceRange::layer_count, D(SubresourceRange, layer_count))
        .def_rw("mip", &SubresourceRange::mip, D(SubresourceRange, mip))
        .def_rw("mip_count", &SubresourceRange::mip_count, D(SubresourceRange, mip_count))
        .def("__repr__", &SubresourceRange::to_string, D(SubresourceRange, to_string));

    nb::class_<BufferDesc>(m, "BufferDesc", D(BufferDesc))
        .def(nb::init<>())
        .def("__init__", [](BufferDesc* self, nb::dict dict) { new (self) BufferDesc(dict_to_BufferDesc(dict)); })
        .def_rw("size", &BufferDesc::size, D(BufferDesc, size))
        .def_rw("struct_size", &BufferDesc::struct_size, D(BufferDesc, struct_size))
        .def_rw("format", &BufferDesc::format, D(BufferDesc, format))
        .def_rw("memory_type", &BufferDesc::memory_type, D(BufferDesc, memory_type))
        .def_rw("usage", &BufferDesc::usage, D(BufferDesc, usage))
        .def_rw("default_state", &BufferDesc::default_state, D(BufferDesc, default_state))
        .def_rw("label", &BufferDesc::label, D(BufferDesc, label));
    nb::implicitly_convertible<nb::dict, BufferDesc>();

    nb::class_<Buffer, Resource>(m, "Buffer", D(Buffer))
        .def_prop_ro("desc", &Buffer::desc, D(Buffer, desc))
        .def_prop_ro("size", &Buffer::size, D(Buffer, size))
        .def_prop_ro("struct_size", &Buffer::struct_size, D(Buffer, struct_size))
        .def_prop_ro("device_address", &Buffer::device_address, D(Buffer, device_address))
        .def_prop_ro("shared_handle", &Buffer::shared_handle, D(Buffer, shared_handle))
        .def_prop_ro("descriptor_handle_ro", &Buffer::descriptor_handle_ro, D(Buffer, descriptor_handle_ro))
        .def_prop_ro("descriptor_handle_rw", &Buffer::descriptor_handle_rw, D(Buffer, descriptor_handle_rw))
        .def("to_numpy", &buffer_to_numpy, D(buffer_to_numpy))
        .def("copy_from_numpy", &buffer_copy_from_numpy, "data"_a, D(buffer_from_numpy))
        .def(
            "to_torch",
            &buffer_to_torch,
            "type"_a = DataType::void_,
            "shape"_a = std::vector<size_t>{},
            "strides"_a = std::vector<int64_t>{},
            "offset"_a = 0,
            D(buffer_to_torch)
        );

    nb::class_<BufferOffsetPair>(m, "BufferOffsetPair", D(BufferOffsetPair))
        .def(nb::init<>())
        .def(nb::init<Buffer*>(), "buffer"_a)
        .def(nb::init<Buffer*, DeviceOffset>(), "buffer"_a, "offset"_a = 0)
        .def(
            "__init__",
            [](BufferOffsetPair* self, nb::dict dict) { new (self) BufferOffsetPair(dict_to_BufferOffsetPair(dict)); }
        )
        .def_rw("buffer", &BufferOffsetPair::buffer, D(BufferOffsetPair, buffer))
        .def_rw("offset", &BufferOffsetPair::offset, D(BufferOffsetPair, offset));
    nb::implicitly_convertible<nb::dict, BufferOffsetPair>();
    nb::implicitly_convertible<Buffer, BufferOffsetPair>();

    nb::class_<TextureDesc>(m, "TextureDesc", D(TextureDesc))
        .def(nb::init<>())
        .def("__init__", [](TextureDesc* self, nb::dict dict) { new (self) TextureDesc(dict_to_TextureDesc(dict)); })
        .def_rw("type", &TextureDesc::type, D(TextureDesc, type))
        .def_rw("format", &TextureDesc::format, D(TextureDesc, format))
        .def_rw("width", &TextureDesc::width, D(TextureDesc, width))
        .def_rw("height", &TextureDesc::height, D(TextureDesc, height))
        .def_rw("depth", &TextureDesc::depth, D(TextureDesc, depth))
        .def_rw("array_length", &TextureDesc::array_length, D(TextureDesc, array_length))
        .def_rw("mip_count", &TextureDesc::mip_count, D(TextureDesc, mip_count))
        .def_rw("sample_count", &TextureDesc::sample_count, D(TextureDesc, sample_count))
        .def_rw("sample_quality", &TextureDesc::sample_quality, D(TextureDesc, sample_quality))
        .def_rw("memory_type", &TextureDesc::memory_type, D(TextureDesc, memory_type))
        .def_rw("usage", &TextureDesc::usage, D(TextureDesc, usage))
        .def_rw("default_state", &TextureDesc::default_state, D(TextureDesc, default_state))
        .def_rw("label", &TextureDesc::label, D(TextureDesc, label));
    nb::implicitly_convertible<nb::dict, TextureDesc>();

    nb::class_<TextureViewDesc>(m, "TextureViewDesc", D(TextureViewDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](TextureViewDesc* self, nb::dict dict) { new (self) TextureViewDesc(dict_to_TextureViewDesc(dict)); }
        )
        .def_rw("format", &TextureViewDesc::format, D(TextureViewDesc, format))
        .def_rw("aspect", &TextureViewDesc::aspect, D(TextureViewDesc, aspect))
        .def_rw("subresource_range", &TextureViewDesc::subresource_range, D(TextureViewDesc, subresource_range))
        .def_rw("label", &TextureViewDesc::label, D(TextureViewDesc, label));
    nb::implicitly_convertible<nb::dict, TextureViewDesc>();

    nb::class_<SubresourceLayout>(m, "SubresourceLayout", D(SubresourceLayout))
        .def(nb::init<>())
        .def_rw("size", &SubresourceLayout::size, D(SubresourceLayout, size))
        .def_rw("col_pitch", &SubresourceLayout::col_pitch, D(SubresourceLayout, col_pitch))
        .def_rw("row_pitch", &SubresourceLayout::row_pitch, D(SubresourceLayout, row_pitch))
        .def_rw("slice_pitch", &SubresourceLayout::slice_pitch, D(SubresourceLayout, slice_pitch))
        .def_rw("size_in_bytes", &SubresourceLayout::size_in_bytes, D(SubresourceLayout, size_in_bytes))
        .def_rw("block_width", &SubresourceLayout::block_width, D(SubresourceLayout, block_width))
        .def_rw("block_height", &SubresourceLayout::block_height, D(SubresourceLayout, block_height))
        .def_rw("row_count", &SubresourceLayout::row_count, D(SubresourceLayout, row_count));

    nb::class_<Texture, Resource>(m, "Texture", D(Texture))
        .def_prop_ro("desc", &Texture::desc, D(Texture, desc))
        .def_prop_ro("type", &Texture::type, D(Texture, type))
        .def_prop_ro("format", &Texture::format, D(Texture, format))
        .def_prop_ro("width", &Texture::width, D(Texture, width))
        .def_prop_ro("height", &Texture::height, D(Texture, height))
        .def_prop_ro("depth", &Texture::depth, D(Texture, depth))
        .def_prop_ro("array_length", &Texture::array_length, D(Texture, array_length))
        .def_prop_ro("mip_count", &Texture::mip_count, D(Texture, mip_count))
        .def_prop_ro("layer_count", &Texture::layer_count, D(Texture, layer_count))
        .def_prop_ro("subresource_count", &Texture::subresource_count, D(Texture, subresource_count))
        .def_prop_ro("shared_handle", &Texture::shared_handle, D(Texture, shared_handle))
        .def("get_mip_width", &Texture::get_mip_width, "mip"_a = 0, D(Texture, get_mip_width))
        .def("get_mip_height", &Texture::get_mip_height, "mip"_a = 0, D(Texture, get_mip_height))
        .def("get_mip_depth", &Texture::get_mip_depth, "mip"_a = 0, D(Texture, get_mip_depth))
        .def("get_mip_size", &Texture::get_mip_size, "mip"_a = 0, D(Texture, get_mip_size))
        .def(
            "get_subresource_layout",
            &Texture::get_subresource_layout,
            "mip"_a,
            "row_alignment"_a = DEFAULT_ALIGNMENT,
            D(Texture, get_subresource_layout)
        )
        .def("create_view", &Texture::create_view, "desc"_a, D(Texture, create_view))
        .def(
            "create_view",
            [](Texture* self, nb::dict dict) { return self->create_view(dict_to_TextureViewDesc(dict)); },
            "dict"_a,
            D(Texture, create_view)
        )
        .def(
            "create_view",
            [](Texture* self,
               Format format,
               TextureAspect aspect,
               uint32_t layer,
               uint32_t layer_count,
               uint32_t mip,
               uint32_t mip_count,
               std::string label)
            {
                TextureViewDesc desc;
                desc.format = format == Format::undefined ? self->format() : format;
                desc.aspect = aspect;
                desc.subresource_range = {
                    .layer = layer,
                    .layer_count = layer_count,
                    .mip = mip,
                    .mip_count = mip_count,
                };
                desc.label = label;
                return self->create_view(desc);
            },
            "format"_a = TextureViewDesc().format,
            "aspect"_a = TextureViewDesc().aspect,
            "layer"_a = TextureViewDesc().subresource_range.layer,
            "layer_count"_a = TextureViewDesc().subresource_range.layer_count,
            "mip"_a = TextureViewDesc().subresource_range.mip,
            "mip_count"_a = TextureViewDesc().subresource_range.mip_count,
            "label"_a = TextureViewDesc().label,
            D(Texture, create_view)
        )
        .def("to_bitmap", &Texture::to_bitmap, "layer"_a = 0, "mip"_a = 0, D(Texture, to_bitmap))
        .def("to_numpy", &texture_to_numpy, "layer"_a = 0, "mip"_a = 0, D(texture_to_numpy))
        .def("copy_from_numpy", &texture_from_numpy, "data"_a, "layer"_a = 0, "mip"_a = 0, D(texture_from_numpy));

    nb::class_<TextureView, DeviceResource>(m, "TextureView", D(TextureView))
        .def_prop_ro("texture", &TextureView::texture, D(TextureView, texture))
        .def_prop_ro("desc", &TextureView::desc, D(TextureView, desc))
        .def_prop_ro("format", &TextureView::format, D(TextureView, format))
        .def_prop_ro("aspect", &TextureView::aspect, D(TextureView, aspect))
        .def_prop_ro("subresource_range", &TextureView::subresource_range, D(TextureView, subresource_range))
        .def_prop_ro("label", &TextureView::label, D(TextureView, label))
        .def_prop_ro("descriptor_handle_ro", &TextureView::descriptor_handle_ro, D(TextureView, descriptor_handle_ro))
        .def_prop_ro("descriptor_handle_rw", &TextureView::descriptor_handle_rw, D(TextureView, descriptor_handle_rw))
        .def_prop_ro("native_handle", &TextureView::native_handle, D(TextureView, native_handle))
        .def("__repr__", &TextureView::to_string, D(TextureView, to_string));
}
