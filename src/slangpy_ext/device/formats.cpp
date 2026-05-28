// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/formats.h"

SGL_PY_EXPORT(device_formats)
{
    using namespace sgl;

    nb::sgl_enum<Format>(m, "Format");
    nb::sgl_enum<FormatType>(m, "FormatType");
    nb::sgl_enum_flags<FormatChannels>(m, "FormatChannels");
    nb::sgl_enum_flags<FormatSupport>(m, "FormatSupport");

    nb::class_<FormatInfo>(m, "FormatInfo", D(FormatInfo))
        .def_ro("format", &FormatInfo::format, D(FormatInfo, format))
        .def_ro("name", &FormatInfo::name, D(FormatInfo, name))
        .def_ro("bytes_per_block", &FormatInfo::bytes_per_block, D(FormatInfo, bytes_per_block))
        .def_ro("channel_count", &FormatInfo::channel_count, D(FormatInfo, channel_count))
        .def_ro("type", &FormatInfo::type, D(FormatInfo, type))
        .def_ro("is_depth", &FormatInfo::is_depth, D(FormatInfo, is_depth))
        .def_ro("is_stencil", &FormatInfo::is_stencil, D(FormatInfo, is_stencil))
        .def_ro("is_compressed", &FormatInfo::is_compressed, D(FormatInfo, is_compressed))
        .def_ro("block_width", &FormatInfo::block_width, D(FormatInfo, block_width))
        .def_ro("block_height", &FormatInfo::block_height, D(FormatInfo, block_height))
        .def_ro("channel_bit_count", &FormatInfo::channel_bit_count, D(FormatInfo, channel_bit_count))
        .def_ro("dxgi_format", &FormatInfo::dxgi_format, D(FormatInfo, dxgi_format))
        .def_ro("vk_format", &FormatInfo::vk_format, D(FormatInfo, vk_format))
        .def("is_depth_stencil", &FormatInfo::is_depth_stencil, D(FormatInfo, is_depth_stencil))
        .def("is_float_format", &FormatInfo::is_float_format, D(FormatInfo, is_float_format))
        .def("is_integer_format", &FormatInfo::is_integer_format, D(FormatInfo, is_integer_format))
        .def("is_normalized_format", &FormatInfo::is_normalized_format, D(FormatInfo, is_normalized_format))
        .def("is_srgb_format", &FormatInfo::is_srgb_format, D(FormatInfo, is_srgb_format))
        .def("get_channels", &FormatInfo::get_channels, D(FormatInfo, get_channels))
        .def("get_channel_bits", &FormatInfo::get_channel_bits, D(FormatInfo, get_channel_bits))
        .def("has_equal_channel_bits", &FormatInfo::has_equal_channel_bits, D(FormatInfo, has_equal_channel_bits))
        .def("__repr__", &FormatInfo::to_string);

    m.def("get_format_info", &get_format_info, D(get_format_info));
}
