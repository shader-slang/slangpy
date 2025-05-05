// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/bitmap.h"
#include "sgl/core/memory_stream.h"
#include "sgl/core/string.h"

#include "sgl/stl/bit.h" // Replace with <bit> when available on all platforms.

SGL_PY_EXPORT(core_bitmap)
{
    using namespace sgl;

    nb::class_<Bitmap, Object> bitmap(m, "Bitmap", D(Bitmap));

    nb::sgl_enum<Bitmap::PixelFormat>(bitmap, "PixelFormat", D(Bitmap, PixelFormat));
    bitmap.attr("ComponentType") = m.attr("DataStruct").attr("Type");
    nb::sgl_enum<Bitmap::FileFormat>(bitmap, "FileFormat", D(Bitmap, FileFormat));

    bitmap //
        .def(
            "__init__",
            [](Bitmap* self,
               Bitmap::PixelFormat pixel_format,
               Bitmap::ComponentType component_type,
               uint32_t width,
               uint32_t height,
               uint32_t channel_count,
               std::vector<std::string> channel_names)
            { new (self) Bitmap(pixel_format, component_type, width, height, channel_count, channel_names); },
            "pixel_format"_a,
            "component_type"_a,
            "width"_a,
            "height"_a,
            "channel_count"_a = 0,
            "channel_names"_a = std::vector<std::string>{},
            D(Bitmap, Bitmap)
        )
        .def(
            "__init__",
            [](Bitmap* self,
               nb::ndarray<nb::device::cpu> data,
               std::optional<Bitmap::PixelFormat> pixel_format_,
               std::optional<std::vector<std::string>> channel_names_)
            {
                if (data.ndim() != 2 && data.ndim() != 3)
                    SGL_THROW("Expect array with dimension 2 or 3.");

                DataStruct::Type component_type;
                if (auto struct_type = dtype_to_data_struct_type(data.dtype()))
                    component_type = *struct_type;
                else
                    SGL_THROW("Unsupported data type.");

                Bitmap::PixelFormat pixel_format = Bitmap::PixelFormat::y;
                uint32_t channel_count = 1;
                if (data.ndim() == 3) {
                    channel_count = narrow_cast<uint32_t>(data.shape(2));
                    switch (channel_count) {
                    case 1:
                        pixel_format = Bitmap::PixelFormat::y;
                        break;
                    case 2:
                        pixel_format = Bitmap::PixelFormat::ya;
                        break;
                    case 3:
                        pixel_format = Bitmap::PixelFormat::rgb;
                        break;
                    case 4:
                        pixel_format = Bitmap::PixelFormat::rgba;
                        break;
                    default:
                        pixel_format = Bitmap::PixelFormat::multi_channel;
                        break;
                    }
                }

                pixel_format = pixel_format_.value_or(pixel_format);

                uint32_t width = narrow_cast<uint32_t>(data.shape(1));
                uint32_t height = narrow_cast<uint32_t>(data.shape(0));
                std::vector channel_names = channel_names_.value_or(std::vector<std::string>{});

                new (self) Bitmap(pixel_format, component_type, width, height, channel_count, channel_names);

                SGL_ASSERT(self->buffer_size() == data.nbytes());

                if (is_ndarray_contiguous(data)) {
                    std::memcpy(self->data(), data.data(), self->buffer_size());
                } else {
                    SGL_THROW("data is not contiguous.");
                }
            },
            "data"_a,
            "pixel_format"_a.none() = nb::none(),
            "channel_names"_a.none() = nb::none()
        )
        .def(
            "__init__",
            [](Bitmap* self, const std::filesystem::path& path) { new (self) Bitmap(path); },
            "path"_a,
            D(Bitmap, Bitmap, 3)
        )
        .def_prop_ro("pixel_format", &Bitmap::pixel_format, D(Bitmap, pixel_format))
        .def_prop_ro("component_type", &Bitmap::component_type, D(Bitmap, component_type))
        .def_prop_ro("pixel_struct", &Bitmap::pixel_struct, D(Bitmap, pixel_struct))
        .def_prop_ro("width", &Bitmap::width, D(Bitmap, width))
        .def_prop_ro("height", &Bitmap::height, D(Bitmap, height))
        .def_prop_ro("pixel_count", &Bitmap::pixel_count, D(Bitmap, pixel_count))
        .def_prop_ro("channel_count", &Bitmap::channel_count, D(Bitmap, channel_count))
        .def_prop_ro("channel_names", &Bitmap::channel_names, D(Bitmap, channel_names))
        .def_prop_rw("srgb_gamma", &Bitmap::srgb_gamma, &Bitmap::set_srgb_gamma, D(Bitmap, srgb_gamma))
        .def("has_alpha", &Bitmap::has_alpha, D(Bitmap, has_alpha))
        .def_prop_ro("bytes_per_pixel", &Bitmap::bytes_per_pixel, D(Bitmap, bytes_per_pixel))
        .def_prop_ro("buffer_size", &Bitmap::buffer_size, D(Bitmap, buffer_size))
        .def("empty", &Bitmap::empty, D(Bitmap, empty))
        .def("clear", &Bitmap::clear, D(Bitmap, clear))
        .def("vflip", &Bitmap::vflip, D(Bitmap, vflip))
        .def("split", &Bitmap::split, D(Bitmap, split))
        .def(
            "convert",
            [](Bitmap& self,
               std::optional<Bitmap::PixelFormat> pixel_format,
               std::optional<Bitmap::ComponentType> component_type,
               std::optional<bool> srgb_gamma) -> ref<Bitmap>
            {
                return self.convert(
                    pixel_format.value_or(self.pixel_format()),
                    component_type.value_or(self.component_type()),
                    srgb_gamma.value_or(self.srgb_gamma())
                );
            },
            "pixel_format"_a.none() = nb::none(),
            "component_type"_a.none() = nb::none(),
            "srgb_gamma"_a.none() = nb::none(),
            D(Bitmap, convert)
        )
        .def(
            "write",
            nb::overload_cast<const std::filesystem::path&, Bitmap::FileFormat, int>(&Bitmap::write, nb::const_),
            "path"_a,
            "format"_a = Bitmap::FileFormat::auto_,
            "quality"_a = -1,
            D(Bitmap, write)
        )
        .def(
            "write_async",
            &Bitmap::write_async,
            "path"_a,
            "format"_a = Bitmap::FileFormat::auto_,
            "quality"_a = -1,
            D(Bitmap, write_async)
        )
        .def_static(
            "read_multiple",
            &Bitmap::read_multiple,
            "paths"_a,
            "format"_a = Bitmap::FileFormat::auto_,
            D(Bitmap, read_multiple)
        )
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def_prop_ro(
            "__array_interface__",
            [](Bitmap& self) -> nb::object
            {
                if (self.empty())
                    return nb::none();

                nb::dict result;
                if (self.channel_count() == 1)
                    result["shape"] = nb::make_tuple(self.height(), self.width());
                else
                    result["shape"] = nb::make_tuple(self.height(), self.width(), self.channel_count());

                std::string format(3, '\0');
                format[0] = stdx::endian::native == stdx::endian::little ? '<' : '>';
                format[1] = DataStruct::is_float(self.component_type())
                    ? 'f'
                    : (DataStruct::is_unsigned(self.component_type()) ? 'u' : 'i');
                format[2] = '0' + static_cast<char>(DataStruct::type_size(self.component_type()));
                result["typestr"] = format;

                result["data"] = nb::make_tuple(reinterpret_cast<uintptr_t>(self.data()), false);
                result["version"] = 3;

                return nb::object(result);
            }
        )
        .def(
            "_repr_html_",
            [](const Bitmap& self) -> nb::object
            {
                if (self.pixel_format() == Bitmap::PixelFormat::multi_channel)
                    return nb::none();

                // Check if bitmap needs conversion to be saved as PNG for display in Jupyter.
                bool needs_conversion = !self.srgb_gamma();
                needs_conversion
                    |= (self.pixel_format() != Bitmap::PixelFormat::y && self.pixel_format() != Bitmap::PixelFormat::rgb
                        && self.pixel_format() != Bitmap::PixelFormat::rgba);
                needs_conversion
                    |= (self.component_type() != Bitmap::ComponentType::uint8
                        && self.component_type() != Bitmap::ComponentType::uint16);

                // Write bitmap to PNG in memory.
                ref<MemoryStream> stream = ref(new MemoryStream());
                if (needs_conversion) {
                    self.convert(
                            self.has_alpha() ? Bitmap::PixelFormat::rgba : Bitmap::PixelFormat::rgb,
                            Bitmap::ComponentType::uint16,
                            true
                    )
                        ->write(stream, Bitmap::FileFormat::png);
                } else {
                    self.write(stream, Bitmap::FileFormat::png);
                }

                std::string html = "<img src=\"data:image/png;base64, ";
                html += string::encode_base64(stream->data(), stream->size());
                html += "\" width=\"400vm\" />";

                return nb::str(html.c_str());
            }
        );
}
