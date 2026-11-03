// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/bc_codec.h"

namespace sgl {
using BCChannelWeights = std::array<uint32_t, 4>;
SGL_DICT_TO_DESC_BEGIN(BCEncodeOptions)
SGL_DICT_TO_DESC_FIELD(quality, BCEncodeQuality)
SGL_DICT_TO_DESC_FIELD(generate_mipmaps, bool)
SGL_DICT_TO_DESC_FIELD(mip_filter, BCMipFilter)
SGL_DICT_TO_DESC_FIELD(channel_weights, BCChannelWeights)
SGL_DICT_TO_DESC_FIELD(has_alpha, bool)
SGL_DICT_TO_DESC_END()
} // namespace sgl

SGL_PY_EXPORT(core_bc_codec)
{
    using namespace sgl;

    nb::sgl_enum<BCEncoderBackend>(m, "BCEncoderBackend", D(BCEncoderBackend));
    nb::sgl_enum<BCEncodeQuality>(m, "BCEncodeQuality", D(BCEncodeQuality));

    nb::class_<BCEncodeOptions>(m, "BCEncodeOptions", D(BCEncodeOptions))
        .def(nb::init<>())
        .def(
            "__init__",
            [](BCEncodeOptions* self, nb::dict dict)
            {
                new (self) BCEncodeOptions(dict_to_BCEncodeOptions(dict));
            }
        )
        .def_rw("quality", &BCEncodeOptions::quality, D(BCEncodeOptions, quality))
        .def_rw("generate_mipmaps", &BCEncodeOptions::generate_mipmaps, D(BCEncodeOptions, generate_mipmaps))
        .def_rw("mip_filter", &BCEncodeOptions::mip_filter, D(BCEncodeOptions, mip_filter))
        .def_rw("channel_weights", &BCEncodeOptions::channel_weights, D(BCEncodeOptions, channel_weights))
        .def_rw("has_alpha", &BCEncodeOptions::has_alpha, D(BCEncodeOptions, has_alpha));
    nb::implicitly_convertible<nb::dict, BCEncodeOptions>();

    nb::class_<BCCompressedMip>(m, "BCCompressedMip", D(BCCompressedMip))
        .def_prop_ro("width", &BCCompressedMip::width, D(BCCompressedMip, width))
        .def_prop_ro("height", &BCCompressedMip::height, D(BCCompressedMip, height))
        .def_prop_ro(
            "data",
            [](const BCCompressedMip& self)
            {
                size_t shape[] = {self.data().size()};
                return nb::ndarray<nb::numpy, const uint8_t>(
                    self.data().data(),
                    1,
                    shape,
                    nb::cast(&self, nb::rv_policy::reference)
                );
            },
            D(BCCompressedMip, data)
        );

    nb::class_<BCCompressedImage>(m, "BCCompressedImage", D(BCCompressedImage))
        .def(
            nb::init<Format, std::vector<BCCompressedMip>>(),
            "format"_a,
            "mip_levels"_a,
            D(BCCompressedImage, BCCompressedImage)
        )
        .def(
            nb::init<const std::filesystem::path&>(),
            "path"_a,
            nb::call_guard<nb::gil_scoped_release>(),
            D(BCCompressedImage, BCCompressedImage, 2)
        )
        .def_prop_ro("format", &BCCompressedImage::format, D(BCCompressedImage, format))
        .def_prop_ro("width", &BCCompressedImage::width, D(BCCompressedImage, width))
        .def_prop_ro("height", &BCCompressedImage::height, D(BCCompressedImage, height))
        .def_prop_ro("mip_count", &BCCompressedImage::mip_count, D(BCCompressedImage, mip_count))
        .def_prop_ro(
            "mip_levels",
            [](const BCCompressedImage& self)
            {
                nb::list result;
                nb::object owner = nb::cast(&self, nb::rv_policy::reference);
                for (const auto& mip : self.mip_levels())
                    result.append(nb::cast(&mip, nb::rv_policy::reference_internal, owner));
                return result;
            },
            nb::sig("def mip_levels(self) -> list[BCCompressedMip]"),
            D(BCCompressedImage, mip_levels)
        )
        .def(
            "write_dds",
            nb::overload_cast<const std::filesystem::path&>(&BCCompressedImage::write_dds, nb::const_),
            "path"_a,
            nb::call_guard<nb::gil_scoped_release>(),
            D(BCCompressedImage, write_dds)
        )
        .def(
            "decode",
            &BCCompressedImage::decode,
            "mip_level"_a = 0,
            nb::call_guard<nb::gil_scoped_release>(),
            D(BCCompressedImage, decode)
        );

    nb::class_<BCEncoder>(m, "BCEncoder", D(BCEncoder))
        .def(nb::init<BCEncoderBackend>(), "backend"_a = BCEncoderBackend::automatic, D(BCEncoder, BCEncoder))
        .def_prop_ro("backend", &BCEncoder::backend, D(BCEncoder, backend))
        .def_static(
            "is_backend_available",
            &BCEncoder::is_backend_available,
            "backend"_a,
            D(BCEncoder, is_backend_available)
        )
        .def("can_encode", &BCEncoder::can_encode, "format"_a, D(BCEncoder, can_encode))
        .def(
            "encode",
            [](BCEncoder& self, const Bitmap* bitmap, Format format, std::optional<BCEncodeOptions> options)
            {
                return self.encode(bitmap, format, options.value_or(BCEncodeOptions{}));
            },
            "bitmap"_a,
            "format"_a,
            "options"_a.none() = nb::none(),
            nb::call_guard<nb::gil_scoped_release>(),
            D(BCEncoder, encode, 2)
        );
}
