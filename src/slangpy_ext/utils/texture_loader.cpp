// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/utils/texture_loader.h"

#include "sgl/device/device.h"

#include "sgl/core/bitmap.h"

namespace sgl {
using TextureLoaderOptions = TextureLoader::Options;
SGL_DICT_TO_DESC_BEGIN(TextureLoaderOptions)
SGL_DICT_TO_DESC_FIELD(load_as_normalized, bool)
SGL_DICT_TO_DESC_FIELD(load_as_srgb, bool)
SGL_DICT_TO_DESC_FIELD(extend_alpha, bool)
SGL_DICT_TO_DESC_FIELD(allocate_mips, bool)
SGL_DICT_TO_DESC_FIELD(generate_mips, bool)
SGL_DICT_TO_DESC_FIELD(usage, TextureUsage)
SGL_DICT_TO_DESC_FIELD(ya_handling, YAHandling)
SGL_DICT_TO_DESC_END()
} // namespace sgl

SGL_PY_EXPORT(utils_texture_loader)
{
    using namespace sgl;

    nb::enum_<YAHandling>(m, "YAHandling")
        .value("expand_to_rgba", YAHandling::expand_to_rgba)
        .value("preserve_as_rg", YAHandling::preserve_as_rg);

    nb::class_<TextureLoader, Object> texture_loader(m, "TextureLoader", D(TextureLoader));

    nb::class_<TextureLoader::Options>(texture_loader, "Options", D(TextureLoader, Options))
        .def(nb::init<>(), D(TextureLoader, Options))
        .def(
            "__init__",
            [](TextureLoader::Options* self, nb::dict dict)
            {
                new (self) TextureLoader::Options(dict_to_TextureLoaderOptions(dict));
            }
        )
        .def_rw(
            "load_as_normalized",
            &TextureLoader::Options::load_as_normalized,
            D(TextureLoader, Options, load_as_normalized)
        )
        .def_rw("load_as_srgb", &TextureLoader::Options::load_as_srgb, D(TextureLoader, Options, load_as_srgb))
        .def_rw("extend_alpha", &TextureLoader::Options::extend_alpha, D(TextureLoader, Options, extend_alpha))
        .def_rw("allocate_mips", &TextureLoader::Options::allocate_mips, D(TextureLoader, Options, allocate_mips))
        .def_rw("generate_mips", &TextureLoader::Options::generate_mips, D(TextureLoader, Options, generate_mips))
        .def_rw("usage", &TextureLoader::Options::usage)
        .def_rw("ya_handling", &TextureLoader::Options::ya_handling);

    nb::implicitly_convertible<nb::dict, TextureLoader::Options>();

    texture_loader //
        .def(nb::init<ref<Device>>(), "device"_a, D(TextureLoader, TextureLoader))
        .def(
            "load_texture",
            nb::overload_cast<const Bitmap*, std::optional<TextureLoader::Options>>(&TextureLoader::load_texture),
            "bitmap"_a,
            "options"_a.none() = nb::none(),
            D(TextureLoader, load_texture)
        )
        .def(
            "load_texture",
            nb::overload_cast<const std::filesystem::path&, std::optional<TextureLoader::Options>>(
                &TextureLoader::load_texture
            ),
            "path"_a,
            "options"_a.none() = nb::none(),
            D(TextureLoader, load_texture, 2)
        )
        .def(
            "load_textures",
            nb::overload_cast<std::span<const Bitmap*>, std::optional<TextureLoader::Options>>(
                &TextureLoader::load_textures
            ),
            "bitmaps"_a,
            "options"_a.none() = nb::none(),
            D(TextureLoader, load_textures)
        )
        .def(
            "load_textures",
            nb::overload_cast<std::span<std::filesystem::path>, std::optional<TextureLoader::Options>>(
                &TextureLoader::load_textures
            ),
            "paths"_a,
            "options"_a.none() = nb::none(),
            D(TextureLoader, load_textures, 2)
        )
        .def(
            "load_texture_array",
            nb::overload_cast<std::span<const Bitmap*>, std::optional<TextureLoader::Options>>(
                &TextureLoader::load_texture_array
            ),
            "bitmaps"_a,
            "options"_a.none() = nb::none(),
            D(TextureLoader, load_texture_array)
        )
        .def(
            "load_texture_array",
            nb::overload_cast<std::span<std::filesystem::path>, std::optional<TextureLoader::Options>>(
                &TextureLoader::load_texture_array
            ),
            "paths"_a,
            "options"_a.none() = nb::none(),
            D(TextureLoader, load_texture_array, 2)
        );
}
