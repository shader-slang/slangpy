// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/surface.h"

namespace sgl {

SGL_DICT_TO_DESC_BEGIN(SurfaceConfig)
SGL_DICT_TO_DESC_FIELD(format, Format)
SGL_DICT_TO_DESC_FIELD(usage, TextureUsage)
SGL_DICT_TO_DESC_FIELD(width, uint32_t)
SGL_DICT_TO_DESC_FIELD(height, uint32_t)
SGL_DICT_TO_DESC_FIELD(desired_image_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(vsync, bool)
SGL_DICT_TO_DESC_END()

} // namespace sgl

SGL_PY_EXPORT(device_surface)
{
    using namespace sgl;

    nb::class_<SurfaceInfo>(m, "SurfaceInfo", D(SurfaceInfo))
        .def_ro("preferred_format", &SurfaceInfo::preferred_format, D(SurfaceInfo, preferred_format))
        .def_ro("supported_usage", &SurfaceInfo::supported_usage, D(SurfaceInfo, supported_usage))
        .def_ro("formats", &SurfaceInfo::formats, D(SurfaceInfo, formats));


    nb::class_<SurfaceConfig>(m, "SurfaceConfig", D(SurfaceConfig))
        .def(nb::init<>())
        .def(
            "__init__",
            [](SurfaceConfig* self, nb::dict dict) { new (self) SurfaceConfig(dict_to_SurfaceConfig(dict)); }
        )
        .def_rw("format", &SurfaceConfig::format, D(SurfaceConfig, format))
        .def_rw("usage", &SurfaceConfig::usage, D(SurfaceConfig, usage))
        .def_rw("width", &SurfaceConfig::width, D(SurfaceConfig, width))
        .def_rw("height", &SurfaceConfig::height, D(SurfaceConfig, height))
        .def_rw("desired_image_count", &SurfaceConfig::desired_image_count, D(SurfaceConfig, desired_image_count))
        .def_rw("vsync", &SurfaceConfig::vsync, D(SurfaceConfig, vsync));
    nb::implicitly_convertible<nb::dict, SurfaceConfig>();

    nb::class_<Surface, Object>(m, "Surface", D(Surface))
        .def_prop_ro("info", &Surface::info, D(Surface, info))
        .def_prop_ro("config", &Surface::config, D(Surface, config))
        .def(
            "configure",
            [](Surface* self,
               uint32_t width,
               uint32_t height,
               Format format,
               TextureUsage usage,
               uint32_t desired_image_count,
               bool vsync)
            {
                self->configure({
                    .format = format,
                    .usage = usage,
                    .width = width,
                    .height = height,
                    .desired_image_count = desired_image_count,
                    .vsync = vsync,
                });
            },
            "width"_a,
            "height"_a,
            "format"_a = SurfaceConfig().format,
            "usage"_a = SurfaceConfig().usage,
            "desired_image_count"_a = SurfaceConfig().desired_image_count,
            "vsync"_a = SurfaceConfig().vsync,
            D(Surface, configure)
        )
        .def("configure", &Surface::configure, "config"_a, D(Surface, configure))
        .def("unconfigure", &Surface::unconfigure, D(Surface, unconfigure))
        .def("acquire_next_image", &Surface::acquire_next_image, D(Surface, acquire_next_image))
        .def("present", &Surface::present, D(Surface, present));
}
