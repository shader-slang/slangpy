// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/utils/tev.h"
#include "sgl/core/bitmap.h"
#include "sgl/device/resource.h"

SGL_PY_EXPORT(utils_tev)
{
    using namespace sgl;

    nb::module_ tev = nb::module_::import_("slangpy.tev");

    tev.def(
        "show",
        nb::overload_cast<const Bitmap*, std::string, std::string, uint16_t, uint32_t>(&tev::show),
        "bitmap"_a,
        "name"_a = "",
        "host"_a = "127.0.0.1",
        "port"_a = 14158,
        "max_retries"_a = 3,
        D(tev, show)
    );
    tev.def(
        "show",
        nb::overload_cast<const Texture*, std::string, std::string, uint16_t, uint32_t>(&tev::show),
        "texture"_a,
        "name"_a = "",
        "host"_a = "127.0.0.1",
        "port"_a = 14158,
        "max_retries"_a = 3,
        D(tev, show, 2)
    );
    tev.def(
        "show_async",
        nb::overload_cast<const Bitmap*, std::string, std::string, uint16_t, uint32_t>(&tev::show_async),
        "bitmap"_a,
        "name"_a = "",
        "host"_a = "127.0.0.1",
        "port"_a = 14158,
        "max_retries"_a = 3,
        D(tev, show_async)
    );
    tev.def(
        "show_async",
        nb::overload_cast<const Texture*, std::string, std::string, uint16_t, uint32_t>(&tev::show_async),
        "texture"_a,
        "name"_a = "",
        "host"_a = "127.0.0.1",
        "port"_a = 14158,
        "max_retries"_a = 3,
        D(tev, show_async, 2)
    );
}
