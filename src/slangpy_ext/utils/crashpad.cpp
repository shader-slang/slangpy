// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/utils/crashpad.h"

SGL_PY_EXPORT(utils_crashpad)
{
    using namespace sgl;

    nb::module_ crashpad = nb::module_::import_("slangpy.crashpad");

    crashpad.def("is_supported", &crashpad::is_supported, D(crashpad, is_supported));
    crashpad.def(
        "start_handler",
        &crashpad::start_handler,
        "handler"_a = "",
        "database"_a = "",
        "annotations"_a = std::map<std::string, std::string>{},
        D(crashpad, start_handler)
    );
}
