// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/thread.h"

SGL_PY_EXPORT(core_thread)
{
    using namespace sgl::thread;

    nb::module_ thread = nb::module_::import_("slangpy.thread");

    thread.def(
        "wait_for_tasks",
        []()
        {
            nb::gil_scoped_release guard;
            wait_for_tasks();
        },
        D(thread, wait_for_tasks)
    );
}
