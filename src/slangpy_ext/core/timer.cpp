// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/timer.h"

SGL_PY_EXPORT(core_timer)
{
    using namespace sgl;

    nb::class_<Timer>(m, "Timer", D(Timer))
        .def(nb::init<>())
        .def("reset", &Timer::reset, D(Timer, reset))
        .def("elapsed_s", &Timer::elapsed_s, D(Timer, elapsed_s))
        .def("elapsed_ms", &Timer::elapsed_ms, D(Timer, elapsed_ms))
        .def("elapsed_us", &Timer::elapsed_us, D(Timer, elapsed_us))
        .def("elapsed_ns", &Timer::elapsed_ns, D(Timer, elapsed_ns))
        .def_static("delta_s", &Timer::delta_s, "start"_a, "end"_a, D(Timer, delta_s))
        .def_static("delta_ms", &Timer::delta_ms, "start"_a, "end"_a, D(Timer, delta_ms))
        .def_static("delta_us", &Timer::delta_us, "start"_a, "end"_a, D(Timer, delta_us))
        .def_static("delta_ns", &Timer::delta_ns, "start"_a, "end"_a, D(Timer, delta_ns))
        .def_static("now", &Timer::now, D(Timer, now));
}
