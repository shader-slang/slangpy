// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/rfilter.h"

SGL_PY_EXPORT(core_rfilter)
{
    using namespace sgl;

    nb::sgl_enum<FilterBoundaryCondition>(m, "FilterBoundaryCondition", D(FilterBoundaryCondition));

    nb::class_<BoxFilter>(m, "BoxFilter", D(BoxFilter))
        .def(nb::init<>())
        .def("eval", &BoxFilter::eval, "x"_a)
        .def_prop_ro("radius", &BoxFilter::radius);

    nb::class_<TentFilter>(m, "TentFilter", D(TentFilter))
        .def(nb::init<float>(), "radius"_a = 1.f)
        .def("eval", &TentFilter::eval, "x"_a)
        .def_prop_ro("radius", &TentFilter::radius);

    nb::class_<GaussianFilter>(m, "GaussianFilter", D(GaussianFilter))
        .def(nb::init<float>(), "stddev"_a = 0.5f)
        .def("eval", &GaussianFilter::eval, "x"_a)
        .def_prop_ro("radius", &GaussianFilter::radius);

    nb::class_<MitchellFilter>(m, "MitchellFilter", D(MitchellFilter))
        .def(nb::init<float, float>(), "b"_a = 1.f / 3.f, "c"_a = 1.f / 3.f)
        .def("eval", &MitchellFilter::eval, "x"_a)
        .def_prop_ro("radius", &MitchellFilter::radius);

    nb::class_<LanczosFilter>(m, "LanczosFilter", D(LanczosFilter))
        .def(nb::init<int>(), "lobes"_a = 3)
        .def("eval", &LanczosFilter::eval, "x"_a)
        .def_prop_ro("radius", &LanczosFilter::radius);
}
