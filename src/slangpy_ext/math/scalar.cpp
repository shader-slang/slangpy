// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/math/scalar_types.h"
#include "sgl/math/scalar_math.h"
#include "sgl/core/traits.h"

#include <array>
#include <type_traits>

namespace sgl::math {

inline void bind_enums(nb::module_& m)
{
    nb::enum_<Handedness>(m, "Handedness")
        .value("right_handed", Handedness::right_handed)
        .value("left_handed", Handedness::left_handed);
}

inline void bind_float16(nb::module_& m)
{
    nb::class_<float16_t> float16(m, "float16_t");
    float16.def(nb::init<float>(), "value"_a);
    float16.def(nb::init_implicit<float>(), "value"_a);
    float16.def("__float__", &float16_t::operator float);
}

inline void bind_scalar(nb::module_& m)
{
    // Floating point checks

    m.def("isfinite", isfinite<float>, "x"_a);
    m.def("isfinite", isfinite<double>, "x"_a);
    m.def("isfinite", isfinite<float16_t>, "x"_a);

    m.def("isinf", isinf<float>, "x"_a);
    m.def("isinf", isinf<double>, "x"_a);
    m.def("isinf", isinf<float16_t>, "x"_a);

    m.def("isnan", isnan<float>, "x"_a);
    m.def("isnan", isnan<double>, "x"_a);
    m.def("isnan", isnan<float16_t>, "x"_a);

    // Rounding

    m.def("floor", floor<float>, "x"_a);
    m.def("floor", floor<double>, "x"_a);
    // m.def("floor", floor<float16_t>, "x"_a);

    m.def("ceil", ceil<float>, "x"_a);
    m.def("ceil", ceil<double>, "x"_a);
    // m.def("ceil", ceil<float16_t>, "x"_a);

    m.def("trunc", trunc<float>, "x"_a);
    m.def("trunc", trunc<double>, "x"_a);
    // m.def("trunc", trunc<float16_t>, "x"_a);

    m.def("round", round<float>, "x"_a);
    m.def("round", round<double>, "x"_a);
    // m.def("round", round<float16_t>, "x"_a);

    // Exponential

    m.def("pow", pow<float>, "x"_a, "y"_a);
    m.def("pow", pow<double>, "x"_a, "y"_a);
    // m.def("pow", pow<float16_t>, "x"_a, "y"_a);

    m.def("sqrt", sqrt<float>, "x"_a);
    m.def("sqrt", sqrt<double>, "x"_a);
    // m.def("sqrt", sqrt<float16_t>, "x"_a);

    m.def("rsqrt", rsqrt<float>, "x"_a);
    m.def("rsqrt", rsqrt<double>, "x"_a);
    // m.def("rsqrt", rsqrt<float16_t>, "x"_a);

    m.def("exp", exp<float>, "x"_a);
    m.def("exp", exp<double>, "x"_a);
    m.def("exp", exp<float16_t>, "x"_a);

    m.def("exp2", exp2<float>, "x"_a);
    m.def("exp2", exp2<double>, "x"_a);
    m.def("exp2", exp2<float16_t>, "x"_a);

    m.def("log", log<float>, "x"_a);
    m.def("log", log<double>, "x"_a);
    m.def("log", log<float16_t>, "x"_a);

    m.def("log2", log2<float>, "x"_a);
    m.def("log2", log2<double>, "x"_a);
    // m.def("log2", log2<float16_t>, "x"_a);

    m.def("log10", log10<float>, "x"_a);
    m.def("log10", log10<double>, "x"_a);
    // m.def("log10", log10<float16_t>, "x"_a);

    // Trigonometry

    m.def("radians", radians<float>, "x"_a);
    m.def("radians", radians<double>, "x"_a);

    m.def("degrees", degrees<float>, "x"_a);
    m.def("degrees", degrees<double>, "x"_a);

    m.def("sin", sin<float>, "x"_a);
    m.def("sin", sin<double>, "x"_a);

    m.def("cos", cos<float>, "x"_a);
    m.def("cos", cos<double>, "x"_a);

    m.def("tan", tan<float>, "x"_a);
    m.def("tan", tan<double>, "x"_a);

    m.def("asin", asin<float>, "x"_a);
    m.def("asin", asin<double>, "x"_a);

    m.def("acos", acos<float>, "x"_a);
    m.def("acos", acos<double>, "x"_a);

    m.def("atan", atan<float>, "x"_a);
    m.def("atan", atan<double>, "x"_a);

    m.def("atan2", atan2<float>, "y"_a, "x"_a);
    m.def("atan2", atan2<double>, "y"_a, "x"_a);

    m.def("sinh", sinh<float>, "x"_a);
    m.def("sinh", sinh<double>, "x"_a);

    m.def("cosh", cosh<float>, "x"_a);
    m.def("cosh", cosh<double>, "x"_a);

    m.def("tanh", tanh<float>, "x"_a);
    m.def("tanh", tanh<double>, "x"_a);

    // Misc

    m.def("fmod", fmod<float>, "x"_a, "y"_a);
    m.def("fmod", fmod<double>, "x"_a, "y"_a);

    m.def("frac", frac<float>, "x"_a);
    m.def("frac", frac<double>, "x"_a);

    m.def("lerp", lerp<float>, "x"_a, "y"_a, "s"_a);
    m.def("lerp", lerp<double>, "x"_a, "y"_a, "s"_a);

    m.def("rcp", rcp<float>, "x"_a);
    m.def("rcp", rcp<double>, "x"_a);

    m.def("saturate", saturate<float>, "x"_a);
    m.def("saturate", saturate<double>, "x"_a);

    m.def("step", step<float>, "x"_a, "y"_a);
    m.def("step", step<double>, "x"_a, "y"_a);

    m.def("smoothstep", smoothstep<float>, "min"_a, "max"_a, "x"_a);
    m.def("smoothstep", smoothstep<double>, "min"_a, "max"_a, "x"_a);

    // Conversion

    m.def("f16tof32", nb::overload_cast<uint>(f16tof32), "x"_a);
    m.def("f32tof16", nb::overload_cast<float>(f32tof16), "x"_a);

    m.def("asfloat", nb::overload_cast<uint32_t>(asfloat), "x"_a);
    m.def("asfloat", nb::overload_cast<int32_t>(asfloat), "x"_a);
    m.def("asfloat16", asfloat16, "x"_a);

    m.def("asuint", nb::overload_cast<float>(asuint), "x"_a);
    m.def("asint", nb::overload_cast<float>(asint), "x"_a);
    m.def("asuint16", asuint16, "x"_a);
}

} // namespace sgl::math

SGL_PY_EXPORT(math_scalar)
{
    nb::module_ math = nb::module_::import_("slangpy.math");

    sgl::math::bind_enums(math);

    sgl::math::bind_float16(math);
    sgl::math::bind_scalar(math);

    m.attr("float16_t") = math.attr("float16_t");
}
