// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/math/vector.h"
#include "sgl/core/traits.h"

#include "math/primitivetype.h"

#include <array>
#include <type_traits>

namespace sgl::math {

template<typename T, bool BindIntrinsics = true>
void bind_vector_type(nb::module_& m, const char* name)
{
    auto constexpr dimension = T::dimension;
    using value_type = typename T::value_type;

    static_assert(dimension >= 1 && dimension <= 4, "Invalid dimension");

    nb::class_<T> vec(m, name);

    // Constructors

    auto init_empty = [](T* self) { new (self) T(value_type(0)); };
    vec.def("__init__", init_empty);

    vec.def(nb::init<value_type>(), "scalar"_a);

    if constexpr (dimension == 2) {
        vec.def(nb::init<value_type, value_type>(), "x"_a, "y"_a);
    } else if constexpr (dimension == 3) {
        vec.def(nb::init<value_type, value_type, value_type>(), "x"_a, "y"_a, "z"_a);
        vec.def(nb::init<vector<value_type, 2>, value_type>(), "xy"_a, "z"_a);
        vec.def(nb::init<value_type, vector<value_type, 2>>(), "x"_a, "yz"_a);
    } else if constexpr (dimension == 4) {
        vec.def(nb::init<value_type, value_type, value_type, value_type>(), "x"_a, "y"_a, "z"_a, "w"_a);
        vec.def(nb::init<vector<value_type, 2>, vector<value_type, 2>>(), "xy"_a, "zw"_a);
        vec.def(nb::init<vector<value_type, 3>, value_type>(), "xyz"_a, "w"_a);
        vec.def(nb::init<value_type, vector<value_type, 3>>(), "x"_a, "yzw"_a);
    }

    vec.def(nb::init_implicit<std::array<value_type, dimension>>(), "a"_a);

    // Field access

    vec.def_rw("x", &T::x);
    if constexpr (dimension >= 2)
        vec.def_rw("y", &T::y);
    if constexpr (dimension >= 3)
        vec.def_rw("z", &T::z);
    if constexpr (dimension >= 4)
        vec.def_rw("w", &T::w);

    // Including all swizzles is overkill, but these 2 are very useful for conversions.
    if constexpr (dimension == 3) {
        vec.def_prop_ro("xy", &T::xy);
    }
    if constexpr (dimension == 4) {
        vec.def_prop_ro("xyz", &T::xyz);
    }

    vec.def("__len__", [](const T&) { return dimension; });
    vec.def(
        "__getitem__",
        [](const T& self, Py_ssize_t i)
        {
            i = detail::sanitize_getitem_index(i, dimension);
            return self[int(i)];
        }
    );
    vec.def(
        "__setitem__",
        [](T& self, int i, value_type v)
        {
            if (i >= dimension)
                throw nb::index_error();
            self[i] = v;
        }
    );

    vec.def_prop_ro(
        "shape",
        [dimension](const T& self)
        {
            SGL_UNUSED(self);
            return nb::make_tuple(dimension);
        }
    );

    vec.def_prop_ro(
        "element_type",
        [](const T& self)
        {
            SGL_UNUSED(self);
            return nb::handle(PrimitiveType<value_type>::python_type());
        }
    );

    // Conversion

    auto to_string_ = [](const T& self) { return to_string(self); };
    vec.def("__repr__", to_string_);
    vec.def("__str__", to_string_);

    // Operators

    if constexpr (arithmetic<value_type> && !boolean<value_type>) {
        vec.def(+nb::self);
        vec.def(-nb::self);

        vec.def(nb::self + nb::self);
        vec.def(nb::self + value_type());
        vec.def(value_type() + nb::self);

        vec.def(nb::self - nb::self);
        vec.def(nb::self - value_type());
        vec.def(value_type() - nb::self);

        vec.def(nb::self * nb::self);
        vec.def(nb::self * value_type());
        vec.def(value_type() * nb::self);

        vec.def(nb::self / nb::self);
        vec.def(nb::self / value_type());
        vec.def(value_type() / nb::self);

        vec.def(nb::self += nb::self);
        vec.def(nb::self += value_type());

        vec.def(nb::self -= nb::self);
        vec.def(nb::self -= value_type());

        vec.def(nb::self *= nb::self);
        vec.def(nb::self *= value_type());

        vec.def(nb::self /= nb::self);
        vec.def(nb::self /= value_type());
    }

    if constexpr (integral<value_type> && !boolean<value_type>) {
        vec.def(nb::self % nb::self);
        vec.def(nb::self % value_type());
        vec.def(value_type() % nb::self);

        vec.def(nb::self << nb::self);
        vec.def(nb::self << value_type());
        vec.def(value_type() << nb::self);

        vec.def(nb::self >> nb::self);
        vec.def(nb::self >> value_type());
        vec.def(value_type() >> nb::self);

        vec.def(nb::self | nb::self);
        vec.def(nb::self | value_type());
        vec.def(value_type() | nb::self);

        vec.def(nb::self & nb::self);
        vec.def(nb::self & value_type());
        vec.def(value_type() & nb::self);

        vec.def(nb::self ^ nb::self);
        vec.def(nb::self ^ value_type());
        vec.def(value_type() ^ nb::self);

        vec.def(nb::self %= nb::self);
        vec.def(nb::self %= value_type());

        vec.def(nb::self <<= nb::self);
        vec.def(nb::self <<= value_type());

        vec.def(nb::self >>= nb::self);
        vec.def(nb::self >>= value_type());

        vec.def(nb::self |= nb::self);
        vec.def(nb::self |= value_type());

        vec.def(nb::self &= nb::self);
        vec.def(nb::self &= value_type());

        vec.def(nb::self ^= nb::self);
        vec.def(nb::self ^= value_type());
    }

    if constexpr (boolean<value_type>) {
        // vec.def(nb::self || nb::self);
        // vec.def(nb::self || value_type());
        // vec.def(value_type() || nb::self);
        // vec.def(nb::self && nb::self);
        // vec.def(nb::self && value_type());
        // vec.def(value_type() && nb::self);

        // The vector comparisons operators use component-wise comparisons.
        // In order to compare lists of vectors in Python, we need to
        // implement the `bool` operator, which returns `True` if all
        // components are `True`.
        vec.def("__bool__", [](const T& self) { return all(self); });
    }

    vec.def(nb::self == nb::self);
    vec.def(nb::self == value_type());
    vec.def(value_type() == nb::self);
    vec.def(nb::self != nb::self);
    vec.def(nb::self != value_type());
    vec.def(value_type() != nb::self);

    if constexpr (arithmetic<value_type> && !boolean<value_type>) {
        vec.def(nb::self < nb::self);
        vec.def(nb::self < value_type());
        vec.def(value_type() < nb::self);
        vec.def(nb::self > nb::self);
        vec.def(nb::self > value_type());
        vec.def(value_type() > nb::self);
        vec.def(nb::self <= nb::self);
        vec.def(nb::self <= value_type());
        vec.def(value_type() <= nb::self);
        vec.def(nb::self >= nb::self);
        vec.def(nb::self >= value_type());
        vec.def(value_type() >= nb::self);
    }

    // Intrinsics

#define WRAP_INTRINSIC_X(name) [](const T& x) { return name(x); }, "x"_a
#define WRAP_INTRINSIC_XY(name) [](const T& x, const T& y) { return name(x, y); }, "x"_a, "y"_a
#define WRAP_INTRINSIC_YX(name) [](const T& y, const T& x) { return name(y, x); }, "y"_a, "x"_a

    if constexpr (BindIntrinsics) {

        // Reductions

        if constexpr (boolean<value_type>) {
            m.def("any", WRAP_INTRINSIC_X(any));
            m.def("all", WRAP_INTRINSIC_X(all));
            m.def("none", WRAP_INTRINSIC_X(none));
        }

        // Basic functions

        if constexpr (arithmetic<value_type>) {
            m.def("min", WRAP_INTRINSIC_XY(min));
            m.def("max", WRAP_INTRINSIC_XY(max));
            m.def(
                "clamp",
                [](const T& x, const T& min_, const T& max_) { return clamp(x, min_, max_); },
                "x"_a,
                "min"_a,
                "max"_a
            );
        }

        if constexpr (signed_number<value_type>) {
            m.def("abs", WRAP_INTRINSIC_X(abs));
            m.def("sign", WRAP_INTRINSIC_X(sign));
        }

        // Floating point checks

        if constexpr (floating_point<value_type>) {
            m.def("isfinite", WRAP_INTRINSIC_X(isfinite));
            m.def("isinf", WRAP_INTRINSIC_X(isinf));
            m.def("isnan", WRAP_INTRINSIC_X(isnan));
        }

        // Rounding

        if constexpr (floating_point<value_type>) {
            m.def("floor", WRAP_INTRINSIC_X(floor));
            m.def("ceil", WRAP_INTRINSIC_X(ceil));
            m.def("trunc", WRAP_INTRINSIC_X(trunc));
            m.def("round", WRAP_INTRINSIC_X(round));
        }

        // Exponential

        if constexpr (floating_point<value_type>) {
            m.def("pow", WRAP_INTRINSIC_XY(pow));
            m.def("sqrt", WRAP_INTRINSIC_X(sqrt));
            m.def("rsqrt", WRAP_INTRINSIC_X(rsqrt));
            m.def("exp", WRAP_INTRINSIC_X(exp));
            m.def("exp2", WRAP_INTRINSIC_X(exp2));
            m.def("log", WRAP_INTRINSIC_X(log));
            m.def("log2", WRAP_INTRINSIC_X(log2));
            m.def("log10", WRAP_INTRINSIC_X(log10));
        }

        // Trigonometry

        if constexpr (floating_point<value_type>) {
            m.def("radians", WRAP_INTRINSIC_X(radians));
            m.def("degrees", WRAP_INTRINSIC_X(degrees));
            m.def("sin", WRAP_INTRINSIC_X(sin));
            m.def("cos", WRAP_INTRINSIC_X(cos));
            m.def("tan", WRAP_INTRINSIC_X(tan));
            m.def("asin", WRAP_INTRINSIC_X(asin));
            m.def("acos", WRAP_INTRINSIC_X(acos));
            m.def("atan", WRAP_INTRINSIC_X(atan));
            m.def("atan2", WRAP_INTRINSIC_YX(atan2));
            m.def("sinh", WRAP_INTRINSIC_X(sinh));
            m.def("cosh", WRAP_INTRINSIC_X(cosh));
            m.def("tanh", WRAP_INTRINSIC_X(tanh));
        }

        // Misc

        if constexpr (floating_point<value_type>) {
            m.def("fmod", WRAP_INTRINSIC_XY(fmod));
            m.def("frac", WRAP_INTRINSIC_X(frac));
            m.def(
                "lerp",
                [](const T& x, const T& y, const T& s) { return lerp(x, y, s); },
                "x"_a,
                "y"_a,
                "s"_a
            );
            m.def(
                "lerp",
                [](const T& x, const T& y, const value_type& s) { return lerp(x, y, s); },
                "x"_a,
                "y"_a,
                "s"_a
            );
            m.def("rcp", WRAP_INTRINSIC_X(rcp));
            m.def("saturate", WRAP_INTRINSIC_X(saturate));
            m.def(
                "smoothstep",
                [](const T& min_, const T& max_, const T& x) { return smoothstep(min_, max_, x); },
                "min"_a,
                "max"_a,
                "x"_a
            );
            m.def("step", WRAP_INTRINSIC_XY(step));
        }

        if constexpr (arithmetic<value_type> && !boolean<value_type>) {
            m.def("dot", WRAP_INTRINSIC_XY(dot));
            if constexpr (dimension == 3)
                m.def("cross", WRAP_INTRINSIC_XY(cross));
        }

        if constexpr (floating_point<value_type>) {
            m.def("length", WRAP_INTRINSIC_X(length));
            m.def("normalize", WRAP_INTRINSIC_X(normalize));
            m.def(
                "reflect",
                [](const T& i, const T& n) { return reflect(i, n); },
                "i"_a,
                "n"_a
            );
        }
    }

#undef WRAP_INTRINSIC_X
#undef WRAP_INTRINSIC_XY
#undef WRAP_INTRINSIC_YX
}

inline void bind_vector(nb::module_& m)
{
    bind_vector_type<float1>(m, "float1");
    bind_vector_type<float2>(m, "float2");
    bind_vector_type<float3>(m, "float3");
    bind_vector_type<float4>(m, "float4");
    bind_vector_type<float16_t1, false>(m, "float16_t1");
    bind_vector_type<float16_t2, false>(m, "float16_t2");
    bind_vector_type<float16_t3, false>(m, "float16_t3");
    bind_vector_type<float16_t4, false>(m, "float16_t4");
    bind_vector_type<uint1>(m, "uint1");
    bind_vector_type<uint2>(m, "uint2");
    bind_vector_type<uint3>(m, "uint3");
    bind_vector_type<uint4>(m, "uint4");
    bind_vector_type<int1>(m, "int1");
    bind_vector_type<int2>(m, "int2");
    bind_vector_type<int3>(m, "int3");
    bind_vector_type<int4>(m, "int4");
    bind_vector_type<bool1>(m, "bool1");
    bind_vector_type<bool2>(m, "bool2");
    bind_vector_type<bool3>(m, "bool3");
    bind_vector_type<bool4>(m, "bool4");

#define WRAP_INTRINSIC_X(name, type) [](const type& x) { return name(x); }, "x"_a

    m.def("f16tof32", WRAP_INTRINSIC_X(f16tof32, uint2));
    m.def("f16tof32", WRAP_INTRINSIC_X(f16tof32, uint3));
    m.def("f16tof32", WRAP_INTRINSIC_X(f16tof32, uint4));
    m.def("f32tof16", WRAP_INTRINSIC_X(f32tof16, float2));
    m.def("f32tof16", WRAP_INTRINSIC_X(f32tof16, float3));
    m.def("f32tof16", WRAP_INTRINSIC_X(f32tof16, float4));

    m.def("asuint", WRAP_INTRINSIC_X(asuint, float2));
    m.def("asuint", WRAP_INTRINSIC_X(asuint, float3));
    m.def("asuint", WRAP_INTRINSIC_X(asuint, float4));

#undef WRAP_INTRINSIC_X
}

} // namespace sgl::math

SGL_PY_EXPORT(math_vector)
{
    nb::module_ math = nb::module_::import_("slangpy.math");

    sgl::math::bind_vector(math);

    m.attr("float1") = math.attr("float1");
    m.attr("float2") = math.attr("float2");
    m.attr("float3") = math.attr("float3");
    m.attr("float4") = math.attr("float4");
    m.attr("float16_t1") = math.attr("float16_t1");
    m.attr("float16_t2") = math.attr("float16_t2");
    m.attr("float16_t3") = math.attr("float16_t3");
    m.attr("float16_t4") = math.attr("float16_t4");
    m.attr("uint1") = math.attr("uint1");
    m.attr("uint2") = math.attr("uint2");
    m.attr("uint3") = math.attr("uint3");
    m.attr("uint4") = math.attr("uint4");
    m.attr("int1") = math.attr("int1");
    m.attr("int2") = math.attr("int2");
    m.attr("int3") = math.attr("int3");
    m.attr("int4") = math.attr("int4");
    m.attr("bool1") = math.attr("bool1");
    m.attr("bool2") = math.attr("bool2");
    m.attr("bool3") = math.attr("bool3");
    m.attr("bool4") = math.attr("bool4");
}
