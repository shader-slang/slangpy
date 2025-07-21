// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/math/quaternion.h"
#include "sgl/core/traits.h"

#include "math/primitivetype.h"

#include <array>
#include <type_traits>

namespace sgl::math {

template<typename T>
void bind_quaternion_type(nb::module_& m, const char* name)
{
    using value_type = typename T::value_type;

    nb::class_<T> quat(m, name);

    // Constructors

    quat.def(nb::init<>());
    quat.def(nb::init<value_type, value_type, value_type, value_type>(), "x"_a, "y"_a, "z"_a, "w"_a);
    quat.def(nb::init<vector<value_type, 3>, value_type>(), "xyz"_a, "w"_a);
    quat.def(nb::init_implicit<std::array<value_type, 4>>(), "a"_a);

    quat.def_static("identity", &T::identity);

    // Field access

    quat.def_rw("x", &T::x);
    quat.def_rw("y", &T::y);
    quat.def_rw("z", &T::z);
    quat.def_rw("w", &T::w);

    quat.def(
        "__getitem__",
        [](const T& self, Py_ssize_t i)
        {
            i = detail::sanitize_getitem_index(i, 4);
            return self[i];
        }
    );
    quat.def("__setitem__", [](T& self, int i, value_type v) { self[i] = v; });

    quat.def_prop_ro(
        "shape",
        [](const T& self)
        {
            SGL_UNUSED(self);
            return nb::make_tuple(4);
        }
    );

    quat.def_prop_ro(
        "element_type",
        [](const T& self)
        {
            SGL_UNUSED(self);
            return nb::handle(PrimitiveType<value_type>::python_type());
        }
    );

    // Conversion

    auto to_string_ = [](const T& self) { return to_string(self); };
    quat.def("__repr__", to_string_);
    quat.def("__str__", to_string_);

    // Operators

    quat.def(+nb::self);
    quat.def(-nb::self);

    quat.def(nb::self + nb::self);
    quat.def(nb::self + value_type());
    quat.def(value_type() + nb::self);

    quat.def(nb::self - nb::self);
    quat.def(nb::self - value_type());
    quat.def(value_type() - nb::self);

    quat.def(nb::self * value_type());
    quat.def(value_type() * nb::self);

    quat.def(nb::self / value_type());

    quat.def(nb::self == nb::self);
    quat.def(nb::self != nb::self);

    // Multiplication

    m.def(
        "mul",
        [](const T& x, const T& y) { return mul(x, y); },
        "x"_a,
        "y"_a
    );
    m.def(
        "mul",
        [](const T& x, const vector<value_type, 3>& y) { return mul(x, y); },
        "x"_a,
        "y"_a
    );

    m.def(
        "transform_vector",
        [](const T& q, const vector<value_type, 3>& v) { return transform_vector(q, v); },
        "q"_a,
        "v"_a
    );

    // Intrinsics

#define WRAP_INTRINSIC_X(name) [](const T& x) { return name(x); }, "x"_a
#define WRAP_INTRINSIC_XY(name) [](const T& x, const T& y) { return name(x, y); }, "x"_a, "y"_a
#define WRAP_INTRINSIC_XYS(name) [](const T& x, const T& y, value_type s) { return name(x, y, s); }, "x"_a, "y"_a, "s"_a

    m.def("isfinite", WRAP_INTRINSIC_X(isfinite));
    m.def("isinf", WRAP_INTRINSIC_X(isinf));
    m.def("isnan", WRAP_INTRINSIC_X(isnan));

    m.def("dot", WRAP_INTRINSIC_XY(dot));
    m.def("cross", WRAP_INTRINSIC_XY(cross));

    m.def("length", WRAP_INTRINSIC_X(length));
    m.def("normalize", WRAP_INTRINSIC_X(normalize));
    m.def("conjugate", WRAP_INTRINSIC_X(conjugate));
    m.def("inverse", WRAP_INTRINSIC_X(inverse));
    m.def("lerp", WRAP_INTRINSIC_XYS(lerp));
    m.def("slerp", WRAP_INTRINSIC_XYS(slerp));

    m.def("pitch", WRAP_INTRINSIC_X(pitch));
    m.def("yaw", WRAP_INTRINSIC_X(yaw));
    m.def("roll", WRAP_INTRINSIC_X(roll));
    m.def("euler_angles", WRAP_INTRINSIC_X(euler_angles));

#undef WRAP_INTRINSIC_X
#undef WRAP_INTRINSIC_XY
#undef WRAP_INTRINSIC_XYS

    // Construction

    m.def(
        "quat_from_angle_axis",
        [](value_type angle, const vector<value_type, 3>& axis) { return quat_from_angle_axis(angle, axis); },
        "angle"_a,
        "axis"_a
    );

    m.def(
        "quat_from_rotation_between_vectors",
        [](const vector<value_type, 3>& from, const vector<value_type, 3>& to)
        { return quat_from_rotation_between_vectors(from, to); },
        "from_"_a,
        "to"_a
    );

    m.def(
        "quat_from_euler_angles",
        [](const vector<value_type, 3>& angles) { return quat_from_euler_angles(angles); },
        "angles"_a
    );

    m.def(
        "quat_from_matrix",
        [](const matrix<value_type, 3, 3>& m) { return quat_from_matrix(m); },
        "m"_a
    );

    m.def(
        "quat_from_look_at",
        [](const vector<value_type, 3>& dir, const vector<value_type, 3>& up, Handedness handedness)
        { return quat_from_look_at(dir, up, handedness); },
        "dir"_a,
        "up"_a,
        "handedness"_a = Handedness::right_handed
    );
}

inline void bind_quaternion(nb::module_& m)
{
    bind_quaternion_type<quatf>(m, "quatf");
}

} // namespace sgl::math

SGL_PY_EXPORT(math_quaternion)
{
    nb::module_ math = nb::module_::import_("slangpy.math");

    sgl::math::bind_quaternion(math);

    m.attr("quatf") = math.attr("quatf");
}
