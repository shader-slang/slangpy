// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/math/posrotscale.h"
#include "sgl/math/posrot.h"
#include "sgl/core/traits.h"

#include "math/primitivetype.h"

#include <array>
#include <type_traits>

namespace sgl::math {

template<typename T>
void bind_posrotscale_type(nb::module_& m, const char* name)
{
    using value_type = typename T::value_type;

    nb::class_<T> posrotscale(m, name);

    // Constructors

    posrotscale.def(nb::init<>());
    posrotscale.def(nb::init<vector<value_type, 3>>(), "position"_a);
    posrotscale.def(nb::init<quat<value_type>>(), "rotation"_a);
    // Scale constructor - commented out due to ambiguity with position constructor
    // posrotscale.def(nb::init<vector<value_type, 3>>(), "scale"_a);
    posrotscale.def(nb::init<vector<value_type, 3>, quat<value_type>>(), "position"_a, "rotation"_a);
    posrotscale.def(nb::init<vector<value_type, 3>, vector<value_type, 3>>(), "position"_a, "scale"_a);
    posrotscale.def(nb::init<quat<value_type>, vector<value_type, 3>>(), "rotation"_a, "scale"_a);
    posrotscale.def(
        nb::init<vector<value_type, 3>, quat<value_type>, vector<value_type, 3>>(),
        "position"_a,
        "rotation"_a,
        "scale"_a
    );
    posrotscale.def(nb::init<posrot<value_type>>(), "posrot"_a);
    posrotscale.def(nb::init_implicit<std::array<value_type, 10>>(), "a"_a);

    posrotscale.def_static("identity", []() { return T::identity(); });

    // Field access

    posrotscale.def_rw("pos", &T::pos, "Position component");
    posrotscale.def_rw("rot", &T::rot, "Rotation component");
    posrotscale.def_rw("scale", &T::scale, "Scale component");


    // Operators

    posrotscale.def(nb::self == nb::self);
    posrotscale.def(nb::self != nb::self);

    // Transform operations
    posrotscale.def(nb::self * nb::self, "Multiply transforms (concatenation)");
    posrotscale.def(nb::self * posrot<value_type>(), "Multiply with posrot");
    posrotscale.def(posrot<value_type>() * nb::self, "Multiply posrot with posrotscale");


    // Math functions
    posrotscale.def(
        "inverse",
        [](const T& self) { return inverse(self); },
        "Get inverse transform"
    );
    posrotscale.def(
        "normalize",
        [](const T& self) { return normalize(self); },
        "Normalize rotation component"
    );
    posrotscale.def(
        "lerp",
        [](const T& self, const T& other, value_type t) { return lerp(self, other, t); },
        "Linear interpolation"
    );
    posrotscale.def(
        "slerp",
        [](const T& self, const T& other, value_type t) { return slerp(self, other, t); },
        "Spherical linear interpolation"
    );

    // Matrix conversions
    posrotscale.def(
        "to_matrix3x4",
        [](const T& self) { return matrix_from_posrotscale_3x4(self); },
        "Convert to 3x4 matrix"
    );
    posrotscale.def(
        "to_matrix4x4",
        [](const T& self) { return matrix_from_posrotscale(self); },
        "Convert to 4x4 matrix"
    );

    // String representation

    posrotscale.def(
        "__repr__",
        [name](const T& self)
        { return fmt::format("{}(pos={}, rot={}, scale={})", name, self.pos, self.rot, self.scale); }
    );
}

void bind_posrotscale(nb::module_& m)
{
    bind_posrotscale_type<posrotscalef>(m, "posrotscalef");

    // Module-level functions
    m.def(
        "inverse",
        [](const posrotscalef& p) { return inverse(p); },
        "posrotscale"_a
    );
    m.def(
        "lerp",
        [](const posrotscalef& a, const posrotscalef& b, float t) { return lerp(a, b, t); },
        "a"_a,
        "b"_a,
        "t"_a
    );
    m.def(
        "slerp",
        [](const posrotscalef& a, const posrotscalef& b, float t) { return slerp(a, b, t); },
        "a"_a,
        "b"_a,
        "t"_a
    );
    m.def(
        "normalize",
        [](const posrotscalef& p) { return normalize(p); },
        "posrotscale"_a
    );
    // Matrix conversion functions
    m.def(
        "matrix_from_posrotscale_3x4",
        [](const posrotscalef& p) { return matrix_from_posrotscale_3x4(p); },
        "posrotscale"_a
    );
    m.def(
        "matrix_from_posrotscale",
        [](const posrotscalef& p) { return matrix_from_posrotscale(p); },
        "posrotscale"_a
    );
    m.def(
        "posrotscale_from_matrix3x4",
        [](const matrix<float, 3, 4>& m) { return posrotscale_from_matrix3x4(m); },
        "matrix"_a
    );
    m.def(
        "posrotscale_from_matrix4x4",
        [](const matrix<float, 4, 4>& m) { return posrotscale_from_matrix4x4(m); },
        "matrix"_a
    );

    // Factory functions
    m.def(
        "posrotscale_from_translation",
        [](const vector<float, 3>& pos) { return posrotscale_from_translation(pos); },
        "position"_a
    );
    m.def(
        "posrotscale_from_rotation",
        [](const quatf& rot) { return posrotscale_from_rotation(rot); },
        "rotation"_a
    );
    m.def(
        "posrotscale_from_scaling",
        [](const vector<float, 3>& scale) { return posrotscale_from_scaling(scale); },
        "scale"_a
    );
    m.def(
        "posrotscale_from_uniform_scaling",
        [](float factor) { return posrotscale_from_uniform_scaling(factor); },
        "factor"_a
    );
    m.def(
        "posrotscale_from_pos_rot",
        [](const vector<float, 3>& pos, const quatf& rot) { return posrotscale_from_pos_rot(pos, rot); },
        "position"_a,
        "rotation"_a
    );
    m.def(
        "posrotscale_from_pos_rot_scale",
        [](const vector<float, 3>& pos, const quatf& rot, const vector<float, 3>& scale)
        { return posrotscale_from_pos_rot_scale(pos, rot, scale); },
        "position"_a,
        "rotation"_a,
        "scale"_a
    );
    m.def(
        "posrotscale_from_posrot",
        [](const posrotf& pr) { return posrotscale_from_posrot(pr); },
        "posrot"_a
    );

    // Transform functions
    m.def(
        "transform_point",
        [](const posrotscalef& transform, const vector<float, 3>& point) { return transform_point(transform, point); },
        "transform"_a,
        "point"_a
    );
    m.def(
        "transform_vector",
        [](const posrotscalef& transform, const vector<float, 3>& vec) { return transform_vector(transform, vec); },
        "transform"_a,
        "vector"_a
    );

    // Conversion functions
    m.def(
        "posrot_from_posrotscale",
        [](const posrotscalef& transform) { return posrot_from_posrotscale(transform); },
        "posrotscale"_a
    );
}

} // namespace sgl::math

SGL_PY_EXPORT(math_posrotscale)
{
    nb::module_ math = nb::module_::import_("slangpy.math");

    sgl::math::bind_posrotscale(math);

    m.attr("posrotscalef") = math.attr("posrotscalef");
}
