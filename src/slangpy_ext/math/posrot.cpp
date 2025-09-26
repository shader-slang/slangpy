// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/math/posrot.h"
#include "sgl/core/traits.h"

#include "math/primitivetype.h"

#include <array>
#include <type_traits>

namespace sgl::math {

template<typename T>
void bind_posrot_type(nb::module_& m, const char* name)
{
    using value_type = typename T::value_type;

    nb::class_<T> posrot(m, name);

    // Constructors

    posrot.def(nb::init<>());
    posrot.def(nb::init<vector<value_type, 3>>(), "position"_a);
    posrot.def(nb::init<quat<value_type>>(), "rotation"_a);
    posrot.def(nb::init<vector<value_type, 3>, quat<value_type>>(), "position"_a, "rotation"_a);
    posrot.def(nb::init_implicit<std::array<value_type, 7>>(), "a"_a);

    posrot.def_static("identity", []() { return T::identity(); });

    // Field access

    posrot.def_rw("pos", &T::pos, "Position component");
    posrot.def_rw("rot", &T::rot, "Rotation component");


    // Operators

    posrot.def(nb::self == nb::self);
    posrot.def(nb::self != nb::self);

    // Transform operations
    posrot.def(nb::self * nb::self, "Multiply transforms (concatenation)");

    // Math functions
    posrot.def(
        "inverse",
        [](const T& self) { return inverse(self); },
        "Get inverse transform"
    );
    posrot.def(
        "normalize",
        [](const T& self) { return normalize(self); },
        "Normalize rotation component"
    );
    posrot.def(
        "lerp",
        [](const T& self, const T& other, value_type t) { return lerp(self, other, t); },
        "Linear interpolation"
    );
    posrot.def(
        "slerp",
        [](const T& self, const T& other, value_type t) { return slerp(self, other, t); },
        "Spherical linear interpolation"
    );

    // Matrix conversions
    posrot.def(
        "to_matrix3x4",
        [](const T& self) { return matrix_from_posrot_3x4(self); },
        "Convert to 3x4 matrix"
    );
    posrot.def(
        "to_matrix4x4",
        [](const T& self) { return matrix_from_posrot(self); },
        "Convert to 4x4 matrix"
    );

    // String representation

    posrot.def(
        "__repr__",
        [name](const T& self) { return fmt::format("{}(pos={}, rot={})", name, self.pos, self.rot); }
    );
}

void bind_posrot(nb::module_& m)
{
    bind_posrot_type<posrotf>(m, "posrotf");

    // Module-level functions
    m.def(
        "inverse",
        [](const posrotf& p) { return inverse(p); },
        "posrot"_a
    );
    m.def(
        "lerp",
        [](const posrotf& a, const posrotf& b, float t) { return lerp(a, b, t); },
        "a"_a,
        "b"_a,
        "t"_a
    );
    m.def(
        "slerp",
        [](const posrotf& a, const posrotf& b, float t) { return slerp(a, b, t); },
        "a"_a,
        "b"_a,
        "t"_a
    );
    m.def(
        "normalize",
        [](const posrotf& p) { return normalize(p); },
        "posrot"_a
    );
    // Matrix conversion functions
    m.def(
        "matrix_from_posrot_3x4",
        [](const posrotf& p) { return matrix_from_posrot_3x4(p); },
        "posrot"_a
    );
    m.def(
        "matrix_from_posrot",
        [](const posrotf& p) { return matrix_from_posrot(p); },
        "posrot"_a
    );
    m.def(
        "posrot_from_matrix3x4",
        [](const matrix<float, 3, 4>& m) { return posrot_from_matrix3x4(m); },
        "matrix"_a
    );
    m.def(
        "posrot_from_matrix4x4",
        [](const matrix<float, 4, 4>& m) { return posrot_from_matrix4x4(m); },
        "matrix"_a
    );

    // Factory functions
    m.def(
        "posrot_from_translation",
        [](const vector<float, 3>& pos) { return posrot_from_translation(pos); },
        "position"_a
    );
    m.def(
        "posrot_from_rotation",
        [](const quatf& rot) { return posrot_from_rotation(rot); },
        "rotation"_a
    );
    m.def(
        "posrot_from_pos_rot",
        [](const vector<float, 3>& pos, const quatf& rot) { return posrot_from_pos_rot(pos, rot); },
        "position"_a,
        "rotation"_a
    );

    // Transform functions
    m.def(
        "transform_point",
        [](const posrotf& transform, const vector<float, 3>& point) { return transform_point(transform, point); },
        "transform"_a,
        "point"_a
    );
    m.def(
        "transform_vector",
        [](const posrotf& transform, const vector<float, 3>& vec) { return transform_vector(transform, vec); },
        "transform"_a,
        "vector"_a
    );
}

} // namespace sgl::math

SGL_PY_EXPORT(math_posrot)
{
    nb::module_ math = nb::module_::import_("slangpy.math");

    sgl::math::bind_posrot(math);

    m.attr("posrotf") = math.attr("posrotf");
}
