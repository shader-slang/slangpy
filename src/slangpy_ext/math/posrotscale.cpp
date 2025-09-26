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
    posrotscale.def_static(
        "translation",
        [](const vector<value_type, 3>& pos) { return T::translation(pos); },
        "position"_a
    );
    posrotscale.def_static(
        "from_rotation",
        [](const quat<value_type>& rot) { return T::rotation(rot); },
        "rotation"_a
    );
    posrotscale.def_static(
        "from_scaling",
        [](const vector<value_type, 3>& scale) { return T::scaling(scale); },
        "scale"_a
    );
    posrotscale.def_static(
        "uniform_scaling",
        [](value_type factor) { return T::uniform_scaling(factor); },
        "factor"_a
    );

    // Field access

    posrotscale.def_rw("pos", &T::pos, "Position component");
    posrotscale.def_rw("rot", &T::rot, "Rotation component");
    posrotscale.def_rw("scale", &T::scale, "Scale component");

    // Property access
    posrotscale.def("position", nb::overload_cast<>(&T::position), "Get position", nb::rv_policy::reference_internal);
    posrotscale.def("rotation", nb::overload_cast<>(&T::rotation), "Get rotation", nb::rv_policy::reference_internal);
    posrotscale.def("scaling", nb::overload_cast<>(&T::scaling), "Get scale", nb::rv_policy::reference_internal);

    // Conversions
    posrotscale.def("to_posrot", &T::to_posrot, "Convert to posrot (ignoring scale)");

    // Operators

    posrotscale.def(nb::self == nb::self);
    posrotscale.def(nb::self != nb::self);

    // Transform operations
    posrotscale.def(nb::self * nb::self, "Multiply transforms (concatenation)");
    posrotscale.def(nb::self * posrot<value_type>(), "Multiply with posrot");
    posrotscale.def(posrot<value_type>() * nb::self, "Multiply posrot with posrotscale");
    posrotscale.def(nb::self * vector<value_type, 3>(), "Transform a point");

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
        [](const T& self) { return to_matrix3x4(self); },
        "Convert to 3x4 matrix"
    );
    posrotscale.def(
        "to_matrix4x4",
        [](const T& self) { return to_matrix4x4(self); },
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
    m.def(
        "to_matrix3x4",
        [](const posrotscalef& p) { return to_matrix3x4(p); },
        "posrotscale"_a
    );
    m.def(
        "to_matrix4x4",
        [](const posrotscalef& p) { return to_matrix4x4(p); },
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
}

} // namespace sgl::math

SGL_PY_EXPORT(math_posrotscale)
{
    nb::module_ math = nb::module_::import_("slangpy.math");

    sgl::math::bind_posrotscale(math);

    m.attr("posrotscalef") = math.attr("posrotscalef");
}
