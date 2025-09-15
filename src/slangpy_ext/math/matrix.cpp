// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/math/matrix.h"
#include "sgl/core/traits.h"

#include "math/primitivetype.h"
#include "sgl/math/matrix_types.h"

#include <array>
#include <type_traits>

namespace sgl::math {

template<typename T>
void bind_matrix_type(nb::module_& m, const char* name)
{
    nb::class_<T> mat(m, name);

    auto constexpr rows = T::rows;
    auto constexpr cols = T::cols;
    using value_type = typename T::value_type;
    using row_type = typename T::row_type;
    using col_type = typename T::col_type;

    // Constructors

    mat.def(nb::init<>());

    if constexpr (rows == 4 && cols == 4) {
        mat.def(nb::init<matrix<value_type, 3, 3>>());
        mat.def(nb::init<matrix<value_type, 3, 4>>());
    }
    if constexpr (rows == 3 && cols == 4) {
        mat.def(nb::init<matrix<value_type, 3, 3>>());
        mat.def(nb::init<matrix<value_type, 4, 4>>());
    }
    if constexpr (rows == 3 && cols == 3) {
        mat.def(nb::init<matrix<value_type, 4, 4>>());
        mat.def(nb::init<matrix<value_type, 3, 4>>());
    }

    mat.def(nb::init<std::array<value_type, rows * cols>>());

    // Initialization from ndarray.
    mat.def(
        "__init__",
        [](T* self, nb::ndarray<value_type, nb::shape<rows, cols>> a)
        {
            new (self) T();
            for (int c = 0; c < cols; ++c)
                for (int r = 0; r < rows; ++r)
                    (*self)[r][c] = a(r, c);
        }
    );
    nb::implicitly_convertible<nb::ndarray<value_type, nb::shape<rows, cols>>, T>();

    mat.def_static("zeros", &T::zeros);
    mat.def_static("identity", &T::identity);

    // Field access

    mat.def(
        "__getitem__",
        [](const T& self, Py_ssize_t i)
        {
            i = detail::sanitize_getitem_index(i, rows);
            return self[int(i)];
        }
    );
    mat.def(
        "__setitem__",
        [](T& self, int i, const row_type& v)
        {
            if (i > rows)
                throw nb::index_error();
            self[int(i)] = v;
        }
    );

    mat.def(
        "__getitem__",
        [](const T& self, std::array<Py_ssize_t, 2> ij)
        {
            ij[0] = detail::sanitize_getitem_index(ij[0], rows);
            ij[1] = detail::sanitize_getitem_index(ij[1], cols);
            return self[int(ij[0])][int(ij[1])];
        }
    );
    mat.def(
        "__setitem__",
        [](T& self, std::array<int, 2> ij, const value_type& v)
        {
            if (ij[0] > rows || ij[1] > cols)
                throw nb::index_error();
            self[ij[0]][ij[1]] = v;
        }
    );

    mat.def("get_row", nb::overload_cast<int>(&T::get_row, nb::const_), "row"_a);
    mat.def("set_row", &T::set_row, "row"_a, "value"_a);
    mat.def("get_col", &T::get_col, "col"_a);
    mat.def("set_col", &T::set_col, "col"_a, "value"_a);

    mat.def_prop_ro(
        "shape",
        [rows, cols](const T& self)
        {
            SGL_UNUSED(self);
            return nb::make_tuple(rows, cols);
        }
    );

    mat.def_prop_ro(
        "element_type",
        [](const T& self)
        {
            SGL_UNUSED(self);
            return nb::handle(PrimitiveType<value_type>::python_type());
        }
    );

    // Conversion

    mat.def(
        "__dlpack__",
        [](nb::handle_t<T> self)
        {
            const T& m = nb::cast<const T&>(self);
            const size_t shape[2] = {T::rows, T::cols};
            return nb::ndarray<value_type>((void*)(m.data()), 2, shape, self);
        }
    );

    mat.def(
        "to_numpy",
        [](const T& self)
        {
            size_t shape[2] = {rows, cols};
            return nb::ndarray<nb::numpy, const value_type, nb::shape<rows, cols>>(&self, 2, shape, nb::handle());
        }
    );

    auto to_string_ = [](const T& self) { return to_string(self); };
    mat.def("__repr__", to_string_);
    mat.def("__str__", to_string_);

    // Operators

    mat.def(nb::self == nb::self);
    mat.def(nb::self != nb::self);

    // Intrinsics

    m.def(
        "transpose",
        [](const T& x) { return transpose(x); },
        "x"_a
    );

    if constexpr (rows == cols) {
        m.def(
            "determinant",
            [](const T& x) { return determinant(x); },
            "x"_a
        );
        m.def(
            "inverse",
            [](const T& x) { return inverse(x); },
            "x"_a
        );
    }

    m.def(
        "mul",
        [](const matrix<value_type, rows, cols>& x, const matrix<value_type, cols, rows>& y) { return mul(x, y); },
        "x"_a,
        "y"_a
    );
    m.def(
        "mul",
        [](const T& x, const row_type& y) { return mul(x, y); },
        "x"_a,
        "y"_a
    );
    m.def(
        "mul",
        [](const col_type& x, const T& y) { return mul(x, y); },
        "x"_a,
        "y"_a
    );
}

inline void bind_matrix(nb::module_& m)
{
    bind_matrix_type<float2x2>(m, "float2x2");
    bind_matrix_type<float2x3>(m, "float2x3");
    bind_matrix_type<float2x4>(m, "float2x4");

    bind_matrix_type<float3x2>(m, "float3x2");
    bind_matrix_type<float3x3>(m, "float3x3");
    bind_matrix_type<float3x4>(m, "float3x4");

    bind_matrix_type<float4x2>(m, "float4x2");
    bind_matrix_type<float4x3>(m, "float4x3");
    bind_matrix_type<float4x4>(m, "float4x4");

    m.def(
        "transform_point",
        [](const float4x4& m, const float3& v) { return transform_point(m, v); },
        "m"_a,
        "v"_a
    );
    m.def(
        "transform_vector",
        [](const float3x3& m, const float3& v) { return transform_vector(m, v); },
        "m"_a,
        "v"_a
    );
    m.def(
        "transform_vector",
        [](const float4x4& m, const float3& v) { return transform_vector(m, v); },
        "m"_a,
        "v"_a
    );

    m.def(
        "translate",
        [](const float4x4& m, const float3& v) { return translate(m, v); },
        "m"_a,
        "v"_a
    );
    m.def(
        "rotate",
        [](const float4x4& m, float angle, const float3& axis) { return rotate(m, angle, axis); },
        "m"_a,
        "angle"_a,
        "axis"_a
    );
    m.def(
        "scale",
        [](const float4x4& m, const float3& v) { return scale(m, v); },
        "m"_a,
        "v"_a
    );

    m.def(
        "perspective",
        [](float fovy, float aspect, float z_near, float z_far) { return perspective(fovy, aspect, z_near, z_far); },
        "fovy"_a,
        "aspect"_a,
        "z_near"_a,
        "z_far"_a
    );
    m.def(
        "ortho",
        [](float left, float right, float bottom, float top, float z_near, float z_far)
        { return ortho(left, right, bottom, top, z_near, z_far); },
        "left"_a,
        "right"_a,
        "bottom"_a,
        "top"_a,
        "z_near"_a,
        "z_far"_a
    );

    m.def(
        "matrix_from_translation",
        [](const float3& v) { return matrix_from_translation(v); },
        "v"_a
    );

    m.def(
        "matrix_from_scaling",
        [](const float3& v) { return matrix_from_scaling(v); },
        "v"_a
    );

    m.def(
        "matrix_from_rotation",
        [](float angle, const float3& axis) { return matrix_from_rotation(angle, axis); },
        "angle"_a,
        "axis"_a
    );

    m.def(
        "matrix_from_rotation_x",
        [](float angle) { return matrix_from_rotation_x(angle); },
        "angle"_a
    );
    m.def(
        "matrix_from_rotation_y",
        [](float angle) { return matrix_from_rotation_y(angle); },
        "angle"_a
    );
    m.def(
        "matrix_from_rotation_z",
        [](float angle) { return matrix_from_rotation_z(angle); },
        "angle"_a
    );
    m.def(
        "matrix_from_rotation_xyz",
        [](float angle_x, float angle_y, float angle_z) { return matrix_from_rotation_xyz(angle_x, angle_y, angle_z); },
        "angle_x"_a,
        "angle_y"_a,
        "angle_z"_a
    );
    m.def(
        "matrix_from_rotation_xyz",
        [](const float3& angles) { return matrix_from_rotation_xyz(angles); },
        "angles"_a
    );
    m.def(
        "matrix_from_look_at",
        [](const float3& eye, const float3& center, const float3& up, Handedness handedness)
        { return matrix_from_look_at(eye, center, up, handedness); },
        "eye"_a,
        "center"_a,
        "up"_a,
        "handedness"_a = Handedness::right_handed
    );
    m.def(
        "matrix_from_quat",
        [](const quatf& q) { return matrix_from_quat(q); },
        "q"_a
    );
    m.def(
        "matrix_4x4_from_3x4",
        [](const float3x4& m) { return matrix_4x4_from_3x4(m); },
        "m"_a
    );
    m.def(
        "decompose",
        [](const float4x4& model_matrix,
           float3& scale,
           quatf& orientation,
           float3& translation,
           float3& skew,
           float4& perspective) { return decompose(model_matrix, scale, orientation, translation, skew, perspective); },
        "model_matrix"_a,
        "scale"_a,
        "orientation"_a,
        "translation"_a,
        "skew"_a,
        "perspective"_a
    );
}

} // namespace sgl::math

SGL_PY_EXPORT(math_matrix)
{
    nb::module_ math = nb::module_::import_("slangpy.math");

    sgl::math::bind_matrix(math);

    m.attr("float2x2") = math.attr("float2x2");
    m.attr("float2x3") = math.attr("float2x3");
    m.attr("float2x4") = math.attr("float2x4");

    m.attr("float3x2") = math.attr("float3x2");
    m.attr("float3x3") = math.attr("float3x3");
    m.attr("float3x4") = math.attr("float3x4");

    m.attr("float4x2") = math.attr("float4x2");
    m.attr("float4x3") = math.attr("float4x3");
    m.attr("float4x4") = math.attr("float4x4");
}
