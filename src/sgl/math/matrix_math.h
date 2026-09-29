// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Most of this code is derived from the GLM library at https://github.com/g-truc/glm
// License: https://github.com/g-truc/glm/blob/master/copying.txt

#pragma once

#include "sgl/math/matrix_types.h"
#include "sgl/math/vector.h"
#include "sgl/math/quaternion.h"
#include "sgl/core/error.h"
#include "sgl/core/format.h"

#include <cmath>
#include <limits>

namespace sgl::math {

// ----------------------------------------------------------------------------
// Binary operators (component-wise)
// ----------------------------------------------------------------------------

/// Binary * operator
template<typename T, int R, int C>
[[nodiscard]] matrix<T, R, C> operator*(const matrix<T, R, C>& lhs, const T& rhs)
{
    matrix<T, R, C> result;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            result[r][c] = lhs[r][c] * rhs;
    return result;
}

// ----------------------------------------------------------------------------
// Multiplication
// ----------------------------------------------------------------------------

/// Multiply matrix and matrix.
template<typename T, int M, int N, int P>
[[nodiscard]] matrix<T, M, P> mul(const matrix<T, M, N>& lhs, const matrix<T, N, P>& rhs)
{
    matrix<T, M, P> result;
    for (int m = 0; m < M; ++m)
        for (int p = 0; p < P; ++p)
            result[m][p] = dot(lhs.get_row(m), rhs.get_col(p));
    return result;
}
/// Multiply matrix and vector. Vector is treated as a column vector.
template<typename T, int R, int C>
[[nodiscard]] vector<T, R> mul(const matrix<T, R, C>& lhs, const vector<T, C>& rhs)
{
    vector<T, R> result;
    for (int r = 0; r < R; ++r)
        result[r] = dot(lhs.get_row(r), rhs);
    return result;
}

/// Multiply vector and matrix. Vector is treated as a row vector.
template<typename T, int R, int C>
[[nodiscard]] vector<T, C> mul(const vector<T, R>& lhs, const matrix<T, R, C>& rhs)
{
    vector<T, C> result;
    for (int c = 0; c < C; ++c)
        result[c] = dot(lhs, rhs.get_col(c));
    return result;
}

/// Transform a point by a 4x4 matrix. The point is treated as a column vector with a 1 in the 4th component.
template<typename T>
[[nodiscard]] vector<T, 3> transform_point(const matrix<T, 4, 4>& m, const vector<T, 3>& v)
{
    return mul(m, vector<T, 4>(v, T(1))).xyz();
}

/// Transform a vector by a 3x3 matrix.
template<typename T>
[[nodiscard]] vector<T, 3> transform_vector(const matrix<T, 3, 3>& m, const vector<T, 3>& v)
{
    return mul(m, v);
}

/// Transform a vector by a 4x4 matrix. The vector is treated as a column vector with a 0 in the 4th component.
template<typename T>
[[nodiscard]] vector<T, 3> transform_vector(const matrix<T, 4, 4>& m, const vector<T, 3>& v)
{
    return mul(m, vector<T, 4>(v, T(0))).xyz();
}

// ----------------------------------------------------------------------------
// Functions
// ----------------------------------------------------------------------------

/// Transpose a matrix.
template<typename T, int R, int C>
matrix<T, C, R> transpose(const matrix<T, R, C>& m)
{
    matrix<T, C, R> result;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            result[c][r] = m[r][c];
    return result;
}

/// Apply a translation to a 4x4 matrix.
template<typename T>
matrix<T, 4, 4> translate(const matrix<T, 4, 4>& m, const vector<T, 3>& v)
{
    matrix<T, 4, 4> result(m);
    result.set_col(3, m.get_col(0) * v.x + m.get_col(1) * v.y + m.get_col(2) * v.z + m.get_col(3));
    return result;
}

/// Apply a 2D translation to a 3x3 matrix.
template<typename T>
matrix<T, 3, 3> translate_2d(const matrix<T, 3, 3>& m, const vector<T, 2>& v)
{
    matrix<T, 3, 3> result(m);
    result.set_col(2, m.get_col(0) * v.x + m.get_col(1) * v.y + m.get_col(2));
    return result;
}

/// Apply a rotation around an axis to a 4x4 matrix.
template<typename T>
matrix<T, 4, 4> rotate(const matrix<T, 4, 4>& m, T angle, const vector<T, 3>& axis_)
{
    T a = angle;
    T c = cos(a);
    T s = sin(a);

    vector<T, 3> axis(normalize(axis_));
    vector<T, 3> temp((T(1) - c) * axis);

    matrix<T, 4, 4> rotate;
    rotate[0][0] = c + temp[0] * axis[0];
    rotate[0][1] = temp[1] * axis[0] - s * axis[2];
    rotate[0][2] = temp[2] * axis[0] + s * axis[1];

    rotate[1][0] = temp[0] * axis[1] + s * axis[2];
    rotate[1][1] = c + temp[1] * axis[1];
    rotate[1][2] = temp[2] * axis[1] - s * axis[0];

    rotate[2][0] = temp[0] * axis[2] - s * axis[1];
    rotate[2][1] = temp[1] * axis[2] + s * axis[0];
    rotate[2][2] = c + temp[2] * axis[2];

    matrix<T, 4, 4> result;
    result.set_col(0, m.get_col(0) * rotate[0][0] + m.get_col(1) * rotate[1][0] + m.get_col(2) * rotate[2][0]);
    result.set_col(1, m.get_col(0) * rotate[0][1] + m.get_col(1) * rotate[1][1] + m.get_col(2) * rotate[2][1]);
    result.set_col(2, m.get_col(0) * rotate[0][2] + m.get_col(1) * rotate[1][2] + m.get_col(2) * rotate[2][2]);
    result.set_col(3, m.get_col(3));

    return result;
}

/// Apply a 2d rotation to a 3x3 matrix.
template<typename T>
matrix<T, 3, 3> rotate_2d(const matrix<T, 3, 3>& m, T angle)
{
    T c = cos(angle);
    T s = sin(angle);

    matrix<T, 3, 3> result;
    result.set_col(0, m.get_col(0) * c + m.get_col(1) * s);
    result.set_col(1, m.get_col(0) * -s + m.get_col(1) * c);
    result.set_col(2, m.get_col(2));

    return result;
}

/// Apply a scale to a 4x4 matrix.
template<typename T>
matrix<T, 4, 4> scale(const matrix<T, 4, 4>& m, const vector<T, 3>& v)
{
    matrix<T, 4, 4> result;
    result.set_col(0, m.get_col(0) * v[0]);
    result.set_col(1, m.get_col(1) * v[1]);
    result.set_col(2, m.get_col(2) * v[2]);
    result.set_col(3, m.get_col(3));
    return result;
}

/// Apply a scale to a 3x3 matrix.
template<typename T>
matrix<T, 3, 3> scale_2d(const matrix<T, 3, 3>& m, const vector<T, 2>& v)
{
    matrix<T, 3, 3> result;
    result.set_col(0, m.get_col(0) * v[0]);
    result.set_col(1, m.get_col(1) * v[1]);
    result.set_col(2, m.get_col(2));
    return result;
}

/// Compute determinant of a 2x2 matrix.
template<typename T>
[[nodiscard]] inline T determinant(const matrix<T, 2, 2>& m)
{
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

/// Compute determinant of a 3x3 matrix.
template<typename T>
[[nodiscard]] inline T determinant(const matrix<T, 3, 3>& m)
{
    T a = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]);
    T b = m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]);
    T c = m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
    return a - b + c;
}

/// Compute determinant of a 4x4 matrix.
template<typename T>
[[nodiscard]] inline T determinant(const matrix<T, 4, 4>& m)
{
    T sub_factor_00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    T sub_factor_01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    T sub_factor_02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    T sub_factor_03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    T sub_factor_04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    T sub_factor_05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    vector<T, 4> detCof(
        +(m[1][1] * sub_factor_00 - m[1][2] * sub_factor_01 + m[1][3] * sub_factor_02), //
        -(m[1][0] * sub_factor_00 - m[1][2] * sub_factor_03 + m[1][3] * sub_factor_04), //
        +(m[1][0] * sub_factor_01 - m[1][1] * sub_factor_03 + m[1][3] * sub_factor_05), //
        -(m[1][0] * sub_factor_02 - m[1][1] * sub_factor_04 + m[1][2] * sub_factor_05)  //
    );

    return m[0][0] * detCof[0] + m[0][1] * detCof[1] + m[0][2] * detCof[2] + m[0][3] * detCof[3];
}

/// Compute inverse of a 2x2 matrix.
template<typename T>
[[nodiscard]] inline matrix<T, 2, 2> inverse(const matrix<T, 2, 2>& m)
{
    T one_over_det = T(1) / determinant(m);
    return matrix<T, 2, 2>{
        +m[1][1] * one_over_det,
        -m[0][1] * one_over_det, // row 0
        -m[1][0] * one_over_det,
        +m[0][0] * one_over_det // row 1
    };
}

/// Compute inverse of a 3x3 matrix.
template<typename T>
[[nodiscard]] inline matrix<T, 3, 3> inverse(const matrix<T, 3, 3>& m)
{
    T one_over_det = T(1) / determinant(m);

    matrix<T, 3, 3> result;
    result[0][0] = +(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * one_over_det;
    result[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * one_over_det;
    result[0][2] = +(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * one_over_det;
    result[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * one_over_det;
    result[1][1] = +(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * one_over_det;
    result[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * one_over_det;
    result[2][0] = +(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * one_over_det;
    result[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * one_over_det;
    result[2][2] = +(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * one_over_det;
    return result;
}

/// Compute inverse of a 4x4 matrix.
template<typename T>
[[nodiscard]] inline matrix<T, 4, 4> inverse(const matrix<T, 4, 4>& m)
{
    T c00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
    T c02 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
    T c03 = m[2][1] * m[3][2] - m[2][2] * m[3][1];

    T c04 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
    T c06 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
    T c07 = m[1][1] * m[3][2] - m[1][2] * m[3][1];

    T c08 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
    T c10 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
    T c11 = m[1][1] * m[2][2] - m[1][2] * m[2][1];

    T c12 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
    T c14 = m[0][1] * m[3][3] - m[0][3] * m[3][1];
    T c15 = m[0][1] * m[3][2] - m[0][2] * m[3][1];

    T c16 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
    T c18 = m[0][1] * m[2][3] - m[0][3] * m[2][1];
    T c19 = m[0][1] * m[2][2] - m[0][2] * m[2][1];

    T c20 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
    T c22 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
    T c23 = m[0][1] * m[1][2] - m[0][2] * m[1][1];

    vector<T, 4> fac0(c00, c00, c02, c03);
    vector<T, 4> fac1(c04, c04, c06, c07);
    vector<T, 4> fac2(c08, c08, c10, c11);
    vector<T, 4> fac3(c12, c12, c14, c15);
    vector<T, 4> fac4(c16, c16, c18, c19);
    vector<T, 4> fac5(c20, c20, c22, c23);

    vector<T, 4> vec0(m[0][1], m[0][0], m[0][0], m[0][0]);
    vector<T, 4> vec1(m[1][1], m[1][0], m[1][0], m[1][0]);
    vector<T, 4> vec2(m[2][1], m[2][0], m[2][0], m[2][0]);
    vector<T, 4> vec3(m[3][1], m[3][0], m[3][0], m[3][0]);

    vector<T, 4> inv0(vec1 * fac0 - vec2 * fac1 + vec3 * fac2);
    vector<T, 4> inv1(vec0 * fac0 - vec2 * fac3 + vec3 * fac4);
    vector<T, 4> inv2(vec0 * fac1 - vec1 * fac3 + vec3 * fac5);
    vector<T, 4> inv3(vec0 * fac2 - vec1 * fac4 + vec2 * fac5);

    vector<T, 4> sign_a(+1, -1, +1, -1);
    vector<T, 4> sign_b(-1, +1, -1, +1);
    matrix<T, 4, 4> inverse = matrix_from_columns(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b);

    vector<T, 4> row0(inverse[0][0], inverse[0][1], inverse[0][2], inverse[0][3]);

    vector<T, 4> dot0(m.get_col(0) * row0);
    T dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);

    T one_over_det = T(1) / dot1;

    return inverse * one_over_det;
}

/// Compute the (X * Y * Z) euler angles of a 4x4 matrix.
template<typename T>
void extract_euler_angle_xyz(const matrix<T, 4, 4>& m, float& angle_x, float& angle_y, float& angle_z)
{
    T t1 = atan2(m[1][2], m[2][2]);
    T c2 = sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1]);
    T t2 = atan2(-m[0][2], c2);
    T s1 = sin(t1);
    T c1 = cos(t1);
    T t3 = atan2(s1 * m[2][0] - c1 * m[1][0], c1 * m[1][1] - s1 * m[2][1]);
    angle_x = -t1;
    angle_y = -t2;
    angle_z = -t3;
}

/// Decomposes a homogeneous matrix into translation, rotation, scale, shear and perspective.
/// The factors reconstruct model_matrix / model_matrix[3][3] as P * T * R * H * S,
/// where H has unit diagonal and upper entries (H01, H02, H12) = (skew.z, skew.y, skew.x).
/// P has an identity upper three rows and perspective as its last row. Reflections
/// use three negative scales and a proper rotation, matching the existing convention.
/// Returns false for nonfinite input, zero homogeneous weight, numerically dependent
/// spatial columns, or factors outside the output type's range. Outputs are unchanged on failure.
template<typename T>
inline bool decompose(
    const matrix<T, 4, 4>& model_matrix,
    vector<T, 3>& scale,
    quat<T>& orientation,
    vector<T, 3>& translation,
    vector<T, 3>& skew,
    vector<T, 4>& perspective
)
{
    // Double intermediates preserve float input across its exponent range. Equilibrating
    // each spatial column also makes the rank test independent of axis scale and units.
    double local_matrix[4][4];
    const double homogeneous = double(model_matrix[3][3]);
    if (homogeneous == 0.0 || !std::isfinite(homogeneous))
        return false;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            const double value = double(model_matrix[row][col]) / homogeneous;
            if (!std::isfinite(value))
                return false;
            local_matrix[row][col] = value;
        }
    }

    double upper[3][3];
    vector<double, 3> column_sizes;
    for (int col = 0; col < 3; ++col) {
        const double size = std::hypot(local_matrix[0][col], local_matrix[1][col], local_matrix[2][col]);
        if (size == 0.0 || !std::isfinite(size))
            return false;
        column_sizes[col] = size;
        for (int row = 0; row < 3; ++row)
            upper[row][col] = local_matrix[row][col] / size;
    }

    // QR factorization with Givens rotations: normalized spatial columns = rotation * upper.
    // Norms never square unscaled input, and no determinant or general matrix inverse is needed.
    double rotation[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (int col = 0; col < 2; ++col) {
        for (int row = 2; row > col; --row) {
            const double a = upper[row - 1][col];
            const double b = upper[row][col];
            if (b == 0.0)
                continue;
            const double size = std::hypot(a, b);
            const double c = a / size;
            const double s = b / size;
            for (int k = col; k < 3; ++k) {
                const double x = upper[row - 1][k];
                const double y = upper[row][k];
                upper[row - 1][k] = c * x + s * y;
                upper[row][k] = -s * x + c * y;
            }
            upper[row][col] = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double x = rotation[k][row - 1];
                const double y = rotation[k][row];
                rotation[k][row - 1] = c * x + s * y;
                rotation[k][row] = -s * x + c * y;
            }
        }
    }

    // Test angular independence after column equilibration, not the product of axis scales.
    constexpr double RANK_TOLERANCE = 16.0 * std::numeric_limits<double>::epsilon();
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(upper[axis][axis]) <= RANK_TOLERANCE)
            return false;
        if (upper[axis][axis] < 0.0) {
            for (int k = 0; k < 3; ++k) {
                upper[axis][k] = -upper[axis][k];
                rotation[k][axis] = -rotation[k][axis];
            }
        }
    }

    vector<double, 3> result_scale;
    for (int axis = 0; axis < 3; ++axis)
        result_scale[axis] = upper[axis][axis] * column_sizes[axis];
    const vector<double, 3> result_skew(
        upper[1][2] / upper[2][2],
        upper[0][2] / upper[2][2],
        upper[0][1] / upper[1][1]
    );
    const vector<double, 3> result_translation
        = vector<double, 3>(local_matrix[0][3], local_matrix[1][3], local_matrix[2][3]);

    // Solve A^T * p.xyz = last_row.xyz using the scaled QR factors. Unlike forming
    // inverse(A), this does not overflow on tiny axes whose perspective is still finite.
    vector<double, 3> rhs;
    for (int axis = 0; axis < 3; ++axis) {
        double value = local_matrix[3][axis] / column_sizes[axis];
        for (int k = 0; k < axis; ++k)
            value -= upper[k][axis] * rhs[k];
        rhs[axis] = value / upper[axis][axis];
    }
    vector<double, 3> p;
    for (int row = 0; row < 3; ++row)
        p[row] = rotation[row][0] * rhs.x + rotation[row][1] * rhs.y + rotation[row][2] * rhs.z;
    const vector<double, 4> result_perspective(p, 1.0 - dot(result_translation, p));

    vector<double, 3> columns[3];
    for (int col = 0; col < 3; ++col)
        columns[col] = vector<double, 3>(rotation[0][col], rotation[1][col], rotation[2][col]);
    if (dot(columns[0], cross(columns[1], columns[2])) < 0.0) {
        result_scale *= -1.0;
        for (auto& column : columns)
            column *= -1.0;
    }

    // Extract the quaternion from the proper orthogonal factor, retaining the usual sign convention.
    quat<double> result_orientation;
    const double trace = columns[0].x + columns[1].y + columns[2].z;
    if (trace > 0.0) {
        double root = std::sqrt(trace + 1.0);
        result_orientation.w = 0.5 * root;
        root = 0.5 / root;
        result_orientation.x = root * (columns[1].z - columns[2].y);
        result_orientation.y = root * (columns[2].x - columns[0].z);
        result_orientation.z = root * (columns[0].y - columns[1].x);
    } else {
        int i = 0;
        if (columns[1].y > columns[0].x)
            i = 1;
        if (columns[2].z > columns[i][i])
            i = 2;
        const int j = (i + 1) % 3;
        const int k = (j + 1) % 3;
        double root = std::sqrt(columns[i][i] - columns[j][j] - columns[k][k] + 1.0);
        result_orientation[i] = 0.5 * root;
        root = 0.5 / root;
        result_orientation[j] = root * (columns[i][j] + columns[j][i]);
        result_orientation[k] = root * (columns[i][k] + columns[k][i]);
        result_orientation.w = root * (columns[j][k] - columns[k][j]);
    }
    result_orientation = normalize(result_orientation);

    const auto representable = [](double value)
    {
        return std::isfinite(value) && std::abs(value) <= double(std::numeric_limits<T>::max())
            && (value == 0.0 || T(value) != T(0));
    };
    for (int axis = 0; axis < 3; ++axis) {
        if (!representable(result_scale[axis]) || T(result_scale[axis]) == T(0)
            || !representable(result_translation[axis]) || !representable(result_skew[axis]))
            return false;
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (!representable(result_orientation[axis]) || !representable(result_perspective[axis]))
            return false;
    }
    scale = vector<T, 3>(result_scale);
    orientation
        = quat<T>(T(result_orientation.x), T(result_orientation.y), T(result_orientation.z), T(result_orientation.w));
    translation = vector<T, 3>(result_translation);
    skew = vector<T, 3>(result_skew);
    perspective = vector<T, 4>(result_perspective);
    return true;
}

// ----------------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------------

/// Creates a matrix from coefficients in row-major order.
template<typename T, int R, int C>
[[nodiscard]] inline matrix<T, R, C> matrix_from_coefficients(const T* coeffs)
{
    matrix<T, R, C> m;
    std::memcpy(&m, coeffs, sizeof(T) * R * C);
    return m;
}

/// Creates a matrix from column vectors.
template<typename T, int R>
[[nodiscard]] inline matrix<T, R, 1> matrix_from_columns(const vector<T, R>& col0)
{
    matrix<T, R, 1> m;
    m.set_col(0, col0);
    return m;
}

/// Creates a matrix from column vectors.
template<typename T, int R>
[[nodiscard]] inline matrix<T, R, 2> matrix_from_columns(const vector<T, R>& col0, const vector<T, R>& col1)
{
    matrix<T, R, 2> m;
    m.set_col(0, col0);
    m.set_col(1, col1);
    return m;
}

/// Creates a matrix from column vectors.
template<typename T, int R>
[[nodiscard]] inline matrix<T, R, 3>
matrix_from_columns(const vector<T, R>& col0, const vector<T, R>& col1, const vector<T, R>& col2)
{
    matrix<T, R, 3> m;
    m.set_col(0, col0);
    m.set_col(1, col1);
    m.set_col(2, col2);
    return m;
}

/// Creates a matrix from column vectors.
template<typename T, int R>
[[nodiscard]] inline matrix<T, R, 4> matrix_from_columns(
    const vector<T, R>& col0,
    const vector<T, R>& col1,
    const vector<T, R>& col2,
    const vector<T, R>& col3
)
{
    matrix<T, R, 4> m;
    m.set_col(0, col0);
    m.set_col(1, col1);
    m.set_col(2, col2);
    m.set_col(3, col3);
    return m;
}

/// Creates a square matrix from a diagonal vector.
template<typename T, int N>
[[nodiscard]] inline matrix<T, N, N> matrix_from_diagonal(const vector<T, N>& diag)
{
    matrix<T, N, N> m = matrix<T, N, N>::zeros();
    for (int i = 0; i < N; i++)
        m[i][i] = diag[i];
    return m;
}

/// Creates a right-handed perspective projection matrix. Depth is mapped to [0, 1].
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> perspective(T fovy, T aspect, T z_near, T z_far)
{
    SGL_ASSERT(abs(aspect - std::numeric_limits<T>::epsilon()) > T(0));

    T tan_half_fovy = tan(fovy / T(2));

    matrix<T, 4, 4> m = matrix<T, 4, 4>::zeros();
    m[0][0] = T(1) / (aspect * tan_half_fovy);
    m[1][1] = T(1) / (tan_half_fovy);
    m[2][2] = z_far / (z_near - z_far);
    m[3][2] = -T(1);
    m[2][3] = -(z_far * z_near) / (z_far - z_near);
    return m;
}

/// Creates a right-handed orthographic projection matrix. Depth is mapped to [0, 1].
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> ortho(T left, T right, T bottom, T top, T z_near, T z_far)
{
    matrix<T, 4, 4> m = matrix<T, 4, 4>::identity();
    m[0][0] = T(2) / (right - left);
    m[1][1] = T(2) / (top - bottom);
    m[2][2] = -T(1) / (z_far - z_near);
    m[0][3] = -(right + left) / (right - left);
    m[1][3] = -(top + bottom) / (top - bottom);
    m[2][3] = -z_near / (z_far - z_near);
    return m;
}

/// Creates a translation matrix.
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_translation(const vector<T, 3>& v)
{
    matrix<T, 4, 4> m = matrix<T, 4, 4>::identity();
    m[0][3] = v.x;
    m[1][3] = v.y;
    m[2][3] = v.z;
    return m;
}

/// Creates a translation matrix.
template<floating_point T>
[[nodiscard]] inline matrix<T, 3, 3> matrix_from_translation_2d(const vector<T, 2>& v)
{
    matrix<T, 3, 3> m = matrix<T, 3, 3>::identity();
    m[0][2] = v.x;
    m[1][2] = v.y;
    return m;
}


/// Creates a rotation matrix from an angle and an axis.
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_rotation(T angle, const vector<T, 3>& axis)
{
    return rotate(matrix<T, 4, 4>::identity(), angle, axis);
}

/// Creates a 2D rotation matrix from an angle.
template<floating_point T>
[[nodiscard]] inline matrix<T, 3, 3> matrix_from_rotation_2d(T angle)
{
    T c = cos(angle);
    T s = sin(angle);
    matrix<T, 3, 3> m = matrix<T, 3, 3>::identity();
    m[0][0] = c;
    m[0][1] = -s;
    m[1][0] = s;
    m[1][1] = c;
    return m;
}

/// Creates a rotation matrix around the X-axis.
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_rotation_x(T angle)
{
    T c = cos(angle);
    T s = sin(angle);

    // clang-format off
    return matrix<T, 4, 4>{
        T(1),   T(0),   T(0),   T(0),   // row 0
        T(0),   c,      -s,     T(0),   // row 1
        T(0),   s,      c,      T(0),   // row 2
        T(0),   T(0),   T(0),   T(1)    // row 3
    };
    // clang-format on
}

/// Creates a rotation matrix around the Y-axis.
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_rotation_y(T angle)
{
    T c = cos(angle);
    T s = sin(angle);

    // clang-format off
    return matrix<T, 4, 4>{
        c,      T(0),   s,      T(0),   // row 0
        T(0),   T(1),   T(0),   T(0),   // row 1
        -s,     T(0),   c,      T(0),   // row 2
        T(0),   T(0),   T(0),   T(1)    // row 3
    };
    // clang-format on
}

/// Creates a rotation matrix around the Z-axis.
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_rotation_z(T angle)
{
    T c = cos(angle);
    T s = sin(angle);

    // clang-format off
    return matrix<T, 4, 4>{
        c,      -s,     T(0),   T(0),   // row 0
        s,      c,      T(0),   T(0),   // row 1
        T(0),   T(0),   T(1),   T(0),   // row 2
        T(0),   T(0),   T(0),   T(1)    // row 3
    };
    // clang-format on
}

/// Creates a rotation matrix (X * Y * Z).
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_rotation_xyz(T angle_x, T angle_y, T angle_z)
{
    T c1 = cos(-angle_x);
    T c2 = cos(-angle_y);
    T c3 = cos(-angle_z);
    T s1 = sin(-angle_x);
    T s2 = sin(-angle_y);
    T s3 = sin(-angle_z);

    matrix<T, 4, 4> m;
    m[0][0] = c2 * c3;
    m[0][1] = c2 * s3;
    m[0][2] = -s2;
    m[0][3] = T(0);

    m[1][0] = -c1 * s3 + s1 * s2 * c3;
    m[1][1] = c1 * c3 + s1 * s2 * s3;
    m[1][2] = s1 * c2;
    m[1][3] = T(0);

    m[2][0] = s1 * s3 + c1 * s2 * c3;
    m[2][1] = -s1 * c3 + c1 * s2 * s3;
    m[2][2] = c1 * c2;
    m[2][3] = T(0);

    m[3][0] = T(0);
    m[3][1] = T(0);
    m[3][2] = T(0);
    m[3][3] = T(1);

    return m;
}

template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_rotation_xyz(const vector<T, 3>& angles)
{
    return matrix_from_rotation_xyz(angles.x, angles.y, angles.z);
}

/// Creates a scaling matrix.
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_scaling(const vector<T, 3>& v)
{
    matrix<T, 4, 4> m = matrix<T, 4, 4>::identity();
    m[0][0] = v.x;
    m[1][1] = v.y;
    m[2][2] = v.z;
    return m;
}

/// Creates a scaling matrix.
template<floating_point T>
[[nodiscard]] inline matrix<T, 3, 3> matrix_from_scaling_2d(const vector<T, 2>& v)
{
    matrix<T, 3, 3> m = matrix<T, 3, 3>::identity();
    m[0][0] = v.x;
    m[1][1] = v.y;
    return m;
}

/**
 * Build a look-at matrix.
 * If right handed, forward direction is mapped onto -Z axis.
 * If left handed, forward direction is mapped onto +Z axis.
 * \param eye Eye position
 * \param center Center position
 * \param up Up vector
 * \param handedness Coordinate system handedness.
 */
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_look_at(
    const vector<T, 3>& eye,
    const vector<T, 3>& center,
    const vector<T, 3>& up,
    Handedness handedness = Handedness::right_handed
)
{
    vector<T, 3> f(handedness == Handedness::right_handed ? normalize(eye - center) : normalize(center - eye));
    vector<T, 3> r(normalize(cross(up, f)));
    vector<T, 3> u(cross(f, r));

    matrix<T, 4, 4> result = matrix<T, 4, 4>::identity();
    result[0][0] = r.x;
    result[0][1] = r.y;
    result[0][2] = r.z;
    result[1][0] = u.x;
    result[1][1] = u.y;
    result[1][2] = u.z;
    result[2][0] = f.x;
    result[2][1] = f.y;
    result[2][2] = f.z;
    result[0][3] = -dot(r, eye);
    result[1][3] = -dot(u, eye);
    result[2][3] = -dot(f, eye);

    return result;
}

template<floating_point T>
[[nodiscard]] inline matrix<T, 3, 3> matrix_from_quat(const quat<T>& q)
{
    matrix<T, 3, 3> m;
    T qxx(q.x * q.x);
    T qyy(q.y * q.y);
    T qzz(q.z * q.z);
    T qxz(q.x * q.z);
    T qxy(q.x * q.y);
    T qyz(q.y * q.z);
    T qwx(q.w * q.x);
    T qwy(q.w * q.y);
    T qwz(q.w * q.z);

    m[0][0] = T(1) - T(2) * (qyy + qzz);
    m[0][1] = T(2) * (qxy - qwz);
    m[0][2] = T(2) * (qxz + qwy);

    m[1][0] = T(2) * (qxy + qwz);
    m[1][1] = T(1) - T(2) * (qxx + qzz);
    m[1][2] = T(2) * (qyz - qwx);

    m[2][0] = T(2) * (qxz - qwy);
    m[2][1] = T(2) * (qyz + qwx);
    m[2][2] = T(1) - T(2) * (qxx + qyy);

    return m;
}

template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_4x4_from_3x4(const matrix<T, 3, 4>& m)
{
    matrix<T, 4, 4> result;
    for (int r = 0; r < 3; ++r) {
        result.set_row(r, m.get_row(r));
    }
    result[3][3] = T(1);
    return result;
}

template<typename T, int R, int C>
[[nodiscard]] std::string to_string(const matrix<T, R, C>& m)
{
    return ::fmt::format("{}", m);
}

} // namespace sgl::math

template<typename T, int R, int C>
struct std::hash<::sgl::math::matrix<T, R, C>> {
    constexpr size_t operator()(const ::sgl::math::matrix<T, R, C>& m) const
    {
        size_t result = 0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                result ^= std::hash<T>()(m[r][c]) + 0x9e3779b9 + (result << 6) + (result >> 2);
        return result;
    }
};

template<typename T, int R, int C>
struct fmt::formatter<sgl::math::matrix<T, R, C>> : formatter<typename sgl::math::matrix<T, R, C>::row_type> {
    using row_type = typename sgl::math::matrix<T, R, C>::row_type;

    template<typename FormatContext>
    auto format(const sgl::math::matrix<T, R, C>& matrix, FormatContext& ctx) const
    {
        auto out = ctx.out();
        for (int r = 0; r < R; ++r) {
            out = ::fmt::format_to(out, "{}", (r == 0) ? "{" : ", ");
            out = formatter<row_type>::format(matrix.get_row(r), ctx);
        }
        out = fmt::format_to(out, "}}");
        return out;
    }
};
