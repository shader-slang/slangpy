// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Most of this code is derived from the GLM library at https://github.com/g-truc/glm
// License: https://github.com/g-truc/glm/blob/master/copying.txt

#pragma once

#include "sgl/math/matrix_types.h"
#include "sgl/math/vector.h"
#include "sgl/math/quaternion.h"
#include "sgl/core/error.h"
#include "sgl/core/format.h"

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

/// Decomposes a model matrix into translation, rotation and scale components.
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
    // See https://caff.de/posts/4X4-matrix-decomposition/decomposition.pdf

    const T eps = std::numeric_limits<T>::epsilon();

    matrix<T, 4, 4> local_matrix(model_matrix);

    // Abort if zero matrix.
    if (abs(local_matrix[3][3]) < eps)
        return false;

    // Normalize the matrix.
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            local_matrix[i][j] /= local_matrix[3][3];

    // perspective_matrix is used to solve for perspective, but it also provides
    // an easy way to test for singularity of the upper 3x3 component.
    matrix<T, 4, 4> perspective_matrix(local_matrix);
    perspective_matrix[3] = vector<T, 4>(0, 0, 0, 1);
    // Determinant contains scale in each axis. Comparing with eps like GLM would reject scale
    // less than 0.0049 (which is convert from cm to m, and then scale down 2x)
    if (abs(determinant(perspective_matrix)) < eps * eps * eps)
        return false;

    // First, isolate perspective. This is the messiest.
    if (abs(local_matrix[3][0]) >= eps || abs(local_matrix[3][1]) >= eps || abs(local_matrix[3][2]) >= eps) {
        // right_hand_side is the right hand side of the equation.
        vector<T, 4> right_hand_side = local_matrix[3];

        // Solve the equation by inverting perspective_matrix and multiplying
        // right_hand_side by the inverse.
        // (This is the easiest way, not necessarily the best.)
        matrix<T, 4, 4> inverse_perspective_matrix = inverse(perspective_matrix);
        matrix<T, 4, 4> transposed_inverse_perspective_matrix = transpose(inverse_perspective_matrix);

        perspective = mul(transposed_inverse_perspective_matrix, right_hand_side);

        // Clear the perspective partition.
        local_matrix[3] = vector<T, 4>(0, 0, 0, 1);
    } else {
        // No perspective.
        perspective = vector<T, 4>(0, 0, 0, 1);
    }

    // Next take care of translation (easy).
    translation = local_matrix.get_col(3).xyz();
    local_matrix.set_row(3, vector<T, 4>(0, 0, 0, 1));

    vector<T, 3> row[3];

    // Now get scale and shear.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            row[i][j] = local_matrix[j][i];

    // Compute X scale factor and normalize first row.
    scale.x = length(row[0]);
    row[0] = normalize(row[0]);

    // Compute XY shear factor and make 2nd row orthogonal to 1st.
    skew.z = dot(row[0], row[1]);
    row[1] = row[1] - skew.z * row[0];

    // Now, compute Y scale and normalize 2nd row.
    scale.y = length(row[1]);
    row[1] = normalize(row[1]);
    skew.z /= scale.y;

    // Compute XZ and YZ shears, orthogonalize 3rd row.
    skew.y = dot(row[0], row[2]);
    row[2] = row[2] - skew.y * row[0];
    skew.x = dot(row[1], row[2]);
    row[2] = row[2] - skew.x * row[1];

    // Next, get Z scale and normalize 3rd row.
    scale.z = length(row[2]);
    row[2] = normalize(row[2]);
    skew.y /= scale.z;
    skew.x /= scale.z;

    // At this point, the matrix (in rows[]) is orthonormal.
    // Check for a coordinate system flip. If the determinant
    // is -1, then negate the matrix and the scaling factors.
    if (dot(row[0], cross(row[1], row[2])) < T(0)) {
        scale *= T(-1);
        for (int i = 0; i < 3; i++)
            row[i] *= T(-1);
    }

    // Now, get the rotations out, as described in the gem.
    int i, j, k = 0;
    T root, trace = row[0].x + row[1].y + row[2].z;
    if (trace > T(0)) {
        root = sqrt(trace + T(1));
        orientation.w = T(0.5) * root;
        root = T(0.5) / root;
        orientation.x = root * (row[1].z - row[2].y);
        orientation.y = root * (row[2].x - row[0].z);
        orientation.z = root * (row[0].y - row[1].x);
    } // end if > 0
    else {
        static int next[3] = {1, 2, 0};
        i = 0;
        if (row[1].y > row[0].x)
            i = 1;
        if (row[2].z > row[i][i])
            i = 2;
        j = next[i];
        k = next[j];

        root = sqrt(row[i][i] - row[j][j] - row[k][k] + T(1));

        orientation[i] = T(0.5) * root;
        root = T(0.5) / root;
        orientation[j] = root * (row[i][j] + row[j][i]);
        orientation[k] = root * (row[i][k] + row[k][i]);
        orientation.w = root * (row[j][k] - row[k][j]);
    } // end if <= 0

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
    return translate(matrix<T, 4, 4>::identity(), v);
}

/// Creates a rotation matrix from an angle and an axis.
template<floating_point T>
[[nodiscard]] inline matrix<T, 4, 4> matrix_from_rotation(T angle, const vector<T, 3>& axis)
{
    return rotate(matrix<T, 4, 4>::identity(), angle, axis);
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
    return scale(matrix<T, 4, 4>::identity(), v);
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

template<typename T, int R, int C>
[[nodiscard]] std::string to_string(const matrix<T, R, C>& m)
{
    return ::fmt::format("{}", m);
}

template<int R, int C, typename T>
bool lex_lt(const matrix<T, R, C>& lhs, const matrix<T, R, C>& rhs)
{
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            if (lhs[r][c] != rhs[r][c])
                return lhs[r][c] < rhs[r][c];
        }
    }
    return false;
}
} // namespace sgl::math

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
