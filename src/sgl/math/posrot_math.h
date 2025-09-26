// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/math/posrot_types.h"
#include "sgl/math/matrix_types.h"
#include "sgl/math/matrix_math.h"
#include "sgl/math/quaternion_math.h"
#include "sgl/math/vector_math.h"
#include "sgl/core/error.h"

namespace sgl::math {

// ----------------------------------------------------------------------------
// Comparison operators
// ----------------------------------------------------------------------------

/// Equality operator
template<typename T>
[[nodiscard]] constexpr bool operator==(const posrot<T>& lhs, const posrot<T>& rhs) noexcept
{
    return all(lhs.pos == rhs.pos) && all(lhs.rot == rhs.rot);
}

/// Inequality operator
template<typename T>
[[nodiscard]] constexpr bool operator!=(const posrot<T>& lhs, const posrot<T>& rhs) noexcept
{
    return !(lhs == rhs);
}

// ----------------------------------------------------------------------------
// Transform operations
// ----------------------------------------------------------------------------

/// Multiply two transforms (concatenation)
template<typename T>
[[nodiscard]] posrot<T> operator*(const posrot<T>& lhs, const posrot<T>& rhs) noexcept
{
    // First apply rhs, then lhs
    // pos = lhs.pos + lhs.rot * rhs.pos
    // rot = lhs.rot * rhs.rot
    return posrot<T>(lhs.pos + transform_vector(lhs.rot, rhs.pos), mul(lhs.rot, rhs.rot));
}

/// Transform a point by the posrot
template<typename T>
[[nodiscard]] vector<T, 3> operator*(const posrot<T>& transform, const vector<T, 3>& point) noexcept
{
    return transform.pos + transform_vector(transform.rot, point);
}

/// Inverse of a posrot transform
template<typename T>
[[nodiscard]] posrot<T> inverse(const posrot<T>& transform) noexcept
{
    quat<T> inv_rot = conjugate(transform.rot);
    return posrot<T>(-transform_vector(inv_rot, transform.pos), inv_rot);
}

/// Convert posrot to 3x4 matrix
template<typename T>
[[nodiscard]] matrix<T, 3, 4> to_matrix3x4(const posrot<T>& transform) noexcept
{
    matrix<T, 3, 3> rot_matrix = matrix_from_quat(transform.rot);
    matrix<T, 3, 4> result;

    // Copy rotation part
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = rot_matrix[i][j];
        }
    }

    // Set translation part
    result[0][3] = transform.pos.x;
    result[1][3] = transform.pos.y;
    result[2][3] = transform.pos.z;

    return result;
}

/// Convert posrot to 4x4 matrix
template<typename T>
[[nodiscard]] matrix<T, 4, 4> to_matrix4x4(const posrot<T>& transform) noexcept
{
    matrix<T, 3, 3> rot_matrix = matrix_from_quat(transform.rot);
    matrix<T, 4, 4> result = matrix<T, 4, 4>::identity();

    // Copy rotation part
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = rot_matrix[i][j];
        }
    }

    // Set translation part
    result[0][3] = transform.pos.x;
    result[1][3] = transform.pos.y;
    result[2][3] = transform.pos.z;

    return result;
}

/// Create posrot from 3x4 matrix
template<typename T>
[[nodiscard]] posrot<T> posrot_from_matrix3x4(const matrix<T, 3, 4>& m) noexcept
{
    // Extract rotation part
    matrix<T, 3, 3> rot_part;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rot_part[i][j] = m[i][j];
        }
    }

    // Extract translation part
    vector<T, 3> pos{m[0][3], m[1][3], m[2][3]};

    return posrot<T>(pos, quat_from_matrix(rot_part));
}

/// Create posrot from 4x4 matrix
template<typename T>
[[nodiscard]] posrot<T> posrot_from_matrix4x4(const matrix<T, 4, 4>& m) noexcept
{
    // Extract rotation part
    matrix<T, 3, 3> rot_part;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rot_part[i][j] = m[i][j];
        }
    }

    // Extract translation part
    vector<T, 3> pos{m[0][3], m[1][3], m[2][3]};

    return posrot<T>(pos, quat_from_matrix(rot_part));
}

/// Linear interpolation between two posrot transforms
template<typename T>
[[nodiscard]] posrot<T> lerp(const posrot<T>& a, const posrot<T>& b, T t) noexcept
{
    return posrot<T>(lerp(a.pos, b.pos, t), slerp(a.rot, b.rot, t));
}

/// Spherical linear interpolation between two posrot transforms
template<typename T>
[[nodiscard]] posrot<T> slerp(const posrot<T>& a, const posrot<T>& b, T t) noexcept
{
    return posrot<T>(lerp(a.pos, b.pos, t), slerp(a.rot, b.rot, t));
}

/// Normalize the rotation component
template<typename T>
[[nodiscard]] posrot<T> normalize(const posrot<T>& transform) noexcept
{
    return posrot<T>(transform.pos, normalize(transform.rot));
}

} // namespace sgl::math
