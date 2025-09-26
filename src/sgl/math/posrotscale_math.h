// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/math/posrotscale_types.h"
#include "sgl/math/posrot_math.h"
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
[[nodiscard]] constexpr bool operator==(const posrotscale<T>& lhs, const posrotscale<T>& rhs) noexcept
{
    return all(lhs.pos == rhs.pos) && all(lhs.rot == rhs.rot) && all(lhs.scale == rhs.scale);
}

/// Inequality operator
template<typename T>
[[nodiscard]] constexpr bool operator!=(const posrotscale<T>& lhs, const posrotscale<T>& rhs) noexcept
{
    return !(lhs == rhs);
}

// ----------------------------------------------------------------------------
// Transform operations
// ----------------------------------------------------------------------------

/// Multiply two transforms (concatenation)
template<typename T>
[[nodiscard]] posrotscale<T> operator*(const posrotscale<T>& lhs, const posrotscale<T>& rhs) noexcept
{
    // Transform order: first apply rhs, then lhs
    // Scale is applied first (in local space)
    // Then rotation
    // Then translation

    // Combined scale: lhs.scale * rhs.scale (component-wise)
    vector<T, 3> combined_scale = lhs.scale * rhs.scale;

    // Combined rotation: lhs.rot * rhs.rot
    quat<T> combined_rot = mul(lhs.rot, rhs.rot);

    // Combined position: lhs.pos + lhs.rot * (lhs.scale * rhs.pos)
    vector<T, 3> scaled_rhs_pos = lhs.scale * rhs.pos;
    vector<T, 3> combined_pos = lhs.pos + transform_vector(lhs.rot, scaled_rhs_pos);

    return posrotscale<T>(combined_pos, combined_rot, combined_scale);
}

/// Multiply posrotscale with posrot
template<typename T>
[[nodiscard]] posrotscale<T> operator*(const posrotscale<T>& lhs, const posrot<T>& rhs) noexcept
{
    return lhs * posrotscale<T>(rhs);
}

/// Multiply posrot with posrotscale
template<typename T>
[[nodiscard]] posrotscale<T> operator*(const posrot<T>& lhs, const posrotscale<T>& rhs) noexcept
{
    return posrotscale<T>(lhs) * rhs;
}

/// Transform a point by the posrotscale
template<typename T>
[[nodiscard]] vector<T, 3> operator*(const posrotscale<T>& transform, const vector<T, 3>& point) noexcept
{
    // Apply scale, then rotation, then translation
    vector<T, 3> scaled = transform.scale * point;
    vector<T, 3> rotated = transform_vector(transform.rot, scaled);
    return transform.pos + rotated;
}

/// Inverse of a posrotscale transform
template<typename T>
[[nodiscard]] posrotscale<T> inverse(const posrotscale<T>& transform) noexcept
{
    // Inverse scale (component-wise reciprocal)
    vector<T, 3> inv_scale = vector<T, 3>(T(1)) / transform.scale;

    // Inverse rotation
    quat<T> inv_rot = conjugate(transform.rot);

    // Inverse translation: -inv_rot * (inv_scale * pos)
    vector<T, 3> scaled_pos = inv_scale * transform.pos;
    vector<T, 3> inv_pos = -transform_vector(inv_rot, scaled_pos);

    return posrotscale<T>(inv_pos, inv_rot, inv_scale);
}

/// Convert posrotscale to 3x4 matrix
template<typename T>
[[nodiscard]] matrix<T, 3, 4> to_matrix3x4(const posrotscale<T>& transform) noexcept
{
    matrix<T, 3, 3> rot_matrix = matrix_from_quat(transform.rot);
    matrix<T, 3, 4> result;

    // Apply scale to rotation matrix columns
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = rot_matrix[i][j] * transform.scale[j];
        }
    }

    // Set translation part
    result[0][3] = transform.pos.x;
    result[1][3] = transform.pos.y;
    result[2][3] = transform.pos.z;

    return result;
}

/// Convert posrotscale to 4x4 matrix
template<typename T>
[[nodiscard]] matrix<T, 4, 4> to_matrix4x4(const posrotscale<T>& transform) noexcept
{
    matrix<T, 3, 3> rot_matrix = matrix_from_quat(transform.rot);
    matrix<T, 4, 4> result = matrix<T, 4, 4>::identity();

    // Apply scale to rotation matrix columns
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = rot_matrix[i][j] * transform.scale[j];
        }
    }

    // Set translation part
    result[0][3] = transform.pos.x;
    result[1][3] = transform.pos.y;
    result[2][3] = transform.pos.z;

    return result;
}

/// Create posrotscale from 3x4 matrix (assumes no skew/shear)
template<typename T>
[[nodiscard]] posrotscale<T> posrotscale_from_matrix3x4(const matrix<T, 3, 4>& m) noexcept
{
    // Extract translation
    vector<T, 3> pos{m[0][3], m[1][3], m[2][3]};

    // Extract scale from column lengths
    vector<T, 3> col0{m[0][0], m[1][0], m[2][0]};
    vector<T, 3> col1{m[0][1], m[1][1], m[2][1]};
    vector<T, 3> col2{m[0][2], m[1][2], m[2][2]};

    T scale_x = length(col0);
    T scale_y = length(col1);
    T scale_z = length(col2);

    vector<T, 3> scale{scale_x, scale_y, scale_z};

    // Extract rotation by normalizing the scaled matrix
    matrix<T, 3, 3> rot_part;
    rot_part[0][0] = m[0][0] / scale_x;
    rot_part[0][1] = m[0][1] / scale_y;
    rot_part[0][2] = m[0][2] / scale_z;
    rot_part[1][0] = m[1][0] / scale_x;
    rot_part[1][1] = m[1][1] / scale_y;
    rot_part[1][2] = m[1][2] / scale_z;
    rot_part[2][0] = m[2][0] / scale_x;
    rot_part[2][1] = m[2][1] / scale_y;
    rot_part[2][2] = m[2][2] / scale_z;

    return posrotscale<T>(pos, quat_from_matrix(rot_part), scale);
}

/// Create posrotscale from 4x4 matrix (assumes no skew/shear)
template<typename T>
[[nodiscard]] posrotscale<T> posrotscale_from_matrix4x4(const matrix<T, 4, 4>& m) noexcept
{
    // Extract translation
    vector<T, 3> pos{m[0][3], m[1][3], m[2][3]};

    // Extract scale from column lengths
    vector<T, 3> col0{m[0][0], m[1][0], m[2][0]};
    vector<T, 3> col1{m[0][1], m[1][1], m[2][1]};
    vector<T, 3> col2{m[0][2], m[1][2], m[2][2]};

    T scale_x = length(col0);
    T scale_y = length(col1);
    T scale_z = length(col2);

    vector<T, 3> scale{scale_x, scale_y, scale_z};

    // Extract rotation by normalizing the scaled matrix
    matrix<T, 3, 3> rot_part;
    rot_part[0][0] = m[0][0] / scale_x;
    rot_part[0][1] = m[0][1] / scale_y;
    rot_part[0][2] = m[0][2] / scale_z;
    rot_part[1][0] = m[1][0] / scale_x;
    rot_part[1][1] = m[1][1] / scale_y;
    rot_part[1][2] = m[1][2] / scale_z;
    rot_part[2][0] = m[2][0] / scale_x;
    rot_part[2][1] = m[2][1] / scale_y;
    rot_part[2][2] = m[2][2] / scale_z;

    return posrotscale<T>(pos, quat_from_matrix(rot_part), scale);
}

/// Create posrotscale from 4x4 matrix (alias)
template<typename T>
[[nodiscard]] posrotscale<T> posrotscale_from_matrix4x4_alias(const matrix<T, 4, 4>& m) noexcept
{
    return posrotscale_from_matrix4x4(m);
}

/// Linear interpolation between two posrotscale transforms
template<typename T>
[[nodiscard]] posrotscale<T> lerp(const posrotscale<T>& a, const posrotscale<T>& b, T t) noexcept
{
    return posrotscale<T>(lerp(a.pos, b.pos, t), slerp(a.rot, b.rot, t), lerp(a.scale, b.scale, t));
}

/// Spherical linear interpolation between two posrotscale transforms
template<typename T>
[[nodiscard]] posrotscale<T> slerp(const posrotscale<T>& a, const posrotscale<T>& b, T t) noexcept
{
    return posrotscale<T>(lerp(a.pos, b.pos, t), slerp(a.rot, b.rot, t), lerp(a.scale, b.scale, t));
}

/// Normalize the rotation component
template<typename T>
[[nodiscard]] posrotscale<T> normalize(const posrotscale<T>& transform) noexcept
{
    return posrotscale<T>(transform.pos, normalize(transform.rot), transform.scale);
}

} // namespace sgl::math
