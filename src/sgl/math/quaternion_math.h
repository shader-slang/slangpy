// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Most of this code is derived from the GLM library at https://github.com/g-truc/glm
// License: https://github.com/g-truc/glm/blob/master/copying.txt

#pragma once

#include "sgl/math/quaternion_types.h"
#include "sgl/math/matrix_types.h"
#include "sgl/math/scalar_math.h"
#include "sgl/math/vector_math.h"
#include "sgl/math/constants.h"
#include "sgl/core/error.h"

namespace sgl::math {

// ----------------------------------------------------------------------------
// Unary operators (component-wise)
// ----------------------------------------------------------------------------

/// Unary plus operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator+(const quat<T> q) noexcept
{
    return q;
}

/// Unary minus operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator-(const quat<T> q) noexcept
{
    return quat<T>{-q.x, -q.y, -q.z, -q.w};
}

// ----------------------------------------------------------------------------
// Binary operators (component-wise)
// ----------------------------------------------------------------------------

/// Binary + operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator+(const quat<T>& lhs, const quat<T>& rhs) noexcept
{
    return quat<T>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
}

/// Binary + operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator+(const quat<T>& lhs, T rhs) noexcept
{
    return quat<T>{lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
}

/// Binary + operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator+(T lhs, const quat<T>& rhs) noexcept
{
    return quat<T>{lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
}

/// Binary - operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator-(const quat<T>& lhs, const quat<T>& rhs) noexcept
{
    return quat<T>{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
}

/// Binary - operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator-(const quat<T>& lhs, T rhs) noexcept
{
    return quat<T>{lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
}

/// Binary - operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator-(T lhs, const quat<T>& rhs) noexcept
{
    return quat<T>{lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
}

/// Binary * operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator*(const quat<T>& lhs, T rhs) noexcept
{
    return quat<T>{lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
}

/// Binary * operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator*(T lhs, const quat<T>& rhs) noexcept
{
    return quat<T>{lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
}

/// Binary / operator
template<typename T>
[[nodiscard]] constexpr quat<T> operator/(const quat<T>& lhs, T rhs) noexcept
{
    return quat<T>{lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
}

// ----------------------------------------------------------------------------
// Comparison (component-wise)
// ----------------------------------------------------------------------------

/// Component-wise == comparison.
template<typename T>
[[nodiscard]] constexpr vector<bool, 4> eq(const quat<T>& lhs, const quat<T>& rhs)
{
    return bool4{lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
}

/// Component-wise != comparison.
template<typename T>
[[nodiscard]] constexpr vector<bool, 4> ne(const quat<T>& lhs, const quat<T>& rhs)
{
    return bool4{lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
}

// ----------------------------------------------------------------------------
// Multiplication
// ----------------------------------------------------------------------------

/// Multiply quaternion with another quaternion.
template<typename T>
[[nodiscard]] constexpr quat<T> mul(const quat<T>& lhs, const quat<T>& rhs) noexcept
{
    return quat<T>{
        lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y, // x
        lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z, // y
        lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x, // z
        lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z  // w
    };
}

/// Multiply quaternion and 3 component vector.
template<typename T>
[[nodiscard]] constexpr vector<T, 3> mul(const quat<T>& q, const vector<T, 3>& v) noexcept
{
    vector<T, 3> qv(q.x, q.y, q.z);
    vector<T, 3> uv(cross(qv, v));
    vector<T, 3> uuv(cross(qv, uv));
    return v + ((uv * q.w) + uuv) * T(2);
}

/// Transform a vector by a quaternion.
template<typename T>
[[nodiscard]] constexpr vector<T, 3> transform_vector(const quat<T>& q, const vector<T, 3>& v) noexcept
{
    return mul(q, v);
}

// ----------------------------------------------------------------------------
// Floating point checks
// ----------------------------------------------------------------------------

/// isfinite
template<typename T>
[[nodiscard]] constexpr vector<bool, 4> isfinite(const quat<T>& q)
{
    return vector<bool, 4>{isfinite(q.x), isfinite(q.y), isfinite(q.z), isfinite(q.w)};
}

/// isinf
template<typename T>
[[nodiscard]] constexpr vector<bool, 4> isinf(const quat<T>& q)
{
    return vector<bool, 4>{isinf(q.x), isinf(q.y), isinf(q.z), isinf(q.w)};
}

/// isnan
template<typename T>
[[nodiscard]] constexpr vector<bool, 4> isnan(const quat<T>& q)
{
    return vector<bool, 4>{isnan(q.x), isnan(q.y), isnan(q.z), isnan(q.w)};
}

// ----------------------------------------------------------------------------
// Geometric functions
// ----------------------------------------------------------------------------

/// dot
template<typename T>
[[nodiscard]] constexpr T dot(const quat<T>& lhs, const quat<T>& rhs)
{
    vector<T, 4> tmp{lhs.w * rhs.w, lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    return (tmp.x + tmp.y) + (tmp.z + tmp.w);
}

/// cross
template<typename T>
[[nodiscard]] constexpr quat<T> cross(const quat<T>& lhs, const quat<T>& rhs)
{
    return quat<T>(
        lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y, // x
        lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z, // y
        lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x, // z
        lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z  // w
    );
}

/// length
template<typename T>
[[nodiscard]] T length(const quat<T>& q)
{
    return sqrt(dot(q, q));
}

/// normalize
template<typename T>
[[nodiscard]] constexpr quat<T> normalize(const quat<T>& q)
{
    T len = length(q);
    if (len <= T(0))
        return quat<T>(T(0), T(0), T(0), T(1));
    return q * (T(1) / len);
}

/// conjugate
template<typename T>
[[nodiscard]] constexpr quat<T> conjugate(const quat<T>& q)
{
    return quat<T>(-q.x, -q.y, -q.z, q.w);
}

/// inverse
template<typename T>
quat<T> inverse(const quat<T>& q)
{
    return conjugate(q) / dot(q, q);
}

/// Linear interpolation.
template<typename T>
[[nodiscard]] constexpr quat<T> lerp(const quat<T>& x, const quat<T>& y, T t)
{
    return x * (T(1) - t) + y * t;
}

/// Spherical linear interpolation.
template<typename T>
[[nodiscard]] constexpr quat<T> slerp(const quat<T>& x, const quat<T>& y_, T t)
{
    quat<T> y = y_;

    T cos_theta = dot(x, y);

    // If cos_theta < 0, the interpolation will take the long way around the sphere.
    // To fix this, one quat must be negated.
    if (cos_theta < T(0)) {
        y = -y;
        cos_theta = -cos_theta;
    }

    // Perform a linear interpolation when cos_theta is close to 1 to avoid side effect of sin(angle) becoming a zero
    // denominator
    if (cos_theta > T(1) - std::numeric_limits<T>::epsilon()) {
        // Linear interpolation
        return lerp(x, y, t);
    } else {
        // Essential Mathematics, page 467
        T angle = acos(cos_theta);
        return (sin((T(1) - t) * angle) * x + sin(t * angle) * y) / sin(angle);
    }
}

// ----------------------------------------------------------------------------
// Misc
// ----------------------------------------------------------------------------

/// Returns pitch value of euler angles expressed in radians.
template<typename T>
[[nodiscard]] constexpr T pitch(const quat<T>& q)
{
    T y = T(2) * (q.y * q.z + q.w * q.x);
    T x = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;

    // Handle sigularity, avoid atan2(0,0)
    if (abs(x) < std::numeric_limits<T>::epsilon() && abs(y) < std::numeric_limits<T>::epsilon())
        return T(T(2) * atan2(q.x, q.w));

    return atan2(y, x);
}

/// Returns yaw value of euler angles expressed in radians.
template<typename T>
[[nodiscard]] constexpr T yaw(const quat<T>& q)
{
    return asin(clamp(T(-2) * (q.x * q.z - q.w * q.y), T(-1), T(1)));
}

/// Returns roll value of euler angles expressed in radians.
template<typename T>
[[nodiscard]] constexpr T roll(const quat<T>& q)
{
    return atan2(T(2) * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z);
}

/// Extract the euler angles in radians from a quaternion (pitch as x, yaw as y, roll as z).
template<typename T>
[[nodiscard]] constexpr vector<T, 3> euler_angles(const quat<T>& q)
{
    return vector<T, 3>(pitch(q), yaw(q), roll(q));
}

// ----------------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------------

/**
 * Build a quaternion from an angle and a normalized axis.
 * \param angle Angle expressed in radians.
 * \param axis Axis of the quaternion (must be normalized).
 */
template<typename T>
[[nodiscard]] inline quat<T> quat_from_angle_axis(T angle, const vector<T, 3>& axis)
{
    T s = sin(angle * T(0.5));
    T c = cos(angle * T(0.5));
    return quat<T>(axis * s, c);
}

/**
 * Compute the rotation between two vectors.
 * \param from From vector (must to be normalized).
 * \param to To vector (must to be normalized).
 */
template<typename T>
[[nodiscard]] inline quat<T> quat_from_rotation_between_vectors(const vector<T, 3>& from, const vector<T, 3>& to)
{
    T cos_theta = dot(from, to);
    vector<T, 3> axis;

    if (cos_theta >= T(1) - std::numeric_limits<T>::epsilon()) {
        // from and to point in the same direction
        return quat<T>::identity();
    }

    if (cos_theta < T(-1) + std::numeric_limits<T>::epsilon()) {
        // special case when vectors in opposite directions :
        // there is no "ideal" rotation axis
        // So guess one; any will do as long as it's perpendicular to start
        // This implementation favors a rotation around the Up axis (Y),
        // since it's often what you want to do.
        axis = cross(vector<T, 3>(0, 0, 1), from);
        if (dot(axis, axis) < std::numeric_limits<T>::epsilon()) // bad luck, they were parallel, try again!
            axis = cross(vector<T, 3>(1, 0, 0), from);

        axis = normalize(axis);
        return quat_from_angle_axis(T(M_PI), axis);
    }

    // Implementation from Stan Melax's Game Programming Gems 1 article
    axis = cross(from, to);

    T s = sqrt((T(1) + cos_theta) * T(2));
    T invs = T(1) / s;

    return quat<T>(axis.x * invs, axis.y * invs, axis.z * invs, s * T(0.5f));
}

/**
 * Build a quaternion from euler angles (pitch, yaw, roll), in radians.
 */
template<typename T>
[[nodiscard]] inline quat<T> quat_from_euler_angles(const vector<T, 3>& angles)
{
    vector<T, 3> c = cos(angles * T(0.5));
    vector<T, 3> s = sin(angles * T(0.5));

    return quat<T>(
        s.x * c.y * c.z - c.x * s.y * s.z, // x
        c.x * s.y * c.z + s.x * c.y * s.z, // y
        c.x * c.y * s.z - s.x * s.y * c.z, // z
        c.x * c.y * c.z + s.x * s.y * s.z  // w
    );
}

/**
 * Construct a quaternion from a 3x3 rotation matrix.
 */
template<typename T>
[[nodiscard]] inline quat<T> quat_from_matrix(const matrix<T, 3, 3>& m)
{
    T four_x_squared_minus_1 = m[0][0] - m[1][1] - m[2][2];
    T four_y_squared_minus_1 = m[1][1] - m[0][0] - m[2][2];
    T four_z_squared_minus_1 = m[2][2] - m[0][0] - m[1][1];
    T four_w_squared_minus_1 = m[0][0] + m[1][1] + m[2][2];

    int biggest_index = 0;
    T four_biggest_squared_minus_1 = four_w_squared_minus_1;
    if (four_x_squared_minus_1 > four_biggest_squared_minus_1) {
        four_biggest_squared_minus_1 = four_x_squared_minus_1;
        biggest_index = 1;
    }
    if (four_y_squared_minus_1 > four_biggest_squared_minus_1) {
        four_biggest_squared_minus_1 = four_y_squared_minus_1;
        biggest_index = 2;
    }
    if (four_z_squared_minus_1 > four_biggest_squared_minus_1) {
        four_biggest_squared_minus_1 = four_z_squared_minus_1;
        biggest_index = 3;
    }

    T biggest_val = sqrt(four_biggest_squared_minus_1 + T(1)) * T(0.5);
    T mult = T(0.25) / biggest_val;

    switch (biggest_index) {
    case 0:
        return quat<T>((m[2][1] - m[1][2]) * mult, (m[0][2] - m[2][0]) * mult, (m[1][0] - m[0][1]) * mult, biggest_val);
    case 1:
        return quat<T>(biggest_val, (m[1][0] + m[0][1]) * mult, (m[0][2] + m[2][0]) * mult, (m[2][1] - m[1][2]) * mult);
    case 2:
        return quat<T>((m[1][0] + m[0][1]) * mult, biggest_val, (m[2][1] + m[1][2]) * mult, (m[0][2] - m[2][0]) * mult);
    case 3:
        return quat<T>((m[0][2] + m[2][0]) * mult, (m[2][1] + m[1][2]) * mult, biggest_val, (m[1][0] - m[0][1]) * mult);
    default:
        SGL_UNREACHABLE();
    }
}

/**
 * Build a look-at quaternion.
 * If right handed, forward direction is mapped onto -Z axis.
 * If left handed, forward direction is mapped onto +Z axis.
 * \param dir Forward direction (must to be normalized).
 * \param up Up vector (must be normalized).
 * \param handedness Coordinate system handedness.
 */
template<typename T>
[[nodiscard]] inline quat<T>
quat_from_look_at(const vector<T, 3>& dir, const vector<T, 3>& up, Handedness handedness = Handedness::right_handed)
{
    matrix<T, 3, 3> m;
    m.set_col(2, handedness == Handedness::right_handed ? -dir : dir);
    vector<T, 3> right = normalize(cross(up, m.get_col(2)));
    m.set_col(0, right);
    m.set_col(1, cross(m.get_col(2), m.get_col(0)));

    return quat_from_matrix(m);
}

template<typename T>
[[nodiscard]] std::string to_string(const quat<T>& q)
{
    return ::fmt::format("{}", q);
}

} // namespace sgl::math

template<typename T>
struct std::hash<::sgl::math::quat<T>> {
    constexpr size_t operator()(const ::sgl::math::quat<T>& v) const
    {
        size_t result = 0;
        for (size_t i = 0; i < 4; ++i)
            result ^= std::hash<T>()(v[i]) + 0x9e3779b9 + (result << 6) + (result >> 2);
        return result;
    }
};

/// Quaternion string formatter.
template<typename T>
struct fmt::formatter<sgl::math::quat<T>> : formatter<T> {
    template<typename FormatContext>
    auto format(const sgl::math::quat<T>& v, FormatContext& ctx) const
    {
        auto out = ctx.out();
        out = fmt::format_to(out, "{{{}, {}, {}, {}}}", v.x, v.y, v.z, v.w);
        return out;
    }
};
