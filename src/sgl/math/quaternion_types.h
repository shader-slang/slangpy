// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/math/scalar_types.h"
#include "sgl/math/vector_types.h"

#include <array>
#include <compare>
#include <type_traits>

namespace sgl::math {

/**
 * Quaternion type.
 *
 * A quaternion is an expression of the form:
 *
 * q = w + xi + yj + zk
 *
 * where w, x, y, and z are real numbers and i, j, and k are the imaginary units.
 *
 * The quaternion is normalized if:
 * w^2 + x^2 + y^2 + z^2 = 1
 *
 * Quaternions are stored as (x, y, z, w) to make them better for interop with the GPU.
 *
 * Comparison operators (==, !=, <, etc.) return a single bool for
 * STL compatibility. Use eq(), ne(), etc. for component-wise comparisons.
 *
 * \tparam T Scalar type
 */
template<typename T>
struct quat {
    using value_type = T;
    static_assert(std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>>, "Invalid quaternion type");

    T x, y, z, w;

    constexpr quat() noexcept
        : x{T(0)}
        , y{T(0)}
        , z{T(0)}
        , w{T(1)}
    {
    }

    explicit constexpr quat(const vector<T, 3>& xyz, const T& w) noexcept
        : x{xyz.x}
        , y{xyz.y}
        , z{xyz.z}
        , w{w}
    {
    }

    explicit constexpr quat(const T& x, const T& y, const T& z, const T& w) noexcept
        : x{x}
        , y{y}
        , z{z}
        , w{w}
    {
    }

    template<typename U>
    explicit constexpr quat(const std::array<U, 4>& a) noexcept
        : x{T(a[0])}
        , y{T(a[1])}
        , z{T(a[2])}
        , w{T(a[3])}
    {
    }

    /// Identity quaternion.
    [[nodiscard]] static quat identity() { return quat(T(0), T(0), T(0), T(1)); }

    // Accesses
    value_type& operator[](size_t i) { return (&x)[i]; }
    const value_type& operator[](size_t i) const { return (&x)[i]; }
};

/// Equality operator.
template<typename T>
[[nodiscard]] constexpr bool operator==(const quat<T>& lhs, const quat<T>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

/// Lexicographic three-way operator.
template<typename T>
[[nodiscard]] constexpr auto operator<=>(const quat<T>& lhs, const quat<T>& rhs)
{
    auto x_cmp = lhs.x <=> rhs.x;
    if (x_cmp != 0)
        return x_cmp;
    auto y_cmp = lhs.y <=> rhs.y;
    if (y_cmp != 0)
        return y_cmp;
    auto z_cmp = lhs.z <=> rhs.z;
    if (z_cmp != 0)
        return z_cmp;
    return lhs.w <=> rhs.w;
}

using quatf = quat<float>;

} // namespace sgl::math

namespace sgl {

using quatf = math::quatf;

} // namespace sgl
