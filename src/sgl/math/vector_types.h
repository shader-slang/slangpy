// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/math/scalar_types.h"
#include "sgl/core/error.h"

namespace sgl::math {

// ----------------------------------------------------------------------------
// Vector types
// ----------------------------------------------------------------------------

/**
 * Vector type.
 *
 * The semantics are aligned with Slang:
 * - Math operators are element-wise (e.g. +, -, *, /)
 * - Free standing functions for vector operations (e.g. dot(), cross(), etc.)
 *
 * Exception: Comparison operators (==, !=, <, etc.) return a single bool for
 * STL compatibility. Use equal(), not_equal(), etc. for component-wise comparisons.
 *
 * \tparam T Scalar type
 * \tparam N Number of elements (1-4)
 */
template<typename T, int N>
struct vector;

template<typename T>
struct vector<T, 1> {
    static constexpr int dimension = 1;
    using value_type = T;

    union {
        T x;
        T r;
        T s;
    };
    // clang-format off
    /// Default constructor.
    constexpr vector() noexcept = default;
    /// Copy constructor.
    constexpr vector(const vector<T, 1>& other) noexcept = default;
    /// Explicit basic constructor.
    explicit constexpr vector(T x) noexcept : x{x} {}
    /// Explicit basic constructor (scalar).
    template<typename U>
    explicit constexpr vector(U x) noexcept : x{T(x)} {}
    // clang-format on

    template<typename U>
    constexpr vector(const vector<U, 1>& other) noexcept
        : x{T(other.x)}
    {
    }

    template<typename U>
    explicit constexpr vector(const std::array<U, 1>& a) noexcept
        : x{T(a[0])}
    {
    }

    [[nodiscard]] constexpr T& operator[](int index) noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] constexpr const T& operator[](int index) const noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] static constexpr int length() noexcept { return dimension; }
};

template<typename T>
struct vector<T, 2> {
    static constexpr int dimension = 2;
    using value_type = T;

    union {
        struct {
            T x, y;
        };
        struct {
            T r, g;
        };
        struct {
            T s, t;
        };
    };
    // clang-format off
    /// Default constructor.
    constexpr vector() noexcept = default;
    /// Copy constructor.
    constexpr vector(const vector<T, 2>& other) noexcept = default;
    /// Explicit basic constructor.
    constexpr vector(T x, T y) noexcept : x{x}, y{y} {}
    /// Explicit basic constructor (scalar).
    explicit constexpr vector(T scalar) noexcept : x{scalar}, y{scalar} {}
    /// Explicit basic constructor (scalar).
    template<typename U>
    explicit constexpr vector(U scalar) noexcept : x{T(scalar)}, y{T(scalar)} {}

    template<typename X, typename Y>
    constexpr vector(X x, Y y) noexcept : x{T(x)}, y{T(y)} {}
    // clang-format on

    template<typename U>
    constexpr vector(const vector<U, 2>& other) noexcept
        : x{T(other.x)}
        , y{T(other.y)}
    {
    }

    template<typename U>
    explicit constexpr vector(const std::array<U, 2>& a) noexcept
        : x{T(a[0])}
        , y{T(a[1])}
    {
    }

    [[nodiscard]] constexpr T& operator[](int index) noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] constexpr const T& operator[](int index) const noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] static constexpr int length() noexcept { return dimension; }

#include "vector_swizzle_2.inl"
};

template<typename T>
struct vector<T, 3> {
    static constexpr int dimension = 3;
    using value_type = T;

    union {
        struct {
            T x, y, z;
        };
        struct {
            T r, g, b;
        };
        struct {
            T s, t, p;
        };
    };
    // clang-format off
    /// Default constructor.
    constexpr vector() noexcept = default;
    /// Copy constructor.
    constexpr vector(const vector<T, 3>& other) noexcept = default;
    /// Explicit basic constructor.
    constexpr vector(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    /// Explicit basic constructor (scalar).
    explicit constexpr vector(T scalar) noexcept : x{scalar}, y{scalar}, z{scalar} {}
    /// Explicit basic constructor (scalar).
    template<typename U>
    explicit constexpr vector(U scalar) noexcept : x{T(scalar)}, y{T(scalar)}, z{T(scalar)} {}

    template<typename X, typename Y, typename Z>
    constexpr vector(X x, Y y, Z z) noexcept : x{T(x)}, y{T(y)}, z{T(z)} {}
    template<typename XY, typename Z>
    constexpr vector(vector<XY, 2> xy, Z z) noexcept : x{T(xy.x)}, y{T(xy.y)}, z{T(z)} {}
    template<typename X, typename YZ>
    constexpr vector(X x, vector<YZ, 2> yz) noexcept : x{T(x)}, y{T(yz.x)}, z{T(yz.y)} {}
    // clang-format on

    template<typename U>
    constexpr vector(const vector<U, 3>& other) noexcept
        : x{T(other.x)}
        , y{T(other.y)}
        , z{T(other.z)}
    {
    }

    template<typename U>
    explicit constexpr vector(const std::array<U, 3>& a) noexcept
        : x{T(a[0])}
        , y{T(a[1])}
        , z{T(a[2])}
    {
    }

    [[nodiscard]] constexpr T& operator[](int index) noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] constexpr const T& operator[](int index) const noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] static constexpr int length() noexcept { return dimension; }

#include "vector_swizzle_3.inl"
};

template<typename T>
struct vector<T, 4> {
    static constexpr int dimension = 4;
    using value_type = T;

    union {
        struct {
            T x, y, z, w;
        };
        struct {
            T r, g, b, a;
        };
        struct {
            T s, t, p, q;
        };
    };
    // clang-format off
    /// Default constructor.
    constexpr vector() noexcept = default;
    /// Copy constructor.
    constexpr vector(const vector<T, 4>& other) noexcept = default;
    /// Explicit basic constructor.
    constexpr vector(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
    /// Explicit basic constructor (scalar).
    explicit constexpr vector(T scalar) noexcept : x{scalar}, y{scalar}, z{scalar}, w{scalar} {}
    /// Explicit basic constructor (scalar).
    template<typename U>
    explicit constexpr vector(U scalar) noexcept : x{T(scalar)}, y{T(scalar)}, z{T(scalar)}, w{T(scalar)} {}

    template<typename X, typename Y, typename Z, typename W>
    constexpr vector(X x, Y y, Z z, W w) noexcept : x{T(x)}, y{T(y)}, z{T(z)}, w{T(w)} {}
    template<typename XY, typename Z, typename W>
    constexpr vector(vector<XY, 2> xy, Z z, W w) noexcept : x{T(xy.x)}, y{T(xy.y)}, z{T(z)}, w{T(w)} {}
    template<typename X, typename YZ, typename W>
    constexpr vector(X x, vector<YZ, 2> yz, W w) noexcept : x{T(x)}, y{T(yz.x)}, z{T(yz.y)}, w{T(w)} {}
    template<typename X, typename Y, typename ZW>
    constexpr vector(X x, Y y, vector<ZW, 2> zw) noexcept : x{T(x)}, y{T(y)}, z{T(zw.x)}, w{T(zw.y)} {}
    template<typename XY, typename ZW>
    constexpr vector(vector<XY, 2> xy, vector<ZW, 2> zw) noexcept : x{T(xy.x)}, y{T(xy.y)}, z{T(zw.x)}, w{T(zw.y)} {}
    template<typename XYZ, typename W>
    constexpr vector(vector<XYZ, 3> xyz, W w) noexcept : x{T(xyz.x)}, y{T(xyz.y)}, z{T(xyz.z)}, w{T(w)} {}
    template<typename X, typename YZW>
    constexpr vector(X x, vector<YZW, 3> yzw) noexcept : x{T(x)}, y{T(yzw.x)}, z{T(yzw.y)}, w{T(yzw.z)} {}
    // clang-format on

    template<typename U>
    constexpr vector(const vector<U, 4>& other) noexcept
        : x{T(other.x)}
        , y{T(other.y)}
        , z{T(other.z)}
        , w{T(other.w)}
    {
    }

    template<typename U>
    explicit constexpr vector(const std::array<U, 4>& a) noexcept
        : x{T(a[0])}
        , y{T(a[1])}
        , z{T(a[2])}
        , w{T(a[3])}
    {
    }

    [[nodiscard]] constexpr T& operator[](int index) noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] constexpr const T& operator[](int index) const noexcept
    {
        SGL_ASSERT_LT(index, dimension);
        return (&(this->x))[index];
    }

    [[nodiscard]] static constexpr int length() noexcept { return dimension; }

#include "vector_swizzle_4.inl"
};

/// Equality operator.
template<typename T, int N>
[[nodiscard]] constexpr bool operator==(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    for (int i = 0; i < N; ++i)
        if (lhs[i] != rhs[i])
            return false;
    return true;
}

/// Inequality operator.
template<typename T, int N>
[[nodiscard]] constexpr bool operator!=(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    return !(lhs == rhs);
}

/// Lexicographic less-than operator.
template<arithmetic T, int N>
[[nodiscard]] constexpr bool operator<(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    for (int i = 0; i < N; ++i) {
        if (lhs[i] < rhs[i])
            return true;
        if (rhs[i] < lhs[i])
            return false;
    }
    return false;
}

/// Lexicographic greater-than operator.
template<arithmetic T, int N>
[[nodiscard]] constexpr bool operator>(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    return rhs < lhs;
}

/// Lexicographic less-or-equal operator.
template<arithmetic T, int N>
[[nodiscard]] constexpr bool operator<=(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    return !(rhs < lhs);
}

/// Lexicographic greater-or-equal operator.
template<arithmetic T, int N>
[[nodiscard]] constexpr bool operator>=(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    return !(lhs < rhs);
}

using bool1 = vector<bool, 1>;
using bool2 = vector<bool, 2>;
using bool3 = vector<bool, 3>;
using bool4 = vector<bool, 4>;
using int1 = vector<int, 1>;
using int2 = vector<int, 2>;
using int3 = vector<int, 3>;
using int4 = vector<int, 4>;
using uint1 = vector<uint, 1>;
using uint2 = vector<uint, 2>;
using uint3 = vector<uint, 3>;
using uint4 = vector<uint, 4>;
using float1 = vector<float, 1>;
using float2 = vector<float, 2>;
using float3 = vector<float, 3>;
using float4 = vector<float, 4>;
using float16_t1 = vector<float16_t, 1>;
using float16_t2 = vector<float16_t, 2>;
using float16_t3 = vector<float16_t, 3>;
using float16_t4 = vector<float16_t, 4>;

} // namespace sgl::math

namespace sgl {

using bool1 = math::bool1;
using bool2 = math::bool2;
using bool3 = math::bool3;
using bool4 = math::bool4;
using int1 = math::int1;
using int2 = math::int2;
using int3 = math::int3;
using int4 = math::int4;
using uint1 = math::uint1;
using uint2 = math::uint2;
using uint3 = math::uint3;
using uint4 = math::uint4;
using float1 = math::float1;
using float2 = math::float2;
using float3 = math::float3;
using float4 = math::float4;
using float16_t1 = math::float16_t1;
using float16_t2 = math::float16_t2;
using float16_t3 = math::float16_t3;
using float16_t4 = math::float16_t4;

} // namespace sgl
