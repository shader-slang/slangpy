// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <algorithm>
#include <concepts>

namespace sgl {

/// Clamp x to [lo..hi] range.
template<typename T>
constexpr T clamp(const T& x, const T& lo, const T& hi)
{
    return std::min(std::max(x, lo), hi);
}

/// Linearly interpolate between a and b.
template<typename T, std::floating_point U>
constexpr T lerp(const T& a, const T& b, const U& t)
{
    return (U(1) - t) * a + t * b;
}

/// Returns true if integer value is a power of two.
template<std::integral T>
constexpr bool is_power_of_two(T a)
{
    return (a & (a - (T)1)) == 0;
}

/// Returns always positive modulo value.
template<std::integral T>
constexpr T modulo(T a, T b)
{
    T result = a - (a / b) * b;
    return result < 0 ? result + b : result;
}

/// Divide a by b and round up to the next integer.
template<std::integral T>
constexpr T div_round_up(T a, T b)
{
    return (a + b - T(1)) / b;
}

/// Align an integer value to the given alignment.
template<std::integral T>
constexpr T align_to(T alignment, T value)
{
    return ((value + alignment - T(1)) / alignment) * alignment;
}

} // namespace sgl
