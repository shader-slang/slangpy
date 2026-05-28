// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "scalar_types.h"
#include "vector_types.h"

namespace sgl::math {

template<floating_point T>
[[nodiscard]] inline T linear_to_srgb(T x)
{
    if (x <= T(0.0031308)) {
        return T(12.92) * x;
    } else {
        return T(1.055) * pow(x, T(1.0 / 2.4)) - T(0.055);
    }
}

template<floating_point T, int N>
[[nodiscard]] inline vector<T, N> linear_to_srgb(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, 1>(linear_to_srgb(x.x));
    else if constexpr (N == 2)
        return vector<T, 2>(linear_to_srgb(x.x), linear_to_srgb(x.y));
    else if constexpr (N == 3)
        return vector<T, 3>(linear_to_srgb(x.x), linear_to_srgb(x.y), linear_to_srgb(x.z));
    else if constexpr (N == 4)
        return vector<T, 4>(linear_to_srgb(x.x), linear_to_srgb(x.y), linear_to_srgb(x.z), linear_to_srgb(x.w));
}

template<floating_point T>
[[nodiscard]] inline T srgb_to_linear(T x)
{
    if (x <= T(0.04045)) {
        return T(1.0 / 12.92) * x;
    } else {
        return pow((x + T(0.055)) * T(1.0 / 1.055), T(2.4));
    }
}

template<floating_point T, int N>
[[nodiscard]] inline vector<T, N> srgb_to_linear(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, 1>(srgb_to_linear(x.x));
    else if constexpr (N == 2)
        return vector<T, 2>(srgb_to_linear(x.x), srgb_to_linear(x.y));
    else if constexpr (N == 3)
        return vector<T, 3>(srgb_to_linear(x.x), srgb_to_linear(x.y), srgb_to_linear(x.z));
    else if constexpr (N == 4)
        return vector<T, 4>(srgb_to_linear(x.x), srgb_to_linear(x.y), srgb_to_linear(x.z), srgb_to_linear(x.w));
}

} // namespace sgl::math
