// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/math/scalar_types.h"
#include "sgl/math/vector_types.h"
#include "sgl/math/quaternion_types.h"

#include <array>
#include <type_traits>

namespace sgl::math {

/**
 * Position and rotation transform type.
 *
 * Represents a rigid body transform with translation and rotation components.
 * This is useful for transforms that don't need scaling.
 */
template<typename T>
struct posrot {
    using value_type = T;
    static_assert(std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>>, "Invalid posrot type");

    vector<T, 3> pos;
    quat<T> rot;

    constexpr posrot() noexcept
        : pos{T(0)}
        , rot{}
    {
    }

    explicit constexpr posrot(const vector<T, 3>& position) noexcept
        : pos{position}
        , rot{}
    {
    }

    explicit constexpr posrot(const quat<T>& rotation) noexcept
        : pos{T(0)}
        , rot{rotation}
    {
    }

    explicit constexpr posrot(const vector<T, 3>& position, const quat<T>& rotation) noexcept
        : pos{position}
        , rot{rotation}
    {
    }

    template<typename U>
    explicit constexpr posrot(const std::array<U, 7>& a) noexcept
        : pos{T(a[0]), T(a[1]), T(a[2])}
        , rot{T(a[3]), T(a[4]), T(a[5]), T(a[6])}
    {
    }

    /// Identity transform.
    [[nodiscard]] static posrot identity() { return posrot{}; }
};

using posrotf = posrot<float>;

} // namespace sgl::math

namespace sgl {

using posrotf = math::posrotf;

} // namespace sgl
