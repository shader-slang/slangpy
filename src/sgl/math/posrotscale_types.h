// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/math/scalar_types.h"
#include "sgl/math/vector_types.h"
#include "sgl/math/quaternion_types.h"
#include "sgl/math/posrot_types.h"

#include <array>
#include <type_traits>

namespace sgl::math {

/**
 * Position, rotation, and scale transform type.
 *
 * Represents a full affine transform with translation, rotation, and non-uniform scaling components.
 * This is the most general transform representation for most graphics applications.
 */
template<typename T>
struct posrotscale {
    using value_type = T;
    static_assert(std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>>, "Invalid posrotscale type");

    vector<T, 3> pos;
    quat<T> rot;
    vector<T, 3> scale;

    constexpr posrotscale() noexcept
        : pos{T(0)}
        , rot{}
        , scale{T(1)}
    {
    }

    explicit constexpr posrotscale(const vector<T, 3>& position) noexcept
        : pos{position}
        , rot{}
        , scale{T(1)}
    {
    }

    explicit constexpr posrotscale(const quat<T>& rotation) noexcept
        : pos{T(0)}
        , rot{rotation}
        , scale{T(1)}
    {
    }


    explicit constexpr posrotscale(const vector<T, 3>& position, const quat<T>& rotation) noexcept
        : pos{position}
        , rot{rotation}
        , scale{T(1)}
    {
    }

    explicit constexpr posrotscale(const vector<T, 3>& position, const vector<T, 3>& scaling) noexcept
        : pos{position}
        , rot{}
        , scale{scaling}
    {
    }

    explicit constexpr posrotscale(const quat<T>& rotation, const vector<T, 3>& scaling) noexcept
        : pos{T(0)}
        , rot{rotation}
        , scale{scaling}
    {
    }

    explicit constexpr posrotscale(
        const vector<T, 3>& position,
        const quat<T>& rotation,
        const vector<T, 3>& scaling
    ) noexcept
        : pos{position}
        , rot{rotation}
        , scale{scaling}
    {
    }

    /// Constructor from posrot (with unit scale)
    explicit constexpr posrotscale(const posrot<T>& pr) noexcept
        : pos{pr.pos}
        , rot{pr.rot}
        , scale{T(1)}
    {
    }

    template<typename U>
    explicit constexpr posrotscale(const std::array<U, 10>& a) noexcept
        : pos{T(a[0]), T(a[1]), T(a[2])}
        , rot{T(a[3]), T(a[4]), T(a[5]), T(a[6])}
        , scale{T(a[7]), T(a[8]), T(a[9])}
    {
    }

    /// Identity transform.
    [[nodiscard]] static posrotscale identity() { return posrotscale{}; }
};

using posrotscalef = posrotscale<float>;

} // namespace sgl::math

namespace sgl {

using posrotscalef = math::posrotscalef;

} // namespace sgl
