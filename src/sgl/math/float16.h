// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <cstdint>
#include <functional>
#include <limits>

namespace sgl::math {

SGL_API uint16_t float32_to_float16(float value);
SGL_API float float16_to_float32(uint16_t value);

struct float16_t {
    float16_t() = default;

    float16_t(uint32_t sign, uint32_t exponent, uint32_t fraction)
        : m_bits((sign & 0x01) << 15 | (exponent & 0x1f) << 10 | (fraction & 0x03ff))
    {
    }

    explicit float16_t(float value)
        : m_bits(float32_to_float16(value))
    {
    }

    template<typename T>
    explicit float16_t(T value)
        : m_bits(float32_to_float16(static_cast<float>(value)))
    {
    }

    operator float() const { return float16_to_float32(m_bits); }

    static constexpr float16_t fromBits(uint16_t bits) { return float16_t(bits, FromBits); }
    uint16_t toBits() const { return m_bits; }

    bool operator==(const float16_t other) const { return m_bits == other.m_bits; }
    bool operator!=(const float16_t other) const { return m_bits != other.m_bits; }
    bool operator<(const float16_t other) const { return static_cast<float>(*this) < static_cast<float>(other); }
    bool operator<=(const float16_t other) const { return static_cast<float>(*this) <= static_cast<float>(other); }
    bool operator>(const float16_t other) const { return static_cast<float>(*this) > static_cast<float>(other); }
    bool operator>=(const float16_t other) const { return static_cast<float>(*this) >= static_cast<float>(other); }

    float16_t operator+() const { return *this; }
    float16_t operator-() const { return fromBits(m_bits ^ 0x8000); }

    // TODO: Implement math operators in native fp16 precision. For now using fp32.
    float16_t operator+(const float16_t other) const
    {
        return float16_t(static_cast<float>(*this) + static_cast<float>(other));
    }
    float16_t operator-(const float16_t other) const
    {
        return float16_t(static_cast<float>(*this) - static_cast<float>(other));
    }
    float16_t operator*(const float16_t other) const
    {
        return float16_t(static_cast<float>(*this) * static_cast<float>(other));
    }
    float16_t operator/(const float16_t other) const
    {
        return float16_t(static_cast<float>(*this) / static_cast<float>(other));
    }

    float16_t operator+=(const float16_t other) { return *this = *this + other; }
    float16_t operator-=(const float16_t other) { return *this = *this - other; }
    float16_t operator*=(const float16_t other) { return *this = *this * other; }
    float16_t operator/=(const float16_t other) { return *this = *this / other; }

    constexpr bool isfinite() const noexcept { return exponent() < 31; }
    constexpr bool isinf() const noexcept { return exponent() == 31 && mantissa() == 0; }
    constexpr bool isnan() const noexcept { return exponent() == 31 && mantissa() != 0; }

    constexpr bool is_normalized() const noexcept { return exponent() > 0 && exponent() < 31; }
    constexpr bool is_denormalized() const noexcept { return exponent() == 0 && mantissa() != 0; }

private:
    enum Tag { FromBits };

    constexpr float16_t(uint16_t bits, Tag)
        : m_bits(bits)
    {
    }

    constexpr uint16_t mantissa() const noexcept { return m_bits & 0x3ff; }
    constexpr uint16_t exponent() const noexcept { return (m_bits >> 10) & 0x001f; }

    uint16_t m_bits;
};

SGL_DIAGNOSTIC_PUSH
// disable warning about literal suffixes not starting with an underscore
SGL_DISABLE_MSVC_WARNING(4455)
SGL_DISABLE_CLANG_WARNING("-Wuser-defined-literals")
SGL_DISABLE_GCC_WARNING("-Wliteral-suffix")

/// h suffix for "half float" literals.
inline float16_t operator""h(long double value)
{
    return float16_t(static_cast<float>(value));
}

SGL_DIAGNOSTIC_POP

} // namespace sgl::math

namespace std {

template<>
class numeric_limits<sgl::math::float16_t> {
public:
    static constexpr bool is_specialized = true;
    static constexpr sgl::math::float16_t min() noexcept { return sgl::math::float16_t::fromBits(0x0200); }
    static constexpr sgl::math::float16_t max() noexcept { return sgl::math::float16_t::fromBits(0x7bff); }
    static constexpr sgl::math::float16_t lowest() noexcept { return sgl::math::float16_t::fromBits(0xfbff); }
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr sgl::math::float16_t epsilon() noexcept { return sgl::math::float16_t::fromBits(0x1200); }
    static constexpr sgl::math::float16_t round_error() noexcept { return sgl::math::float16_t::fromBits(0x3c00); }
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr sgl::math::float16_t infinity() noexcept { return sgl::math::float16_t::fromBits(0x7c00); }
    static constexpr sgl::math::float16_t quiet_NaN() noexcept { return sgl::math::float16_t::fromBits(0x7fff); }
    static constexpr sgl::math::float16_t signaling_NaN() noexcept { return sgl::math::float16_t::fromBits(0x7dff); }
    static constexpr sgl::math::float16_t denorm_min() noexcept { return sgl::math::float16_t::fromBits(0); }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};

// Add hash specializations
template<>
struct hash<sgl::math::float16_t> {
    size_t operator()(const sgl::math::float16_t& v) const noexcept
    {
        // Hash the bits
        return hash<uint16_t>()(v.toBits());
    }
};

} // namespace std
