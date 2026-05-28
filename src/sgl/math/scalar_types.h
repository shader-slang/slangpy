// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/math/float16.h"
#include "sgl/core/macros.h"
#include "sgl/core/format.h"

#include <concepts>
#include <string>
#include <cstdint>

namespace sgl::math {

enum class Handedness {
    right_handed,
    left_handed,
};

using uint = uint32_t;

// clang-format off
template<typename T> [[nodiscard]] std::string to_string(T v);
template<> [[nodiscard]] inline std::string to_string(bool v) { return v ? "1" : "0"; }
template<> [[nodiscard]] inline std::string to_string(int v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(uint v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(float v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(double v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(float16_t v) { return std::to_string(float(v)); }
// clang-format on

template<typename T>
concept boolean = std::same_as<T, bool>;

// Note: integral also includes bool
using std::integral;
using std::signed_integral;

template<typename T>
concept floating_point = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, float16_t>;

template<typename T>
concept arithmetic = integral<T> || floating_point<T>;

template<typename T>
concept signed_number = std::signed_integral<T> || floating_point<T>;


template<typename T>
struct ScalarTraits { };

template<>
struct ScalarTraits<bool> {
    static constexpr const char* name{"bool"};
};

template<>
struct ScalarTraits<int> {
    static constexpr const char* name{"int"};
};

template<>
struct ScalarTraits<uint> {
    static constexpr const char* name{"uint"};
};

template<>
struct ScalarTraits<float> {
    static constexpr const char* name{"float"};
};

template<>
struct ScalarTraits<double> {
    static constexpr const char* name{"double"};
};

template<>
struct ScalarTraits<float16_t> {
    static constexpr const char* name{"float16_t"};
};

} // namespace sgl::math

namespace sgl {

using uint = math::uint;
using float16_t = math::float16_t;

SGL_DIAGNOSTIC_PUSH
// disable warning about literal suffixes not starting with an underscore
SGL_DISABLE_MSVC_WARNING(4455)

using math::operator""h;

SGL_DIAGNOSTIC_POP

} // namespace sgl

// Formatter for the float16_t.
template<>
struct fmt::formatter<sgl::math::float16_t> : formatter<float> {
    template<typename FormatContext>
    auto format(sgl::math::float16_t value, FormatContext& ctx) const
    {
        return formatter<float>::format(float(value), ctx);
    }
};
