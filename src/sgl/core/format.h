// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

SGL_DIAGNOSTIC_PUSH
SGL_DISABLE_MSVC_WARNING(4061 4459)
#include <fmt/format.h>
#include <fmt/ranges.h>
SGL_DIAGNOSTIC_POP

#include <filesystem>
#include <optional>

template<>
struct fmt::formatter<std::filesystem::path> : formatter<std::string> {
    template<typename FormatContext>
    auto format(const std::filesystem::path& p, FormatContext& ctx) const
    {
        return formatter<std::string>::format(p.string(), ctx);
    }
};

template<typename T>
struct fmt::formatter<std::optional<T>> : formatter<T> {
    template<typename FormatContext>
    auto format(const std::optional<T>& opt, FormatContext& ctx) const
    {
        if (opt) {
            formatter<T>::format(*opt, ctx);
            return ctx.out();
        }
        return fmt::format_to(ctx.out(), "nullopt");
    }
};
