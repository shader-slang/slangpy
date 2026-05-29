// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/error.h"

#include <string_view>
#include <utility>

namespace sgl {

namespace detail {

    /// Return the compiler-specific spelling that contains T's name.
    template<typename T>
    constexpr std::string_view wrapped_type_name()
    {
#if SGL_MSVC
        return __FUNCSIG__;
#else
        return __PRETTY_FUNCTION__;
#endif
    }

    /// Remove MSVC's "class " / "struct " prefix from a type name fragment.
    constexpr std::string_view strip_class_key(std::string_view name)
    {
        if (name.starts_with("class "))
            return name.substr(6);
        if (name.starts_with("struct "))
            return name.substr(7);
        return name;
    }

    /// Best-effort default signature name used when T provides no explicit signature.
    template<typename T>
    constexpr std::string_view type_name()
    {
        constexpr std::string_view wrapped = wrapped_type_name<T>();
#if SGL_MSVC
        constexpr std::string_view marker = "wrapped_type_name<";
        size_t begin = wrapped.find(marker);
        if (begin == std::string_view::npos)
            return wrapped;
        begin += marker.size();
        size_t end = wrapped.rfind(">(void)");
        if (end == std::string_view::npos)
            end = wrapped.rfind('>');
#else
        constexpr std::string_view marker = "T = ";
        size_t begin = wrapped.find(marker);
        if (begin == std::string_view::npos)
            return wrapped;
        begin += marker.size();
        size_t end = wrapped.find(';', begin);
        if (end == std::string_view::npos)
            end = wrapped.find(']', begin);
#endif
        if (end == std::string_view::npos || end <= begin)
            return wrapped;
        return strip_class_key(wrapped.substr(begin, end - begin));
    }

} // namespace detail

template<typename R, typename T>
R narrow_cast(T value)
{
    if (!std::in_range<R>(value))
        SGL_THROW("narrow_cast failed");
    return static_cast<R>(value);
}

} // namespace sgl
