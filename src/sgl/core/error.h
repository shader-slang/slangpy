// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/logger.h"
#include "sgl/core/format.h"

#include <stdexcept>
#include <string>
#include <string_view>

/**
 * \file error.h
 * \brief Error handling utilities.
 *
 * Exceptions:
 * - SGL_THROW(msg, ...)
 * - SGL_CHECK(cond, msg, ...)
 * - SGL_UNIMPLEMENTED()
 * - SGL_UNREACHABLE()
 *
 * Assertions:
 * - SGL_ASSERT(cond)
 * - SGL_ASSERT_OP(a, b, op)
 *
 */

/// Source location information.
/// This is similar to std::source_location which unfortunately is still not
/// well supported on all major C++ library implementations.
struct SourceLocation {
    const char* file_name{nullptr};
    const char* function_name{nullptr};
    uint32_t line{0};
    uint32_t column{0};

    [[nodiscard]] static consteval SourceLocation current(
        const char* file_name = __builtin_FILE(),
        const char* function_name = __builtin_FUNCTION(),
        uint32_t line = __builtin_LINE(),
#if !SGL_GCC
        uint32_t column = __builtin_COLUMN()
#else
        uint32_t column = 0
#endif
    )
    {
        return SourceLocation{file_name, function_name, line, column};
    }
};

// -------------------------------------------------------------------------------------------------
// Exceptions
// -------------------------------------------------------------------------------------------------

namespace sgl {

enum class ExceptionDiagnosticFlags {
    none = 0,
    /// Break into debugger when throwing an exception.
    break_debugger = 1,
    /// Log exception message.
    log = 2,
};

SGL_ENUM_CLASS_OPERATORS(ExceptionDiagnosticFlags);

/// Set exception diagnostic options.
void SGL_API set_exception_diagnostics(ExceptionDiagnosticFlags flags);


/// Throw an exception.
[[noreturn]] SGL_API void throw_exception(const SourceLocation& loc, std::string_view msg);

namespace detail {
    [[noreturn]] inline void throw_exception(const SourceLocation& loc, std::string_view msg)
    {
        ::sgl::throw_exception(loc, msg);
    }

    template<typename... Args>
    [[noreturn]] inline void throw_exception(const SourceLocation& loc, fmt::format_string<Args...> fmt, Args&&... args)
    {
        ::sgl::throw_exception(loc, fmt::format(fmt, std::forward<Args>(args)...));
    }

} // namespace detail
} // namespace sgl

/// Helper for throwing exceptions.
/// Logs the exception and a stack trace before throwing.
#define SGL_THROW(...) ::sgl::detail::throw_exception(SourceLocation::current(), __VA_ARGS__)

/// Helper for checking conditions and throwing exceptions.
/// Logs the exception and a stack trace before throwing.
#define SGL_CHECK(cond, ...)                                                                                           \
    do {                                                                                                               \
        if (!(cond))                                                                                                   \
            SGL_THROW(__VA_ARGS__);                                                                                    \
    } while (0)

/// Helper for throwing an exception if a pointer is null.
#define SGL_CHECK_NOT_NULL(arg) SGL_CHECK(arg != nullptr, "\"{}\" must not be null", #arg)

#define SGL_CHECK_EQ(arg, value) SGL_CHECK(arg == value, "\"{}\" must be equal to {}", #arg, value)
#define SGL_CHECK_NE(arg, value) SGL_CHECK(arg != value, "\"{}\" must not be equal to {}", #arg, value)
#define SGL_CHECK_LT(arg, value) SGL_CHECK(arg < value, "\"{}\" must be less than {}", #arg, value)
#define SGL_CHECK_LE(arg, value) SGL_CHECK(arg <= value, "\"{}\" must be less than or equal {}", #arg, value)
#define SGL_CHECK_GT(arg, value) SGL_CHECK(arg > value, "\"{}\" must be greater than {}", #arg, value)
#define SGL_CHECK_GE(arg, value) SGL_CHECK(arg >= value, "\"{}\" must be greater than or equal {}", #arg, value)
#define SGL_CHECK_BOUNDS(arg, min, max)                                                                                \
    SGL_CHECK(arg >= min && arg < max, "\"{}\" must be in range [{}, {}]", #arg, min, max)

/// Helper for marking unimplemented functions.
#define SGL_UNIMPLEMENTED() SGL_THROW("Unimplemented")

/// Helper for marking unreachable code.
#define SGL_UNREACHABLE() SGL_THROW("Unreachable")

// -------------------------------------------------------------------------------------------------
// Assertions
// -------------------------------------------------------------------------------------------------

namespace sgl {

/// Report a failed assertion.
[[noreturn]] SGL_API void report_assertion(const SourceLocation& loc, std::string_view cond);

} // namespace sgl

#if SGL_ENABLE_ASSERTS

#define SGL_ASSERT(cond)                                                                                               \
    if (!(cond)) {                                                                                                     \
        ::sgl::report_assertion(SourceLocation::current(), #cond);                                                     \
    }

#define SGL_ASSERT_OP(a, b, op)                                                                                        \
    if (!((a)op(b))) {                                                                                                 \
        ::sgl::report_assertion(                                                                                       \
            SourceLocation::current(),                                                                                 \
            fmt::format("{} {} {} ({} {} {})", #a, #op, #b, a, #op, b)                                                 \
        );                                                                                                             \
    }

#else // SGL_ENABLE_ASSERTS

#define SGL_ASSERT(a)                                                                                                  \
    {                                                                                                                  \
    }
#define SGL_ASSERT_OP(a, b, op)                                                                                        \
    {                                                                                                                  \
    }

#endif // SGL_ENABLE_ASSERTS

#define SGL_ASSERT_EQ(a, b) SGL_ASSERT_OP(a, b, ==)
#define SGL_ASSERT_NE(a, b) SGL_ASSERT_OP(a, b, !=)
#define SGL_ASSERT_GE(a, b) SGL_ASSERT_OP(a, b, >=)
#define SGL_ASSERT_GT(a, b) SGL_ASSERT_OP(a, b, >)
#define SGL_ASSERT_LE(a, b) SGL_ASSERT_OP(a, b, <=)
#define SGL_ASSERT_LT(a, b) SGL_ASSERT_OP(a, b, <)
