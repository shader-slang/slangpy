// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "error.h"
#include "sgl/core/logger.h"
#include "sgl/core/platform.h"

#include <cstdio>
#include <cstdlib>

namespace sgl {

static ExceptionDiagnosticFlags s_exception_diagnostic_flags{
    ExceptionDiagnosticFlags::break_debugger | ExceptionDiagnosticFlags::log
};

void set_exception_diagnostics(ExceptionDiagnosticFlags flags)
{
    s_exception_diagnostic_flags = flags;
}

void throw_exception(const SourceLocation& loc, std::string_view msg)
{
    std::string error_msg = fmt::format("{}\n", msg);
    error_msg += fmt::format("{}:{} in function {}", loc.file_name, loc.line, loc.function_name);

    bool debugger_present = platform::is_debugger_present();
    bool python_active = platform::is_python_active();
    if (!debugger_present && !python_active)
        error_msg += "\nStack trace:\n" + platform::format_stacktrace(platform::backtrace(), 10);

    if (!python_active && is_set(s_exception_diagnostic_flags, ExceptionDiagnosticFlags::log))
        log_fatal(error_msg);

    if (is_set(s_exception_diagnostic_flags, ExceptionDiagnosticFlags::break_debugger) && debugger_present)
        platform::debug_break();

    throw std::runtime_error(error_msg.c_str());
}

void report_assertion(const SourceLocation& loc, std::string_view cond) noexcept
{
    bool debugger_present = false;
    try {
        debugger_present = platform::is_debugger_present();
        std::string error_msg = fmt::format("Assertion failed: {}\n", cond);
        error_msg += fmt::format("{}:{} in function {}\n", loc.file_name, loc.line, loc.function_name);
        if (!debugger_present)
            error_msg += "Stack trace:\n" + platform::format_stacktrace(platform::backtrace(), 10) + "\n";

        // Do not depend on logger initialization, outputs, or Python exception translation.
        std::fwrite(error_msg.data(), 1, error_msg.size(), stderr);
    } catch (...) {
        // Best effort, without C++ allocation or formatting, even if diagnostics failed.
        std::fputs("Assertion failed: ", stderr);
        if (!cond.empty())
            std::fwrite(cond.data(), 1, cond.size(), stderr);
        std::fprintf(
            stderr,
            "\n%s:%u in function %s\n",
            loc.file_name ? loc.file_name : "<unknown>",
            static_cast<unsigned int>(loc.line),
            loc.function_name ? loc.function_name : "<unknown>"
        );
    }
    std::fflush(stderr);

    if (debugger_present)
        platform::debug_break();

    std::abort();
}

} // namespace sgl
