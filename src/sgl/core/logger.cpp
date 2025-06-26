// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "logger.h"
#include "sgl/core/macros.h"
#include "sgl/core/error.h"
#include "sgl/core/platform.h"
#include "sgl/core/format.h"

#include <fmt/color.h>

#include <array>
#include <iostream>

#if SGL_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace sgl {

static const std::array<std::string_view, 6> s_level_str{
    "",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "FATAL",
};

static const std::array<fmt::terminal_color, 6> s_level_color{
    fmt::terminal_color::white,
    fmt::terminal_color::magenta,
    fmt::terminal_color::cyan,
    fmt::terminal_color::yellow,
    fmt::terminal_color::red,
    fmt::terminal_color::bright_red,
};

ConsoleLoggerOutput::ConsoleLoggerOutput(bool colored)
{
    m_colored = colored && enable_ansi_control_sequences();
}

void ConsoleLoggerOutput::write(LogLevel level, const std::string_view module, const std::string_view msg)
{
    std::string_view level_str = s_level_str[static_cast<int>(level)];
    fmt::terminal_color color = s_level_color[static_cast<int>(level)];

    ::FILE* stream = level >= LogLevel::error ? stderr : stdout;

    if (level == LogLevel::none) {
        fmt::print(stream, "{}\n", msg);
        return;
    }

    if (m_colored) {
        if (module.empty())
            fmt::print(stream, "[{}] {}\n", fmt::styled(level_str, fmt::fg(color)), msg);
        else
            fmt::print(
                stream,
                "[{}] ({}) {}\n",
                fmt::styled(level_str, fmt::fg(color)),
                fmt::styled(module, fmt::fg(fmt::terminal_color::bright_black)),
                msg
            );
    } else {
        if (module.empty())
            fmt::print(stream, "[{}] {}\n", level_str, msg);
        else
            fmt::print(stream, "[{}] ({}) {}\n", level_str, module, msg);
    }

    ::fflush(stream);
}

std::string ConsoleLoggerOutput::to_string() const
{
    return fmt::format("ConsoleLoggerOutput(colored={})", m_colored);
}

bool ConsoleLoggerOutput::enable_ansi_control_sequences()
{
    if (auto value = platform::get_environment_variable("NO_COLOR"); value && !value->empty())
        return false;

#if SGL_WINDOWS
    // Set output mode to handle virtual terminal sequences.
    HANDLE h_output = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h_output == INVALID_HANDLE_VALUE)
        return false;
    DWORD mode = 0;
    if (!GetConsoleMode(h_output, &mode))
        return false;
    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (!SetConsoleMode(h_output, mode))
        return false;
#endif

    return true;
}

FileLoggerOutput::FileLoggerOutput(const std::filesystem::path& path)
    : m_path(path)
{
#if SGL_WINDOWS
    ::_wfopen_s(reinterpret_cast<FILE**>(&m_file), path.wstring().c_str(), L"w");
#else
    m_file = ::fopen(path.string().c_str(), "w");
#endif
}

FileLoggerOutput::~FileLoggerOutput()
{
    if (m_file)
        ::fclose(static_cast<FILE*>(m_file));
}

void FileLoggerOutput::write(LogLevel level, const std::string_view module, const std::string_view msg)
{
    if (level == LogLevel::none) {
        fmt::print(static_cast<FILE*>(m_file), "{}\n", msg);
        return;
    }

    if (module.empty())
        fmt::print(static_cast<FILE*>(m_file), "[{}] {}\n", s_level_str[int(level)], msg);
    else
        fmt::print(static_cast<FILE*>(m_file), "[{}] ({}) {}\n", s_level_str[int(level)], module, msg);
    ::fflush(static_cast<FILE*>(m_file));
}

std::string FileLoggerOutput::to_string() const
{
    return fmt::format("FileLoggerOutput(path=\"{}\")", m_path);
}

void DebugConsoleLoggerOutput::write(LogLevel level, const std::string_view module, const std::string_view msg)
{
#if SGL_WINDOWS
    if (level == LogLevel::none) {
        OutputDebugStringA(fmt::format("{}\n", msg).c_str());
        return;
    }

    if (module.empty())
        OutputDebugStringA(fmt::format("[{}] {}\n", s_level_str[int(level)], msg).c_str());
    else
        OutputDebugStringA(fmt::format("[{}] ({}) {}\n", s_level_str[int(level)], module, msg).c_str());
#else
    if (module.empty())
        fprintf(stdout, "[%s] %s\n", s_level_str[int(level)].data(), msg.data());
    else
        fprintf(stdout, "[%s] (%s) %s\n", s_level_str[int(level)].data(), module.data(), msg.data());

#endif
}

std::string DebugConsoleLoggerOutput::to_string() const
{
    return "DebugConsoleLoggerOutput()";
}

Logger::Logger(LogLevel log_level, const std::string_view name, bool use_default_outputs)
    : m_level(log_level)
    , m_name(name)
{
    if (use_default_outputs) {
        add_output(make_ref<ConsoleLoggerOutput>());
        if (platform::is_debugger_present())
            add_output(make_ref<DebugConsoleLoggerOutput>());
    }
}

ref<LoggerOutput> Logger::add_console_output(bool colored)
{
    ref<LoggerOutput> output = make_ref<ConsoleLoggerOutput>(colored);
    add_output(output);
    return output;
}

ref<LoggerOutput> Logger::add_file_output(const std::filesystem::path& path)
{
    ref<LoggerOutput> output = make_ref<FileLoggerOutput>(path);
    add_output(output);
    return output;
}

ref<LoggerOutput> Logger::add_debug_console_output()
{
    ref<LoggerOutput> output = make_ref<DebugConsoleLoggerOutput>();
    add_output(output);
    return output;
}

void Logger::use_same_outputs(const Logger& other)
{
    remove_all_outputs();
    for (auto& output : other.m_outputs)
        add_output(output);
}

void Logger::add_output(ref<LoggerOutput> output)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_outputs.insert(output);
}

void Logger::remove_output(ref<LoggerOutput> output)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_outputs.erase(output);
}

void Logger::remove_all_outputs()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_outputs.clear();
}

std::string Logger::name() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_name;
}

void Logger::set_name(std::string_view name)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_name = name;
}

LogLevel Logger::level() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_level;
}

void Logger::set_level(LogLevel level)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_level = level;
}

void Logger::log(LogLevel level, const std::string_view msg, LogFrequency frequency)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (level != LogLevel::none && level < m_level)
        return;

    if (frequency == LogFrequency::once && is_duplicate(msg))
        return;

    for (const auto& output : m_outputs)
        output->write(level, m_name, msg);
}

static Logger* s_logger;

Logger& Logger::get()
{
    SGL_ASSERT(s_logger);
    return *s_logger;
}

void Logger::static_init()
{
    s_logger = new Logger();
    // Make sure the logger is not deallocated when passing to Python.
    s_logger->inc_ref();
}

void Logger::static_shutdown()
{
    delete s_logger;
}

bool Logger::is_duplicate(const std::string_view msg)
{
    auto it = m_messages.find(msg);
    if (it != m_messages.end())
        return true;
    m_messages.insert(std::string(msg));
    return false;
}

} // namespace sgl
