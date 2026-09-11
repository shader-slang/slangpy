// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/core/format.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string_view>

namespace sgl {


/// Log level.
enum class LogLevel {
    none,
    debug,
    info,
    warn,
    error,
    fatal,
};

/// Log frequency.
enum class LogFrequency {
    /// Log the message every time.
    always,
    /// Log the message only once.
    once,
};

/// Abstract base class for logger outputs.
///
/// Implementations must be thread-safe. A LoggerOutput can be shared by multiple loggers, and
/// write() may be called concurrently from multiple threads.
class SGL_API LoggerOutput : public Object {
    SGL_OBJECT(LoggerOutput)
public:
    virtual ~LoggerOutput() = default;

    /// Write a log message.
    /// \param level The log level.
    /// \param module The module name.
    /// \param msg The message.
    virtual void write(LogLevel level, const std::string_view module, const std::string_view msg) = 0;
};

/// Logger output that writes to the console.
/// Error messages are printed to stderr, all other messages to stdout.
/// Messages are optionally colored.
class SGL_API ConsoleLoggerOutput : public LoggerOutput {
public:
    ConsoleLoggerOutput(bool colored = true);

    void write(LogLevel level, const std::string_view module, const std::string_view msg) override;

    std::string to_string() const override;

    // pytest's stdout/stderr capturing sometimes leads to bad file descriptor exceptions
    // when logging in sgl. By setting IGNORE_PRINT_EXCEPTION, we ignore those exceptions.
    // Configure this flag before concurrent logging starts; concurrent mutation is not supported.
    static bool IGNORE_PRINT_EXCEPTION;

private:
    static bool enable_ansi_control_sequences();

    bool m_colored;
};

/// Logger output that writes to a file.
class SGL_API FileLoggerOutput : public LoggerOutput {
public:
    FileLoggerOutput(const std::filesystem::path& path);
    ~FileLoggerOutput();

    const std::filesystem::path& path() const { return m_path; }

    void write(LogLevel level, const std::string_view module, const std::string_view msg) override;

    std::string to_string() const override;

private:
    std::filesystem::path m_path;
    void* m_file;
};

/// Logger output that writes to the debug console (Windows only).
class SGL_API DebugConsoleLoggerOutput : public LoggerOutput {
public:
    void write(LogLevel level, const std::string_view module, const std::string_view msg) override;

    std::string to_string() const override;
};

/// Defines a family of logging functions for a given log level.
/// The functions are:
/// - name(msg)
/// - name(fmt, ...)
/// - name_once(msg)
/// - name_once(fmt, ...)
/// The once variants only log the message once per program run.
#define SGL_LOG_FUNC_FAMILY(name, level)                                                                               \
    inline void name(const std::string_view msg)                                                                       \
    {                                                                                                                  \
        log(level, msg, LogFrequency::always);                                                                         \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void name(fmt::format_string<Args...> fmt, Args&&... args)                                                  \
    {                                                                                                                  \
        if (should_log(level))                                                                                         \
            log(level, fmt::format(fmt, std::forward<Args>(args)...), LogFrequency::always);                           \
    }                                                                                                                  \
    inline void name##_once(const std::string_view msg)                                                                \
    {                                                                                                                  \
        log(level, msg, LogFrequency::once);                                                                           \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void name##_once(fmt::format_string<Args...> fmt, Args&&... args)                                           \
    {                                                                                                                  \
        if (should_log(level))                                                                                         \
            log(level, fmt::format(fmt, std::forward<Args>(args)...), LogFrequency::once);                             \
    }


class SGL_API Logger : public Object {
    SGL_OBJECT(Logger)
public:
    /// Constructor.
    /// \param level The log level to use (messages with level >= this will be logged).
    /// \param name The name of the logger.
    /// \param use_default_outputs Whether to use the default outputs (console + debug console on windows).
    Logger(LogLevel level = LogLevel::info, const std::string_view name = {}, bool use_default_outputs = true);

    static ref<Logger>
    create(LogLevel level = LogLevel::info, const std::string_view name = {}, bool use_default_outputs = true)
    {
        return make_ref<Logger>(level, name, use_default_outputs);
    }

    /// Add a console logger output.
    /// \param colored Whether to use colored output.
    /// \return The created logger output.
    ref<LoggerOutput> add_console_output(bool colored = true);

    /// Add a file logger output.
    /// \param path The path to the log file.
    /// \return The created logger output.
    ref<LoggerOutput> add_file_output(const std::filesystem::path& path);

    /// Add a debug console logger output (Windows only).
    /// \return The created logger output.
    ref<LoggerOutput> add_debug_console_output();

    /// Use the same outputs as the given logger.
    /// \param other Logger to copy outputs from.
    void use_same_outputs(const Logger& other);

    /// Add a logger output.
    /// \param output The logger output to add.
    void add_output(ref<LoggerOutput> output);

    /// Remove a logger output.
    /// \param output The logger output to remove.
    void remove_output(ref<LoggerOutput> output);

    /// Remove all logger outputs.
    void remove_all_outputs();

    /// The name of the logger.
    std::string name() const;
    void set_name(std::string_view name);

    /// The log level.
    LogLevel level() const;
    void set_level(LogLevel level);

    /// Log a message.
    /// \param level The log level.
    /// \param msg The message.
    /// \param frequency The log frequency.
    void log(LogLevel level, const std::string_view msg, LogFrequency frequency = LogFrequency::always);

    // Define logging functions.
    SGL_LOG_FUNC_FAMILY(debug, LogLevel::debug)
    SGL_LOG_FUNC_FAMILY(info, LogLevel::info)
    SGL_LOG_FUNC_FAMILY(warn, LogLevel::warn)
    SGL_LOG_FUNC_FAMILY(error, LogLevel::error)
    SGL_LOG_FUNC_FAMILY(fatal, LogLevel::fatal)

    /// Returns the global logger instance.
    static Logger& get();

    static void static_init();
    static void static_shutdown();

private:
    using OutputSet = std::set<ref<LoggerOutput>>;
    using OutputSnapshot = std::shared_ptr<const OutputSet>;

    /// Returns the current immutable output set.
    OutputSnapshot output_snapshot() const;

    /// Publishes an immutable output set and retires the previous set after unlocking.
    void publish_outputs(OutputSnapshot outputs);

    /// Applies a copy-on-write mutation without copying or destroying output references while locked.
    /// The mutator returns true if the copied set changed and may be called again after a publication race.
    template<typename Mutator>
    void mutate_outputs(Mutator&& mutator);

    /// Returns true if a message at the given level should be logged.
    bool should_log(LogLevel level) const
    {
        return level == LogLevel::none || level >= m_level.load(std::memory_order_relaxed);
    }

    /// Checks if the given message has already been logged.
    bool is_duplicate(const std::string_view msg);

    std::atomic<LogLevel> m_level{LogLevel::info};
    std::string m_name;

    // Published output sets are immutable. log() retains one shared snapshot while holding m_mutex,
    // then calls LoggerOutput::write() after releasing the mutex. An output removed concurrently may
    // therefore receive a call already in progress. Output implementations must be thread-safe.
    OutputSnapshot m_outputs;
    std::set<std::string, std::less<>> m_messages;

    mutable std::mutex m_mutex;
};

// Define global logging functions that forward to the global logger. Formatted messages are
// filtered by the Logger methods before formatting.
#define SGL_GLOBAL_LOG_FUNC_FAMILY(name)                                                                               \
    inline void log_##name(const std::string_view msg)                                                                 \
    {                                                                                                                  \
        Logger::get().name(msg);                                                                                       \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void log_##name(fmt::format_string<Args...> fmt, Args&&... args)                                            \
    {                                                                                                                  \
        Logger::get().name(fmt, std::forward<Args>(args)...);                                                          \
    }                                                                                                                  \
    inline void log_##name##_once(const std::string_view msg)                                                          \
    {                                                                                                                  \
        Logger::get().name##_once(msg);                                                                                \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void log_##name##_once(fmt::format_string<Args...> fmt, Args&&... args)                                     \
    {                                                                                                                  \
        Logger::get().name##_once(fmt, std::forward<Args>(args)...);                                                   \
    }

SGL_GLOBAL_LOG_FUNC_FAMILY(debug)
SGL_GLOBAL_LOG_FUNC_FAMILY(info)
SGL_GLOBAL_LOG_FUNC_FAMILY(warn)
SGL_GLOBAL_LOG_FUNC_FAMILY(error)
SGL_GLOBAL_LOG_FUNC_FAMILY(fatal)

#undef SGL_GLOBAL_LOG_FUNC_FAMILY
#undef SGL_LOG_FUNC_FAMILY

} // namespace sgl

/// Prints the given variable name and value.
#define SGL_PRINT(var) ::sgl::Logger::get().log(::sgl::LogLevel::none, ::fmt::format("{} = {}", #var, var))
