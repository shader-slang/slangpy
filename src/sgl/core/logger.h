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
/// Outputs may throw; Logger suppresses each output's exception and continues with the remaining outputs.
/// Direct calls to write() are not covered by Logger's nonthrowing guarantee.
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

    // Suppress runtime_error from direct write() calls (for example, pytest capture failures).
    // Logger already suppresses output exceptions regardless of this flag.
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
/// The once variants attempt delivery once per distinct message per Logger instance.
#define SGL_LOG_FUNC_FAMILY(name, level)                                                                               \
    inline void name(const std::string_view msg) noexcept                                                              \
    {                                                                                                                  \
        log(level, msg, LogFrequency::always);                                                                         \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void name(fmt::format_string<Args...> fmt, Args&&... args) noexcept                                         \
    {                                                                                                                  \
        if (should_log(level))                                                                                         \
            log(level, fmt::format(fmt, std::forward<Args>(args)...), LogFrequency::always);                           \
    }                                                                                                                  \
    inline void name##_once(const std::string_view msg) noexcept                                                       \
    {                                                                                                                  \
        log(level, msg, LogFrequency::once);                                                                           \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void name##_once(fmt::format_string<Args...> fmt, Args&&... args) noexcept                                  \
    {                                                                                                                  \
        if (should_log(level))                                                                                         \
            log(level, fmt::format(fmt, std::forward<Args>(args)...), LogFrequency::once);                             \
    }


/// Diagnostic logger. Emission is noexcept: output exceptions are suppressed, while formatting
/// and other internal exceptions invoke std::terminate with the active exception available to a handler.
/// Configuration operations may throw. Exceptions while evaluating arguments before a logging call
/// are the caller's responsibility.
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

    /// Log a message without propagating exceptions. Output exceptions are suppressed so the remaining
    /// outputs are still attempted. Other internal exceptions invoke std::terminate.
    /// Output failures are not logged recursively.
    /// Once messages are recorded before delivery and are not retried if an output fails.
    /// \param level The log level.
    /// \param msg The message.
    /// \param frequency The log frequency.
    void log(LogLevel level, const std::string_view msg, LogFrequency frequency = LogFrequency::always) noexcept;

    // Define logging functions.
    SGL_LOG_FUNC_FAMILY(debug, LogLevel::debug)
    SGL_LOG_FUNC_FAMILY(info, LogLevel::info)
    SGL_LOG_FUNC_FAMILY(warn, LogLevel::warn)
    SGL_LOG_FUNC_FAMILY(error, LogLevel::error)
    SGL_LOG_FUNC_FAMILY(fatal, LogLevel::fatal)

    /// Returns the lazily initialized global logger instance.
    /// Initialization failure or a call after static_shutdown() begins invokes std::terminate.
    /// The active exception can be inspected by an application-installed termination handler.
    /// Concurrent first use is supported; shutdown must be synchronized with all logger users.
    static Logger& get() noexcept;

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
    bool should_log(LogLevel level) const noexcept
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

/// Log a message through the global logger. See Logger::log() for emission behavior
/// and Logger::get() for initialization and shutdown requirements.
/// \param level The log level.
/// \param msg The message.
/// \param frequency The log frequency.
SGL_API void log(LogLevel level, std::string_view msg, LogFrequency frequency = LogFrequency::always) noexcept;

// Define global logging functions that forward to the global logger. Formatted messages are
// filtered by the Logger methods before formatting.
#define SGL_GLOBAL_LOG_FUNC_FAMILY(name)                                                                               \
    inline void log_##name(const std::string_view msg) noexcept                                                        \
    {                                                                                                                  \
        Logger::get().name(msg);                                                                                       \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void log_##name(fmt::format_string<Args...> fmt, Args&&... args) noexcept                                   \
    {                                                                                                                  \
        Logger::get().name(fmt, std::forward<Args>(args)...);                                                          \
    }                                                                                                                  \
    inline void log_##name##_once(const std::string_view msg) noexcept                                                 \
    {                                                                                                                  \
        Logger::get().name##_once(msg);                                                                                \
    }                                                                                                                  \
    template<typename... Args>                                                                                         \
    inline void log_##name##_once(fmt::format_string<Args...> fmt, Args&&... args) noexcept                            \
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

namespace detail {
    // Keep formatting inside the noexcept boundary, just like the level-specific helpers.
    template<typename T>
    void log_print(std::string_view name, T&& value) noexcept
    {
        Logger::get().log(LogLevel::none, fmt::format("{} = {}", name, std::forward<T>(value)));
    }
} // namespace detail

} // namespace sgl

/// Prints the given variable name and value. Formatting exceptions invoke std::terminate.
#define SGL_PRINT(var) ::sgl::detail::log_print(#var, var)
