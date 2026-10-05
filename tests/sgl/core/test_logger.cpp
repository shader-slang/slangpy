// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/logger.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace sgl;

struct FormatProbe {
    bool* formatted;
    bool fail{false};
};

template<>
struct fmt::formatter<FormatProbe> : fmt::formatter<std::string_view> {
    template<typename FormatContext>
    auto format(const FormatProbe& value, FormatContext& ctx) const
    {
        *value.formatted = true;
        if (value.fail)
            throw std::runtime_error("formatter failed");
        return fmt::formatter<std::string_view>::format("probe", ctx);
    }
};

TEST_SUITE_BEGIN("logger");

static_assert(noexcept(Logger::get()));
static_assert(noexcept(sgl::log(LogLevel::info, std::string_view{})));
static_assert(noexcept(std::declval<Logger&>().log(LogLevel::info, std::string_view{})));
static_assert(noexcept(SGL_PRINT(std::declval<int>())));

// Check the public exception specification as well as the runtime behavior below.
#define CHECK_LOG_NOEXCEPT(name)                                                                                       \
    static_assert(noexcept(std::declval<Logger&>().name(std::string_view{})));                                         \
    static_assert(noexcept(std::declval<Logger&>().name(std::declval<fmt::format_string<int>>(), 1)));                 \
    static_assert(noexcept(std::declval<Logger&>().name##_once(std::string_view{})));                                  \
    static_assert(noexcept(std::declval<Logger&>().name##_once(std::declval<fmt::format_string<int>>(), 1)));          \
    static_assert(noexcept(log_##name(std::string_view{})));                                                           \
    static_assert(noexcept(log_##name(std::declval<fmt::format_string<int>>(), 1)));                                   \
    static_assert(noexcept(log_##name##_once(std::string_view{})));                                                    \
    static_assert(noexcept(log_##name##_once(std::declval<fmt::format_string<int>>(), 1)));

CHECK_LOG_NOEXCEPT(debug)
CHECK_LOG_NOEXCEPT(info)
CHECK_LOG_NOEXCEPT(warn)
CHECK_LOG_NOEXCEPT(error)
CHECK_LOG_NOEXCEPT(fatal)

#undef CHECK_LOG_NOEXCEPT

class ReentrantLoggerOutput : public LoggerOutput {
    SGL_OBJECT(ReentrantLoggerOutput)
public:
    explicit ReentrantLoggerOutput(Logger* logger)
        : m_logger(logger)
    {
    }

    void write(LogLevel level, const std::string_view module, const std::string_view msg) override
    {
        m_level = level;
        m_module = module;
        m_msg = msg;
        m_logger_name = m_logger->name();
        m_write_count++;
    }

    Logger* m_logger;
    LogLevel m_level{LogLevel::none};
    std::string m_module;
    std::string m_msg;
    std::string m_logger_name;
    size_t m_write_count{0};
};

class CountingLoggerOutput : public LoggerOutput {
    SGL_OBJECT(CountingLoggerOutput)
public:
    void write(LogLevel, const std::string_view, const std::string_view) override { m_write_count++; }

    std::atomic<size_t> m_write_count{0};
};

class ThrowingLoggerOutput : public LoggerOutput {
    SGL_OBJECT(ThrowingLoggerOutput)
public:
    explicit ThrowingLoggerOutput(bool nonstandard_exception = false)
        : m_nonstandard_exception(nonstandard_exception)
    {
    }

    void write(LogLevel, const std::string_view, const std::string_view) override
    {
        m_write_count++;
        if (m_nonstandard_exception)
            throw 42;
        throw std::runtime_error("output failed");
    }

    std::atomic<size_t> m_write_count{0};

private:
    bool m_nonstandard_exception;
};

class ScopedGlobalLoggerOutputs {
public:
    ScopedGlobalLoggerOutputs()
        : m_saved(Logger::create(Logger::get().level(), "", false))
    {
        m_saved->use_same_outputs(Logger::get());
        Logger::get().remove_all_outputs();
        Logger::get().set_level(LogLevel::debug);
    }

    ~ScopedGlobalLoggerOutputs()
    {
        Logger::get().use_same_outputs(*m_saved);
        Logger::get().set_level(m_saved->level());
    }

private:
    ref<Logger> m_saved;
};

class RefCountLoggerOutput : public LoggerOutput {
    SGL_OBJECT(RefCountLoggerOutput)
public:
    void write(LogLevel, const std::string_view, const std::string_view) override
    {
        m_ref_count_during_write = ref_count();
    }

    uint64_t m_ref_count_during_write{0};
};

struct BlockingLoggerOutputState {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered{false};
    bool release{false};
    std::atomic<bool> destroyed{false};
};

class BlockingLoggerOutput : public LoggerOutput {
    SGL_OBJECT(BlockingLoggerOutput)
public:
    explicit BlockingLoggerOutput(std::shared_ptr<BlockingLoggerOutputState> state)
        : m_state(std::move(state))
    {
    }

    ~BlockingLoggerOutput() { m_state->destroyed = true; }

    void write(LogLevel, const std::string_view, const std::string_view) override
    {
        std::unique_lock lock(m_state->mutex);
        m_state->entered = true;
        m_state->condition.notify_all();
        m_state->condition.wait(
            lock,
            [&]()
            {
                return m_state->release;
            }
        );
    }

private:
    std::shared_ptr<BlockingLoggerOutputState> m_state;
};

TEST_CASE("output callback can re-enter logger")
{
    auto logger = Logger::create(LogLevel::info, "test", false);
    auto output = make_ref<ReentrantLoggerOutput>(logger.get());
    logger->add_output(output);

    logger->warn("message");

    CHECK_EQ(output->m_write_count, 1);
    CHECK_EQ(output->m_level, LogLevel::warn);
    CHECK_EQ(output->m_module, "test");
    CHECK_EQ(output->m_msg, "message");
    CHECK_EQ(output->m_logger_name, "test");
}

TEST_CASE("logging retains the output set snapshot")
{
    auto logger = Logger::create(LogLevel::info, "test", false);
    auto output = make_ref<RefCountLoggerOutput>();
    logger->add_output(output);
    const uint64_t ref_count_before_log = output->ref_count();

    logger->warn("message");

    CHECK_EQ(output->m_ref_count_during_write, ref_count_before_log);
}

TEST_CASE("output snapshot keeps a removed output alive during a callback")
{
    auto logger = Logger::create(LogLevel::info, "test", false);
    auto state = std::make_shared<BlockingLoggerOutputState>();
    auto output = make_ref<BlockingLoggerOutput>(state);
    logger->add_output(output);

    std::thread worker(
        [logger]()
        {
            logger->warn("message");
        }
    );

    {
        std::unique_lock lock(state->mutex);
        REQUIRE(state->condition.wait_for(
            lock,
            std::chrono::seconds(5),
            [&]()
            {
                return state->entered;
            }
        ));
    }

    logger->remove_output(output);
    output.reset();
    CHECK_FALSE(state->destroyed);

    {
        std::lock_guard lock(state->mutex);
        state->release = true;
    }
    state->condition.notify_all();
    worker.join();

    CHECK(state->destroyed);
}

TEST_CASE("use_same_outputs copies an immutable snapshot")
{
    auto source = Logger::create(LogLevel::info, "source", false);
    auto target = Logger::create(LogLevel::info, "target", false);
    auto first = make_ref<CountingLoggerOutput>();
    auto second = make_ref<CountingLoggerOutput>();
    source->add_output(first);

    target->use_same_outputs(*source);
    target->use_same_outputs(*target);
    source->add_output(second);

    target->warn("target");
    CHECK_EQ(first->m_write_count, 1);
    CHECK_EQ(second->m_write_count, 0);

    source->warn("source");
    CHECK_EQ(first->m_write_count, 2);
    CHECK_EQ(second->m_write_count, 1);

    target->remove_all_outputs();
    target->warn("removed");
    CHECK_EQ(first->m_write_count, 2);
    CHECK_EQ(second->m_write_count, 1);
}

TEST_CASE("concurrent output mutations preserve all updates")
{
    auto logger = Logger::create(LogLevel::info, "test", false);
    constexpr size_t output_count = 16;
    std::vector<ref<CountingLoggerOutput>> outputs;
    std::vector<std::thread> workers;

    outputs.reserve(output_count);
    workers.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i)
        outputs.push_back(make_ref<CountingLoggerOutput>());

    for (const auto& output : outputs)
        workers.emplace_back(
            [logger, output]()
            {
                logger->add_output(output);
            }
        );
    for (auto& worker : workers)
        worker.join();

    logger->warn("added");
    for (const auto& output : outputs)
        CHECK_EQ(output->m_write_count, 1);

    workers.clear();
    for (size_t i = 0; i < output_count; i += 2)
        workers.emplace_back(
            [logger, output = outputs[i]]()
            {
                logger->remove_output(output);
            }
        );
    for (auto& worker : workers)
        worker.join();

    logger->warn("removed");
    for (size_t i = 0; i < output_count; ++i)
        CHECK_EQ(outputs[i]->m_write_count, i % 2 == 0 ? 1 : 2);
}

TEST_CASE("filtered formatted messages avoid formatting")
{
    auto logger = Logger::create(LogLevel::info, "test", false);
    auto output = make_ref<CountingLoggerOutput>();
    logger->add_output(output);
    bool formatted = false;

    logger->debug("{}", FormatProbe{&formatted, true});
    CHECK_FALSE(formatted);
    CHECK_EQ(output->m_write_count, 0);

    logger->set_level(LogLevel::debug);
    CHECK_EQ(logger->level(), LogLevel::debug);
    logger->debug("{}", FormatProbe{&formatted});
    CHECK(formatted);
    CHECK_EQ(output->m_write_count, 1);
}

TEST_CASE("filtered formatted once messages avoid formatting")
{
    auto logger = Logger::create(LogLevel::info, "test", false);
    auto output = make_ref<CountingLoggerOutput>();
    logger->add_output(output);
    bool formatted = false;

    logger->debug_once("{}", FormatProbe{&formatted, true});
    CHECK_FALSE(formatted);
    CHECK_EQ(output->m_write_count, 0);

    logger->set_level(LogLevel::debug);
    logger->debug_once("{}", FormatProbe{&formatted});
    CHECK(formatted);
    CHECK_EQ(output->m_write_count, 1);
}

TEST_CASE("output failures are isolated and once messages are not retried")
{
    auto logger = Logger::create(LogLevel::debug, "test", false);
    auto first = make_ref<ThrowingLoggerOutput>();
    auto second = make_ref<ThrowingLoggerOutput>(true);
    auto healthy = make_ref<CountingLoggerOutput>();
    logger->add_output(first);
    logger->add_output(second);
    logger->add_output(healthy);

    // Both throwing outputs must be called, regardless of pointer ordering in the output set.
    CHECK_NOTHROW(logger->log(LogLevel::info, "plain"));
    CHECK_NOTHROW(logger->warn("formatted {}", 42));
    CHECK_NOTHROW(logger->warn_once("once {}", 42));
    CHECK_NOTHROW(logger->warn_once("once {}", 42));
    CHECK_EQ(first->m_write_count, 3);
    CHECK_EQ(second->m_write_count, 3);
    CHECK_EQ(healthy->m_write_count, 3);

    logger->remove_output(first);
    logger->remove_output(second);
    logger->warn("still usable");
    CHECK_EQ(healthy->m_write_count, 4);
}

TEST_CASE("global logging and SGL_PRINT suppress output failures")
{
    ScopedGlobalLoggerOutputs restore_outputs;
    auto& logger = Logger::get();
    auto first = make_ref<ThrowingLoggerOutput>();
    auto second = make_ref<ThrowingLoggerOutput>(true);
    auto healthy = make_ref<CountingLoggerOutput>();
    logger.add_output(first);
    logger.add_output(second);
    logger.add_output(healthy);

    CHECK_NOTHROW(sgl::log(LogLevel::none, "plain"));
    CHECK_NOTHROW(log_warn("message"));
    CHECK_NOTHROW(log_warn("formatted {}", 42));
    CHECK_NOTHROW(log_warn_once("global once"));
    CHECK_NOTHROW(log_warn_once("global once"));
    CHECK_NOTHROW(log_warn_once("global once {}", 42));
    CHECK_NOTHROW(log_warn_once("global once {}", 42));
    int value = 42;
    CHECK_NOTHROW(SGL_PRINT(value));
    CHECK_EQ(first->m_write_count, 6);
    CHECK_EQ(second->m_write_count, 6);
    CHECK_EQ(healthy->m_write_count, 6);
}

TEST_CASE("output failures during unwinding preserve the original exception")
{
    auto logger = Logger::create(LogLevel::warn, "test", false);
    logger->add_output(make_ref<ThrowingLoggerOutput>());
    auto healthy = make_ref<CountingLoggerOutput>();
    logger->add_output(healthy);
    bool formatted = false;
    struct LogOnDestruction {
        Logger* logger;
        bool* formatted;
        ~LogOnDestruction()
        {
            logger->warn("cleanup diagnostic");
            logger->warn("{}", FormatProbe{formatted});
        }
    };

    auto fail = [&]
    {
        LogOnDestruction guard{logger.get(), &formatted};
        throw std::runtime_error("original failure");
    };
    CHECK_THROWS_WITH_AS(fail(), "original failure", std::runtime_error);
    CHECK(formatted);
    CHECK_EQ(healthy->m_write_count, 2);
}

TEST_SUITE_END();
