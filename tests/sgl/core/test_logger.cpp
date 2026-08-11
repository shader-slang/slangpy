// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/logger.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace sgl;

struct FormatProbe {
    bool* formatted;
};

template<>
struct fmt::formatter<FormatProbe> : fmt::formatter<std::string_view> {
    template<typename FormatContext>
    auto format(const FormatProbe& value, FormatContext& ctx) const
    {
        *value.formatted = true;
        return fmt::formatter<std::string_view>::format("probe", ctx);
    }
};

TEST_SUITE_BEGIN("logger");

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

    logger->debug("{}", FormatProbe{&formatted});
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

    logger->debug_once("{}", FormatProbe{&formatted});
    CHECK_FALSE(formatted);
    CHECK_EQ(output->m_write_count, 0);

    logger->set_level(LogLevel::debug);
    logger->debug_once("{}", FormatProbe{&formatted});
    CHECK(formatted);
    CHECK_EQ(output->m_write_count, 1);
}

TEST_SUITE_END();
