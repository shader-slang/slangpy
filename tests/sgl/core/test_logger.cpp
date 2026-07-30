// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/logger.h"

using namespace sgl;

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

TEST_SUITE_END();
