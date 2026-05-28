// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/logger.h"

#include <slang-rhi.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace sgl {

class DebugLogger : public rhi::IDebugCallback {
public:
    DebugLogger(
        LogLevel layer_log_level = LogLevel::none,
        LogLevel driver_log_level = LogLevel::none,
        LogLevel slang_log_level = LogLevel::none
    )
        : m_layer_log_level(layer_log_level)
        , m_driver_log_level(driver_log_level)
        , m_slang_log_level(slang_log_level)
    {
        m_logger = Logger::create(Logger::get().level(), "rhi", false);
        m_logger->use_same_outputs(Logger::get());
    }

    virtual ~DebugLogger() = default;

    virtual SLANG_NO_THROW void SLANG_MCALL
    handleMessage(rhi::DebugMessageType type, rhi::DebugMessageSource source, const char* message)
    {
        LogLevel level = LogLevel::none;
        switch (type) {
        case rhi::DebugMessageType::Info:
            level = LogLevel::info;
            break;
        case rhi::DebugMessageType::Warning:
            level = LogLevel::warn;
            break;
        case rhi::DebugMessageType::Error:
            level = LogLevel::error;
            break;
        }
        const char* source_str = "";
        LogLevel threshold = LogLevel::none;
        switch (source) {
        case rhi::DebugMessageSource::Layer:
            source_str = "layer";
            threshold = m_layer_log_level;
            break;
        case rhi::DebugMessageSource::Driver:
            source_str = "driver";
            threshold = m_driver_log_level;
            break;
        case rhi::DebugMessageSource::Slang:
            source_str = "slang";
            threshold = m_slang_log_level;
            break;
        }

        // Exit if the message log level is below the threshold for its source.
        if (level < threshold)
            return;

        std::string msg = fmt::format("{}: {}", source_str, message);
        m_logger->log(level, msg);

        // Store the message for later.
        size_t message_index = m_message_count.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_messages.size() <= message_index)
                m_messages.resize(message_index + 1);
            m_messages[message_index] = std::move(msg);
        }
    }

    size_t message_count() const { return m_message_count.load(); }

    std::string get_messages(size_t begin, size_t end)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::string result;
        if (begin < end) {
            for (size_t i = begin; i < end; ++i) {
                if (i >= m_messages.size())
                    break;
                result += m_messages[i] + "\n";
            }
        }
        return result;
    }

private:
    ref<Logger> m_logger;
    LogLevel m_layer_log_level;
    LogLevel m_driver_log_level;
    LogLevel m_slang_log_level;

    std::mutex m_mutex;
    std::atomic<size_t> m_message_count{0};
    std::vector<std::string> m_messages;
};


} // namespace sgl
