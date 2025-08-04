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
    DebugLogger()
    {
        m_logger = Logger::create(Logger::get().level(), "rhi", false);
        m_logger->use_same_outputs(Logger::get());
    }

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
        switch (source) {
        case rhi::DebugMessageSource::Layer:
            source_str = "layer";
            break;
        case rhi::DebugMessageSource::Driver:
            source_str = "driver";
            break;
        case rhi::DebugMessageSource::Slang:
            source_str = "slang";
            break;
        }
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

    static DebugLogger& get()
    {
        static DebugLogger instance;
        return instance;
    }

private:
    ref<Logger> m_logger;

    std::mutex m_mutex;
    std::atomic<size_t> m_message_count{0};
    std::vector<std::string> m_messages;
};


} // namespace sgl
