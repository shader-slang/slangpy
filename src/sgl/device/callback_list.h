// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/error.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace sgl {

template<typename CallbackID, typename Callback>
class CallbackList {
public:
    using callback_id_type = CallbackID;
    using callback_type = Callback;

    CallbackList() = default;

    CallbackList(const CallbackList&) = delete;
    CallbackList& operator=(const CallbackList&) = delete;

    CallbackID register_callback(CallbackID id, Callback callback)
    {
        SGL_CHECK(static_cast<bool>(callback), "callback must not be empty");

        std::lock_guard lock(m_mutex);
        auto callbacks = m_callbacks ? std::make_shared<Storage>(*m_callbacks) : std::make_shared<Storage>();
        callbacks->push_back({id, std::move(callback)});
        m_callbacks = std::move(callbacks);
        return id;
    }

    void unregister_callback(CallbackID id)
    {
        std::lock_guard lock(m_mutex);
        if (!m_callbacks)
            return;
        const Storage& callbacks = *m_callbacks;
        const auto it = std::find_if(
            callbacks.begin(),
            callbacks.end(),
            [id](const Entry& entry)
            {
                return entry.id == id;
            }
        );
        if (it == callbacks.end())
            return;

        auto next_callbacks = std::make_shared<Storage>();
        next_callbacks->reserve(callbacks.size() - 1);
        for (const Entry& entry : callbacks) {
            if (entry.id != id)
                next_callbacks->push_back(entry);
        }
        m_callbacks = std::move(next_callbacks);
    }

    /// Release callbacks without allocating or destroying their captures under the mutex.
    void clear() noexcept
    {
        std::shared_ptr<const Storage> callbacks;
        {
            std::lock_guard lock(m_mutex);
            callbacks = std::move(m_callbacks);
        }
    }

    /// Notify callbacks, stopping and propagating the first exception.
    template<typename... Args>
    void notify(Args&&... args) const
    {
        auto callbacks = snapshot();
        if (!callbacks)
            return;
        for (const Entry& entry : *callbacks)
            entry.callback(args...);
    }

    /// Continue after callback exceptions and return the first exception, or nullptr on success.
    template<typename... Args>
    std::exception_ptr notify_no_throw(Args&&... args) const noexcept
    {
        std::exception_ptr error;
        try {
            auto callbacks = snapshot();
            if (callbacks) {
                for (const Entry& entry : *callbacks) {
                    try {
                        entry.callback(args...);
                    } catch (...) {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
        return error;
    }

private:
    struct Entry {
        CallbackID id;
        Callback callback;
    };

    using Storage = std::vector<Entry>;

    std::shared_ptr<const Storage> snapshot() const
    {
        std::lock_guard lock(m_mutex);
        return m_callbacks;
    }

    mutable std::mutex m_mutex;
    std::shared_ptr<const Storage> m_callbacks;
};

} // namespace sgl
