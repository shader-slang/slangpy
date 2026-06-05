// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/error.h"

#include <algorithm>
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

    CallbackList()
        : m_callbacks(std::make_shared<Storage>())
    {
    }

    CallbackList(const CallbackList&) = delete;
    CallbackList& operator=(const CallbackList&) = delete;

    CallbackID register_callback(CallbackID id, Callback callback)
    {
        SGL_CHECK(static_cast<bool>(callback), "callback must not be empty");

        std::lock_guard lock(m_mutex);
        auto callbacks = std::make_shared<Storage>(*m_callbacks);
        callbacks->push_back({id, std::move(callback)});
        m_callbacks = std::move(callbacks);
        return id;
    }

    void unregister_callback(CallbackID id)
    {
        std::lock_guard lock(m_mutex);
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

    void clear()
    {
        std::lock_guard lock(m_mutex);
        if (!m_callbacks->empty())
            m_callbacks = std::make_shared<Storage>();
    }

    template<typename... Args>
    void notify(Args&&... args) const
    {
        auto callbacks = snapshot();
        for (const Entry& entry : *callbacks)
            entry.callback(args...);
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
