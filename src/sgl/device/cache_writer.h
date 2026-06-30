// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace sgl {

/// Background worker for best-effort cache write jobs.
class SGL_API CacheWriter : public Object {
    SGL_OBJECT(CacheWriter)
public:
    explicit CacheWriter(size_t max_pending_bytes = 64ull * 1024 * 1024);
    ~CacheWriter() override;

    bool enqueue(size_t byte_size, std::function<void()> job);
    bool enqueue(size_t byte_size, std::function<void()> prepare_job, std::function<void()> job);
    void flush() const;

private:
    struct Job {
        size_t byte_size;
        std::function<void()> func;
    };

    void run();
    bool reserve(size_t byte_size);
    void release(size_t byte_size);

    size_t m_max_pending_bytes;

    mutable std::mutex m_mutex;
    mutable std::condition_variable m_cv;
    mutable std::condition_variable m_space_cv;
    std::deque<Job> m_jobs;
    size_t m_pending_bytes{0};
    size_t m_pending_jobs{0};
    bool m_stop{false};
    std::thread m_thread;
};

} // namespace sgl
