// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cache_writer.h"

#include "sgl/core/error.h"
#include "sgl/core/logger.h"

namespace sgl {

CacheWriter::CacheWriter(size_t max_pending_bytes)
    : m_max_pending_bytes(max_pending_bytes)
{
    SGL_CHECK(m_max_pending_bytes > 0, "Cache writer pending byte limit must be greater than zero.");
    m_thread = std::thread(
        [this]
        {
            run();
        }
    );
}

CacheWriter::~CacheWriter()
{
    {
        std::lock_guard lock(m_mutex);
        m_stop = true;
    }
    m_cv.notify_all();
    m_space_cv.notify_all();

    if (m_thread.joinable())
        m_thread.join();
}

bool CacheWriter::enqueue(size_t byte_size, std::function<void()> job)
{
    return enqueue(byte_size, nullptr, std::move(job));
}

bool CacheWriter::enqueue(size_t byte_size, std::function<void()> prepare_job, std::function<void()> job)
{
    SGL_CHECK(static_cast<bool>(job), "Cannot enqueue an empty cache write job.");

    if (!reserve(byte_size))
        return false;

    try {
        if (prepare_job)
            prepare_job();
    } catch (...) {
        release(byte_size);
        throw;
    }

    try {
        std::unique_lock lock(m_mutex);
        if (m_stop) {
            lock.unlock();
            release(byte_size);
            return false;
        }

        m_jobs.push_back(Job{byte_size, std::move(job)});
    } catch (...) {
        release(byte_size);
        throw;
    }

    m_cv.notify_one();
    return true;
}

void CacheWriter::flush() const
{
    std::unique_lock lock(m_mutex);
    m_cv.wait(
        lock,
        [this]
        {
            return m_pending_jobs == 0;
        }
    );
}

bool CacheWriter::reserve(size_t byte_size)
{
    std::unique_lock lock(m_mutex);
    if (byte_size > m_max_pending_bytes)
        return false;

    m_space_cv.wait(
        lock,
        [&]
        {
            return m_stop || byte_size <= m_max_pending_bytes - m_pending_bytes;
        }
    );

    if (m_stop)
        return false;

    m_pending_bytes += byte_size;
    m_pending_jobs++;
    return true;
}

void CacheWriter::release(size_t byte_size)
{
    {
        std::lock_guard lock(m_mutex);
        SGL_ASSERT(m_pending_jobs > 0);
        SGL_ASSERT(m_pending_bytes >= byte_size);
        m_pending_jobs--;
        m_pending_bytes -= byte_size;
    }
    m_space_cv.notify_all();
    m_cv.notify_all();
}

void CacheWriter::run()
{
    while (true) {
        Job job;
        {
            std::unique_lock lock(m_mutex);
            m_cv.wait(
                lock,
                [this]
                {
                    return !m_jobs.empty() || (m_stop && m_pending_jobs == 0);
                }
            );

            if (m_jobs.empty())
                break;

            job = std::move(m_jobs.front());
            m_jobs.pop_front();
        }

        try {
            job.func();
        } catch (const std::exception& e) {
            log_warn("Cache write job failed: {}", e.what());
        } catch (...) {
            log_warn("Cache write job failed with an unknown exception.");
        }

        release(job.byte_size);
    }
}

} // namespace sgl
