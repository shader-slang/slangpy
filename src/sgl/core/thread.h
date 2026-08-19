// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/error.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <mutex>
#include <vector>

namespace rhi {
class ITaskPool;
}

namespace sgl::thread {

// This API is a thin wrapper around nanothread. Nanothread is linked privately into SGL so all
// clients route task operations through the single nanothread instance owned by SGL. Exposing
// nanothread as a public static dependency would give every linked module its own default pool.

SGL_API void static_init();
SGL_API void static_shutdown();

/// Get the SGL task pool installed in slang-rhi.
SGL_API rhi::ITaskPool* rhi_task_pool();

/// Opaque task type.
struct Task;

/// Opaque task handle type.
using TaskHandle = Task*;

/// Task callback type.
using TaskFunc = void (*)(uint32_t index, void* payload);

/// Task payload deleter type.
using TaskPayloadDeleter = void (*)(void* payload);

/// Submit a task with optional dependencies to the SGL thread pool.
/// \param parents Parent tasks to wait for before executing.
/// \param parent_count Number of parent tasks.
/// \param size Number of work units to execute.
/// \param func Function called once for each work unit.
/// \param payload Payload passed to the function.
/// \param payload_size Size of the payload copied into the task when no deleter is provided.
/// \param payload_deleter Optional function that releases the payload after execution.
/// \param always_async Whether to disable synchronous execution of small tasks.
/// \param profile Whether to collect timing information for this task.
/// \return The submitted task, or nullptr if it completed synchronously.
SGL_API TaskHandle task_submit_dep(
    const TaskHandle* parents,
    uint32_t parent_count,
    uint32_t size = 1,
    TaskFunc func = nullptr,
    void* payload = nullptr,
    uint32_t payload_size = 0,
    TaskPayloadDeleter payload_deleter = nullptr,
    bool always_async = false,
    bool profile = false
);

/// Increase the reference count of a task.
SGL_API void task_retain(TaskHandle task);

/// Release a task handle.
SGL_API void task_release(TaskHandle task);

/// Wait for a task to complete.
SGL_API void task_wait(TaskHandle task);

/// Wait for a task to complete and release its handle.
SGL_API void task_wait_and_release(TaskHandle task);

/// Return whether a task has completed.
SGL_API bool task_query(TaskHandle task);

/// Return the execution time of a profiled task in milliseconds.
SGL_API double task_time(TaskHandle task);

/// Return the difference between the start times of two profiled tasks in milliseconds.
SGL_API double task_time_rel(TaskHandle task_1, TaskHandle task_2);

/// Run a function asynchronously in a new task.
/// See `task_submit_dep` for details.
/// \param func Function to call.
/// \param parents Parent tasks to wait for before executing.
/// \param parent_count Number of parent tasks.
/// \return The new task.
template<typename Func>
[[nodiscard]] TaskHandle do_async(Func&& func, const TaskHandle* parents, size_t parent_count)
{
    using BaseFunc = std::decay_t<Func>;

    struct Payload {
        BaseFunc func;
    };

    auto callback = [](uint32_t /* unused */, void* payload)
    {
        ((Payload*)payload)->func();
    };

    if constexpr (std::is_trivially_copyable_v<BaseFunc> && std::is_trivially_destructible_v<BaseFunc>) {
        // Payload is trivially copyable/destructible. Let the task implementation manage the payload memory.
        Payload payload{std::forward<Func>(func)};
        return task_submit_dep(parents, (uint32_t)parent_count, 1, callback, &payload, sizeof(Payload), nullptr, 1);
    } else {
        // Payload is non-trivially copyable/destructible. Manage payload manually.
        Payload* payload = new Payload{std::forward<Func>(func)};
        auto deleter = [](void* payload)
        {
            delete (Payload*)payload;
        };
        return task_submit_dep(parents, (uint32_t)parent_count, 1, callback, payload, 0, deleter, 1);
    }
}

/// Run a function asynchronously in a new task.
/// See `task_submit_dep` for details.
/// \param func Function to call.
/// \param parents Parent tasks to wait for before executing.
/// \return The new task.
template<typename Func>
[[nodiscard]] TaskHandle do_async(Func&& func, std::initializer_list<TaskHandle> parents = {})
{
    return do_async(std::forward<Func>(func), parents.begin(), parents.size());
}


template<typename Int>
struct blocked_range {
public:
    blocked_range(Int begin, Int end, Int block_size = 1)
        : m_begin(begin)
        , m_end(end)
        , m_block_size(block_size)
    {
    }

    struct iterator {
        Int value;

        iterator(Int value)
            : value(value)
        {
        }

        Int operator*() const { return value; }
        operator Int() const { return value; }

        void operator++() { value++; }
        bool operator==(const iterator& it) { return value == it.value; }
        bool operator!=(const iterator& it) { return value != it.value; }
    };

    uint32_t blocks() const { return (uint32_t)((m_end - m_begin + m_block_size - 1) / m_block_size); }

    iterator begin() const { return iterator(m_begin); }
    iterator end() const { return iterator(m_end); }
    Int block_size() const { return m_block_size; }

private:
    Int m_begin;
    Int m_end;
    Int m_block_size;
};

/// Run a parallel for-loop and block until completed.
/// \param range Loop range.
/// \param func Function to execute (taking a `blocked_range<Int>` as an argument).
template<typename Int, typename Func>
void parallel_for(const blocked_range<Int>& range, Func&& func)
{
    struct Payload {
        Func* f;
        Int begin, end, block_size;
    };

    Payload payload{&func, range.begin(), range.end(), range.block_size()};

    auto callback = [](uint32_t index, void* payload)
    {
        Payload* p = (Payload*)payload;
        Int begin = p->begin + p->block_size * (Int)index, end = begin + p->block_size;

        if (end > p->end)
            end = p->end;

        (*p->f)(blocked_range<Int>(begin, end));
    };

    TaskHandle task = task_submit_dep(nullptr, 0, range.blocks(), callback, &payload, 0, nullptr, false);
    task_wait_and_release(task);
}

/// Run a parallel for-loop asynchronously in a new task.
/// See `task_submit_dep` for details.
/// \param range Loop range.
/// \param func Function to execute (taking a `blocked_range<Int>` as an argument).
/// \param parents Parent tasks to wait for before executing.
/// \param parent_count Number of parent tasks.
/// \return The new task.
template<typename Int, typename Func>
[[nodiscard]] TaskHandle
parallel_for_async(const blocked_range<Int>& range, Func&& func, const TaskHandle* parents, size_t parent_count)
{
    using BaseFunc = typename std::decay<Func>::type;

    struct Payload {
        BaseFunc f;
        Int begin, end, block_size;
    };

    auto callback = [](uint32_t index, void* payload)
    {
        Payload* p = (Payload*)payload;
        Int begin = p->begin + p->block_size * (Int)index, end = begin + p->block_size;

        if (end > p->end)
            end = p->end;

        p->f(blocked_range<Int>(begin, end));
    };

    if constexpr (std::is_trivially_copyable<BaseFunc>::value && std::is_trivially_destructible<BaseFunc>::value) {
        Payload payload{std::forward<Func>(func), range.begin(), range.end(), range.block_size()};

        return task_submit_dep(
            parents,
            (uint32_t)parent_count,
            range.blocks(),
            callback,
            &payload,
            sizeof(Payload),
            nullptr,
            1
        );
    } else {
        Payload* payload = new Payload{std::forward<Func>(func), range.begin(), range.end(), range.block_size()};

        auto deleter = [](void* payload)
        {
            delete (Payload*)payload;
        };

        return task_submit_dep(parents, (uint32_t)parent_count, range.blocks(), callback, payload, 0, deleter, 1);
    }
}

/// Run a parallel for-loop asynchronously in a new task.
/// See `task_submit_dep` for details.
/// \param range Loop range.
/// \param func Function to execute (taking a `blocked_range<Int>` as an argument).
/// \param parents Parent tasks to wait for before executing.
/// \return The new task.
template<typename Int, typename Func>
[[nodiscard]] TaskHandle
parallel_for_async(const blocked_range<Int>& range, Func&& func, std::initializer_list<TaskHandle> parents = {})
{
    return parallel_for_async(range, func, parents.begin(), parents.size());
}

/// Helper class for managing a group of tasks.
class SGL_API TaskGroup {
public:
    TaskGroup() = default;

    ~TaskGroup()
    {
        std::lock_guard lock(m_mutex);
        SGL_CHECK(
            m_tasks.empty(),
            "Cannot destroy task group with tasks potentially still running. Call wait() before the destructor."
        );
    }

    /// Run a function asynchronously in a new task.
    /// Adds the created task to this task group.
    template<typename Func>
    TaskHandle do_async(Func&& func, TaskHandle* parents, size_t parent_count)
    {
        TaskHandle task = thread::do_async(std::forward<Func>(func), parents, parent_count);
        add_task(task);
        return task;
    }

    /// Run a function asynchronously in a new task.
    /// Adds the created task to this task group.
    template<typename Func>
    TaskHandle do_async(Func&& func, std::initializer_list<TaskHandle> parents = {})
    {
        TaskHandle task = thread::do_async(std::forward<Func>(func), parents);
        add_task(task);
        return task;
    }

    /// Add a task to this task group.
    void add_task(TaskHandle task)
    {
        std::lock_guard lock(m_mutex);
        m_tasks.push_back(task);
    }

    /// Wait for all tasks in this task group.
    void wait()
    {
        std::vector<TaskHandle> tasks;
        {
            std::lock_guard lock(m_mutex);
            std::swap(tasks, m_tasks);
        }
        for (TaskHandle task : tasks) {
            task_wait_and_release(task);
        }
    }

    SGL_NON_COPYABLE_AND_MOVABLE(TaskGroup);

private:
    std::mutex m_mutex;
    std::vector<TaskHandle> m_tasks;
};

/// Get the global task group.
/// This task group should mostly be used for fire-and-forget tasks without dependencies.
/// All tasks submitted to this group will be waited for on program shutdown.
SGL_API TaskGroup& global_task_group();

/// Wait for all tasks in the global task group.
SGL_API void wait_for_tasks();

} // namespace sgl::thread
