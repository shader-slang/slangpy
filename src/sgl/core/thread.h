// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <nanothread/nanothread.h>

#include <type_traits>
#include <future>

namespace sgl::thread {

SGL_API void static_init();
SGL_API void static_shutdown();

/// Block until all register tasks are completed.
SGL_API void wait_for_tasks();

/// Register a task to be waited for on next wait_for_tasks() call.
SGL_API void register_task(Task* task);

/// Submit a new task.
/// \param func Function to call.
/// \param parents Parent tasks to wait for before executing.
/// \param parent_count Number of parent tasks.
/// \param pool Task pool to submit to (using default pool if not specified).
/// \return The new task.
template<typename Func>
[[nodiscard]] Task* submit_task(Func&& func, const Task* const* parents, size_t parent_count, Pool* pool = nullptr)
{
    using BaseFunc = std::decay_t<Func>;

    struct Payload {
        BaseFunc func;
    };

    auto callback = [](uint32_t /* unused */, void* payload) { ((Payload*)payload)->func(); };

    if constexpr (std::is_trivially_copyable_v<BaseFunc> && std::is_trivially_destructible_v<BaseFunc>) {
        // Payload is trivially copyable/destructible. Let nanothread manage the payload memory.
        Payload payload{std::forward<Func>(func)};
        return task_submit_dep(
            pool,
            parents,
            (uint32_t)parent_count,
            1,
            callback,
            &payload,
            sizeof(Payload),
            nullptr,
            1
        );
    } else {
        // Payload is non-trivially copyable/destructible. Manage payload manually.
        Payload* payload = new Payload{std::forward<Func>(func)};
        auto deleter = [](void* payload) { delete (Payload*)payload; };
        return task_submit_dep(pool, parents, (uint32_t)parent_count, 1, callback, payload, 0, deleter, 1);
    }
}

/// Submit a new task.
/// \param func Function to call.
/// \param parents Parent tasks to wait for before executing.
/// \param pool Task pool to submit to (using default pool if not specified).
/// \return The new task.
template<typename Func>
[[nodiscard]] Task* submit_task(Func&& func, std::initializer_list<const Task*> parents = {}, Pool* pool = nullptr)
{
    return submit_task(func, parents.begin(), parents.size(), pool);
}

/// Asynchronously run a function in the default thread pool.
/// Return a std::future that will eventually hold the result of that function call.
/// \param func Function to call.
/// \param args Arguments passed to the function.
/// \return A future that will eventually hold the result.
template<
    typename Func,
    typename... Args,
    typename Result = std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>>
std::future<Result> do_async(Func&& func, Args&&... args)
{
    struct Payload {
        std::decay_t<Func> func;
        std::tuple<std::decay_t<Args>...> args;
        std::promise<Result> promise;
    };
    Payload* payload
        = new Payload{std::forward<Func>(func), std::make_tuple(std::forward<Args>(args)...), std::promise<Result>()};
    std::future<Result> future = payload->promise.get_future();
    Task* task = task_submit(
        nullptr,
        1,
        [](uint32_t, void* p)
        {
            Payload* payload = static_cast<Payload*>(p);
            if constexpr (std::is_void_v<Result>) {
                std::apply(payload->func, payload->args);
                payload->promise.set_value();
            } else {
                payload->promise.set_value(std::apply(payload->func, payload->args));
            }
            delete payload;
        },
        payload,
        0,
        nullptr,
        1
    );
    task_release(task);
    return future;
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
/// \param pool Task pool to submit to (using default pool if not specified).
template<typename Int, typename Func>
void parallel_for(const blocked_range<Int>& range, Func&& func, Pool* pool = nullptr)
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

    task_submit_and_wait(pool, range.blocks(), callback, &payload);
}

/// Submit a parallel for-loop.
/// \param range Loop range.
/// \param func Function to execute (taking a `blocked_range<Int>` as an argument).
/// \param parents Parent tasks to wait for before executing.
/// \param parent_count Number of parent tasks.
/// \param pool Task pool to submit to (using default pool if not specified).
/// \return The new task.
template<typename Int, typename Func>
[[nodiscard]] Task* submit_parallel_for(
    const blocked_range<Int>& range,
    Func&& func,
    const Task* const* parents,
    size_t parent_count,
    Pool* pool = nullptr
)
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
            pool,
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

        auto deleter = [](void* payload) { delete (Payload*)payload; };

        return task_submit_dep(pool, parents, (uint32_t)parent_count, range.blocks(), callback, payload, 0, deleter, 1);
    }
}

/// Submit a parallel for-loop.
/// \param range Loop range.
/// \param func Function to execute (taking a `blocked_range<Int>` as an argument).
/// \param parents Parent tasks to wait for before executing.
/// \param pool Task pool to submit to (using default pool if not specified).
/// \return The new task.
template<typename Int, typename Func>
[[nodiscard]] Task* submit_parallel_for(
    const blocked_range<Int>& range,
    Func&& func,
    std::initializer_list<const Task*> parents = {},
    Pool* pool = nullptr
)
{
    return submit_parallel_for(range, func, parents.begin(), parents.size(), pool);
}

} // namespace sgl::thread
