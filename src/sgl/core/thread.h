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

template<typename Int, typename Func>
[[nodiscard]] Task* parallel_for_async(
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

template<typename Int, typename Func>
[[nodiscard]] Task* parallel_for_async(
    const blocked_range<Int>& range,
    Func&& func,
    std::initializer_list<const Task*> parents = {},
    Pool* pool = nullptr
)
{
    return parallel_for_async(range, func, parents.begin(), parents.size(), pool);
}

template<typename Func>
[[nodiscard]] Task* do_async(Func&& func, const Task* const* parents, size_t parent_count, Pool* pool = nullptr)
{
    using BaseFunc = typename std::decay<Func>::type;

    struct Payload {
        BaseFunc f;
    };

    auto callback = [](uint32_t /* unused */, void* payload) { ((Payload*)payload)->f(); };

    if constexpr (std::is_trivially_copyable<BaseFunc>::value && std::is_trivially_destructible<BaseFunc>::value) {
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
        Payload* payload = new Payload{std::forward<Func>(func)};

        auto deleter = [](void* payload) { delete (Payload*)payload; };

        return task_submit_dep(pool, parents, (uint32_t)parent_count, 1, callback, payload, 0, deleter, 1);
    }
}

template<typename Func>
[[nodiscard]] Task* do_async(Func&& func, std::initializer_list<const Task*> parents = {}, Pool* pool = nullptr)
{
    return do_async(func, parents.begin(), parents.size(), pool);
}

} // namespace sgl::thread
