// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "thread.h"

#include "sgl/core/error.h"
#include "sgl/core/short_vector.h"

#include <nanothread/nanothread.h>
#include <slang-rhi.h>

#include <mutex>
#include <vector>

namespace sgl::thread {

namespace {

    /// Adapter that executes slang-rhi tasks on a nanothread pool.
    class NanothreadTaskPool : public rhi::ITaskPool {
    public:
        explicit NanothreadTaskPool(Pool* pool = nullptr)
            : m_pool(pool)
        {
        }

        SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid, void** out_object) override
        {
            if (uuid == ISlangUnknown::getTypeGuid() || uuid == rhi::ITaskPool::getTypeGuid()) {
                *out_object = static_cast<rhi::ITaskPool*>(this);
                return SLANG_OK;
            }
            *out_object = nullptr;
            return SLANG_E_NO_INTERFACE;
        }

        // This adapter has process lifetime and is not reference counted.
        SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 2; }
        SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 2; }

        SLANG_NO_THROW rhi::ITaskPool::TaskHandle SLANG_MCALL
        submitTask(void (*func)(void*), void* payload, void (*payload_deleter)(void*), TaskGroupHandle group) override
        {
            SGL_ASSERT(func);

            TaskGroup* task_group = static_cast<TaskGroup*>(group);
            SGL_ASSERT(!task_group || task_group->owner == this);

            TaskPayload* task_payload = new TaskPayload{func, payload, payload_deleter};
            ::Task* task = ::task_submit(m_pool, 1, execute_task, task_payload, 0, delete_task_payload, 1);
            SGL_ASSERT(task);

            if (task_group) {
                // The group owns a reference independently of the handle returned to the caller.
                ::task_retain(task);
                std::lock_guard lock(task_group->mutex);
                task_group->tasks.push_back(task);
            }

            return task;
        }

        SLANG_NO_THROW void SLANG_MCALL releaseTask(rhi::ITaskPool::TaskHandle task) override
        {
            SGL_ASSERT(task);
            ::task_release(static_cast<::Task*>(task));
        }

        SLANG_NO_THROW void SLANG_MCALL waitAndReleaseTask(rhi::ITaskPool::TaskHandle task) override
        {
            SGL_ASSERT(task);
            ::task_wait_and_release(static_cast<::Task*>(task));
        }

        SLANG_NO_THROW TaskGroupHandle SLANG_MCALL createTaskGroup() override { return new TaskGroup{this}; }

        SLANG_NO_THROW void SLANG_MCALL waitAndReleaseTaskGroup(TaskGroupHandle group) override
        {
            SGL_ASSERT(group);
            TaskGroup* task_group = static_cast<TaskGroup*>(group);
            SGL_ASSERT(task_group->owner == this);

            // A task in one batch may submit more tasks to the same group. Waiting for every
            // task in the batch guarantees that all such submissions are visible before the
            // group can be observed as empty.
            std::vector<::Task*> tasks;
            while (true) {
                {
                    std::lock_guard lock(task_group->mutex);
                    if (task_group->tasks.empty())
                        break;
                    tasks.swap(task_group->tasks);
                }

                for (::Task* task : tasks)
                    ::task_wait_and_release(task);
                tasks.clear();
            }

            delete task_group;
        }

    private:
        struct TaskGroup {
            NanothreadTaskPool* owner;
            std::mutex mutex;
            std::vector<::Task*> tasks;
        };

        struct TaskPayload {
            void (*func)(void*);
            void* payload;
            void (*payload_deleter)(void*);
        };

        static void execute_task(uint32_t, void* payload)
        {
            TaskPayload* task_payload = static_cast<TaskPayload*>(payload);
            task_payload->func(task_payload->payload);
        }

        static void delete_task_payload(void* payload)
        {
            TaskPayload* task_payload = static_cast<TaskPayload*>(payload);
            if (task_payload->payload_deleter)
                task_payload->payload_deleter(task_payload->payload);
            delete task_payload;
        }

        Pool* m_pool;
    };

    NanothreadTaskPool s_rhi_task_pool;

    ::Task* unwrap_task(TaskHandle task)
    {
        return reinterpret_cast<::Task*>(task);
    }

    TaskHandle wrap_task(::Task* task)
    {
        return reinterpret_cast<TaskHandle>(task);
    }

} // namespace

TaskHandle task_submit_dep(
    const TaskHandle* parents,
    uint32_t parent_count,
    uint32_t size,
    TaskFunc func,
    void* payload,
    uint32_t payload_size,
    TaskPayloadDeleter payload_deleter,
    bool always_async,
    bool profile
)
{
    short_vector<const ::Task*, 16> unwrapped_parents;
    unwrapped_parents.reserve(parent_count);
    for (uint32_t i = 0; i < parent_count; ++i)
        unwrapped_parents.push_back(unwrap_task(parents[i]));

    return wrap_task(
        ::task_submit_dep(
            nullptr,
            unwrapped_parents.data(),
            parent_count,
            size,
            func,
            payload,
            payload_size,
            payload_deleter,
            always_async ? 1 : 0,
            profile ? 1 : 0
        )
    );
}

void task_retain(TaskHandle task)
{
    ::task_retain(unwrap_task(task));
}

void task_release(TaskHandle task)
{
    ::task_release(unwrap_task(task));
}

void task_wait(TaskHandle task)
{
    ::task_wait(unwrap_task(task));
}

void task_wait_and_release(TaskHandle task)
{
    ::task_wait_and_release(unwrap_task(task));
}

bool task_query(TaskHandle task)
{
    return ::task_query(unwrap_task(task));
}

double task_time(TaskHandle task)
{
    return ::task_time(unwrap_task(task));
}

double task_time_rel(TaskHandle task_1, TaskHandle task_2)
{
    return ::task_time_rel(unwrap_task(task_1), unwrap_task(task_2));
}

void static_init()
{
    SGL_CHECK(SLANG_SUCCEEDED(rhi::getRHI()->setTaskPool(&s_rhi_task_pool)), "Failed to set slang-rhi task pool.");
}

void static_shutdown()
{
    SGL_CHECK(SLANG_SUCCEEDED(rhi::getRHI()->setTaskPool(nullptr)), "Failed to reset slang-rhi task pool.");
}

rhi::ITaskPool* rhi_task_pool()
{
    return &s_rhi_task_pool;
}

TaskGroup& global_task_group()
{
    static TaskGroup s_global_task_group;
    return s_global_task_group;
}

void wait_for_tasks()
{
    global_task_group().wait();
}

} // namespace sgl::thread
