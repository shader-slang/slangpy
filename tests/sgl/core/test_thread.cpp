// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/thread.h"

#include <slang-rhi.h>

#include <atomic>

using namespace sgl;

namespace {

struct RecursiveTaskPayload {
    rhi::ITaskPool* pool;
    rhi::ITaskPool::TaskGroupHandle group;
    std::atomic<uint32_t>* execute_count;
    std::atomic<uint32_t>* delete_count;
    uint32_t depth;
};

void delete_recursive_task_payload(void* data)
{
    auto* payload = static_cast<RecursiveTaskPayload*>(data);
    payload->delete_count->fetch_add(1);
    delete payload;
}

void execute_recursive_task(void* data)
{
    auto* payload = static_cast<RecursiveTaskPayload*>(data);
    payload->execute_count->fetch_add(1);
    if (payload->depth == 0)
        return;

    for (uint32_t i = 0; i < 2; ++i) {
        auto* child = new RecursiveTaskPayload{
            payload->pool,
            payload->group,
            payload->execute_count,
            payload->delete_count,
            payload->depth - 1,
        };
        auto task
            = payload->pool->submitTask(execute_recursive_task, child, delete_recursive_task_payload, payload->group);
        payload->pool->releaseTask(task);
    }
}

} // namespace

TEST_SUITE_BEGIN("thread");

TEST_CASE("public task API executes tasks")
{
    std::atomic<bool> executed{false};
    thread::TaskHandle task = thread::do_async(
        [&executed]
        {
            executed.store(true);
        }
    );
    thread::task_wait_and_release(task);

    CHECK(executed.load());
}

TEST_CASE("rhi task pool executes tasks and deletes payloads")
{
    struct Payload {
        std::atomic<uint32_t>* execute_count;
        std::atomic<uint32_t>* delete_count;
    };

    std::atomic<uint32_t> execute_count{0};
    std::atomic<uint32_t> delete_count{0};
    auto* payload = new Payload{&execute_count, &delete_count};

    rhi::ITaskPool* pool = thread::rhi_task_pool();
    auto task = pool->submitTask(
        [](void* data)
        {
            static_cast<Payload*>(data)->execute_count->fetch_add(1);
        },
        payload,
        [](void* data)
        {
            auto* payload = static_cast<Payload*>(data);
            payload->delete_count->fetch_add(1);
            delete payload;
        }
    );
    pool->waitAndReleaseTask(task);

    CHECK(execute_count.load() == 1);
    CHECK(delete_count.load() == 1);
}

TEST_CASE("rhi task groups include tasks spawned by callbacks")
{
    std::atomic<uint32_t> execute_count{0};
    std::atomic<uint32_t> delete_count{0};
    rhi::ITaskPool* pool = thread::rhi_task_pool();
    auto group = pool->createTaskGroup();
    auto* root = new RecursiveTaskPayload{pool, group, &execute_count, &delete_count, 2};
    auto task = pool->submitTask(execute_recursive_task, root, delete_recursive_task_payload, group);
    pool->releaseTask(task);
    pool->waitAndReleaseTaskGroup(group);

    CHECK(execute_count.load() == 7);
    CHECK(delete_count.load() == 7);
}

TEST_SUITE_END();
