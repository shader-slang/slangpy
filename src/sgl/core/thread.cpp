// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "thread.h"

#include "sgl/core/error.h"

#include <vector>

namespace sgl::thread {

void static_init() { }

void static_shutdown() { }

static std::vector<Task*> s_tasks;
std::mutex s_tasks_mutex;

void wait_for_tasks()
{
    std::vector<Task*> tasks;
    {
        std::lock_guard lock(s_tasks_mutex);
        std::swap(tasks, s_tasks);
    }
    for (Task* task : tasks) {
        task_wait_and_release(task);
    }
}

void register_task(Task* task)
{
    std::lock_guard lock(s_tasks_mutex);
    s_tasks.push_back(task);
}

} // namespace sgl::thread
