// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "thread.h"

#include "sgl/core/error.h"

#include <vector>

namespace sgl::thread {

void static_init() { }

void static_shutdown() { }

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
