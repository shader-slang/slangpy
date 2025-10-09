// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/thread.h"

#include <thread>

using namespace sgl;

TEST_SUITE_BEGIN("thread");

static int compute(int x)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return x * x;
}

TEST_CASE("do_async")
{
    size_t count = 10000;
    std::vector<Task*> tasks(count);
    std::vector<int> results(count);
    for (int i = 0; i < count; ++i) {
        tasks[i] = thread::do_async([&, i]() { results[i] = compute(i); });
    }
    for (int i = 0; i < count; ++i) {
        task_wait_and_release(tasks[i]);
        CHECK_EQ(results[i], i * i);
    }
}

template<typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
std::future<R> do_async2(F&& func, A&&... args)
{
    struct Payload {
        std::decay_t<F> func;
        std::tuple<std::decay_t<A>...> args;
        std::promise<R> promise;
    };
    Payload* payload = new Payload{std::forward<F>(func), std::make_tuple(std::forward<A>(args)...), std::promise<R>()};
    std::future<R> future = payload->promise.get_future();
    Task* task = task_submit(
        nullptr,
        1,
        [](uint32_t, void* p)
        {
            Payload* payload = static_cast<Payload*>(p);
            if constexpr (std::is_void_v<R>) {
                std::apply(payload->task, payload->args);
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

TEST_CASE("do_async2")
{
    size_t count = 10000;
    std::vector<std::future<int>> futures(count);
    for (int i = 0; i < count; ++i) {
        futures[i] = do_async2(compute, i);
    }
    for (int i = 0; i < count; ++i) {
        CHECK_EQ(futures[i].get(), i * i);
    }
}

TEST_SUITE_END();
