// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/cuda_cleanup.h"
#include "sgl/device/cuda_utils.h"
#include "sgl/device/device.h"

#include <exception>
#include <stdexcept>
#include <thread>

using namespace sgl;

TEST_SUITE_BEGIN("cuda_cleanup");

namespace {

class CleanupLogOutput : public LoggerOutput {
    SGL_OBJECT(CleanupLogOutput)
public:
    void write(LogLevel level, std::string_view, std::string_view message) override
    {
        if (level == LogLevel::error) {
            messages.emplace_back(message);
            if (throw_on_write)
                throw std::runtime_error("logger output failed");
        }
    }

    std::vector<std::string> messages;
    bool throw_on_write{false};
};

struct CleanupLogScope {
    ref<CleanupLogOutput> output = make_ref<CleanupLogOutput>();
    CleanupLogScope() { Logger::get().add_output(output); }
    ~CleanupLogScope() { Logger::get().remove_output(output); }
};

struct CleanupCalls {
    std::string failure;
    std::string calls;

    CUresult record(const char* operation)
    {
        if (!calls.empty())
            calls += ',';
        calls += operation;
        return failure == operation ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
    }
};

CleanupCalls* cleanup_calls;

// These entry points belong to the test module. Exercise the same internal
// helper used by ExternalMemory, without overriding CUDA calls in sgl.dll.
struct CleanupDriverScope {
    decltype(cuCtxPushCurrent) push = cuCtxPushCurrent;
    decltype(cuCtxPopCurrent) pop = cuCtxPopCurrent;
    decltype(cuMemFree) free = cuMemFree;
    decltype(cuDestroyExternalMemory) destroy = cuDestroyExternalMemory;

    explicit CleanupDriverScope(CleanupCalls& calls)
    {
        cleanup_calls = &calls;
        cuCtxPushCurrent = [](CUcontext)
        {
            return cleanup_calls->record("cuCtxPushCurrent");
        };
        cuCtxPopCurrent = [](CUcontext*)
        {
            return cleanup_calls->record("cuCtxPopCurrent");
        };
        cuMemFree = [](CUdeviceptr)
        {
            return cleanup_calls->record("cuMemFree");
        };
        cuDestroyExternalMemory = [](CUexternalMemory)
        {
            return cleanup_calls->record("cuDestroyExternalMemory");
        };
    }

    ~CleanupDriverScope()
    {
        cuCtxPushCurrent = push;
        cuCtxPopCurrent = pop;
        cuMemFree = free;
        cuDestroyExternalMemory = destroy;
        cleanup_calls = nullptr;
    }
};

} // namespace

TEST_CASE("external_memory_cleanup_failures")
{
    struct Case {
        const char* failure;
        bool mapped;
        bool has_context;
        const char* calls;
    };
    const Case cases[] = {
        {"", true, true, "cuCtxPushCurrent,cuMemFree,cuDestroyExternalMemory,cuCtxPopCurrent"},
        {"", false, true, "cuCtxPushCurrent,cuDestroyExternalMemory,cuCtxPopCurrent"},
        {"cuCtxPushCurrent", true, true, "cuCtxPushCurrent"},
        {"cuCtxPushCurrent", true, false, ""},
        {"cuMemFree", true, true, "cuCtxPushCurrent,cuMemFree,cuCtxPopCurrent"},
        {"cuDestroyExternalMemory", true, true, "cuCtxPushCurrent,cuMemFree,cuDestroyExternalMemory,cuCtxPopCurrent"},
        {"cuCtxPopCurrent", true, true, "cuCtxPushCurrent,cuMemFree,cuDestroyExternalMemory,cuCtxPopCurrent"},
    };
    for (const auto& test : cases) {
        CAPTURE(test.failure);
        CAPTURE(test.mapped);
        CAPTURE(test.has_context);
        // Even a throwing custom logger output must not skip context restoration.
        for (bool throw_on_write : {false, true}) {
            CAPTURE(throw_on_write);
            CleanupCalls calls{test.failure};
            CleanupDriverScope driver(calls);
            CleanupLogScope log;
            log.output->throw_on_write = throw_on_write;
            int token;
            cuda::detail::destroy_external_memory(
                test.has_context ? reinterpret_cast<CUcontext>(&token) : nullptr,
                test.mapped ? &token : nullptr,
                reinterpret_cast<CUexternalMemory>(&token)
            );
            CHECK(calls.calls == test.calls);
            if (*test.failure) {
                REQUIRE(log.output->messages.size() == 1);
                CHECK(log.output->messages[0].find(test.failure) != std::string::npos);
                CHECK(
                    log.output->messages[0].find(std::to_string(static_cast<int>(CUDA_ERROR_INVALID_CONTEXT)))
                    != std::string::npos
                );
            } else {
                CHECK(log.output->messages.empty());
            }
        }
    }
}

TEST_CASE_GPU("context_pop_failure_preserves_original_exception")
{
    if (ctx.device->info().adapter_name.find("NVIDIA") == std::string::npos)
        SKIP("CUDA cleanup requires an NVIDIA adapter");
    struct DriverScope {
        bool loaded = rhiCudaDriverApiInit();
        ~DriverScope()
        {
            if (loaded)
                rhiCudaDriverApiShutdown();
        }
    } driver;
    if (!driver.loaded)
        SKIP("CUDA driver API is unavailable");

    auto device = make_ref<cuda::Device>(ctx.device);
    CleanupLogScope log;
    for (bool throw_on_write : {false, true}) {
        log.output->throw_on_write = throw_on_write;
        log.output->messages.clear();
        std::exception_ptr exception;
        // A new thread starts with an empty context stack. Remove the scope's
        // context to make its real driver pop fail during exception unwinding.
        std::thread worker(
            [&]()
            {
                try {
                    cuda::ContextScope scope(device.get());
                    CUcontext previous;
                    SGL_CU_CHECK(cuCtxPopCurrent(&previous));
                    throw std::runtime_error("original exception");
                } catch (...) {
                    exception = std::current_exception();
                }
            }
        );
        worker.join();
        REQUIRE(exception);
        CHECK_THROWS_WITH(std::rethrow_exception(exception), "original exception");
        REQUIRE(log.output->messages.size() == 1);
        CHECK(log.output->messages[0].find("cuCtxPopCurrent") != std::string::npos);
    }

    // Ordinary CUDA operations must still report errors as exceptions.
    CHECK_THROWS(cuda::destroy_external_memory(nullptr));
}

TEST_SUITE_END();
