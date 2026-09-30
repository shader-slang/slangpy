// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/logger.h"

#include <slang-rhi/cuda-driver-api.h>

namespace sgl::cuda::detail {

// Destructors cannot propagate CUDA or logger-output exceptions to the caller.
inline bool check_cleanup(CUresult result, const char* operation) noexcept
{
    if (result == CUDA_SUCCESS)
        return true;
    try {
        log_error("CUDA cleanup {} failed with error {}.", operation, static_cast<int>(result));
    } catch (...) {
        // A failing logger output must not interrupt the remaining cleanup.
    }
    return false;
}

inline void pop_context() noexcept
{
    CUcontext previous;
    check_cleanup(cuCtxPopCurrent(&previous), "cuCtxPopCurrent");
}

inline void destroy_external_memory(CUcontext context, void* mapped_data, CUexternalMemory memory) noexcept
{
    if (!check_cleanup(context ? cuCtxPushCurrent(context) : CUDA_ERROR_INVALID_CONTEXT, "cuCtxPushCurrent"))
        return;

    // A failed free leaves mapping cleanup incomplete. Do not destroy its import,
    // but always restore the context after a successful push.
    if (!mapped_data || check_cleanup(cuMemFree(reinterpret_cast<CUdeviceptr>(mapped_data)), "cuMemFree"))
        check_cleanup(cuDestroyExternalMemory(memory), "cuDestroyExternalMemory");

    pop_context();
}

} // namespace sgl::cuda::detail
