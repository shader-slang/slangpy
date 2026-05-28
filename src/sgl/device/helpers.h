// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"

#include "sgl/core/error.h"

#include <slang-rhi.h>


namespace sgl::detail {
SGL_API std::string build_slang_error_message(const char* call, SlangResult result);
SGL_API size_t get_slang_rhi_message_count(const Device* device);
SGL_API std::string
build_slang_rhi_error_message(const Device* device, const char* call, rhi::Result result, size_t before_message_count);
} // namespace sgl::detail

#define SLANG_CALL(call)                                                                                               \
    {                                                                                                                  \
        ::SlangResult result_ = call;                                                                                  \
        if (SLANG_FAILED(result_)) {                                                                                   \
            SGL_THROW(::sgl::detail::build_slang_error_message(#call, result_));                                       \
        }                                                                                                              \
    }

#define SLANG_RHI_CALL(call, device)                                                                                   \
    {                                                                                                                  \
        size_t before_message_count_ = ::sgl::detail::get_slang_rhi_message_count(device);                             \
        ::rhi::Result result_ = call;                                                                                  \
        if (SLANG_FAILED(result_)) {                                                                                   \
            SGL_THROW(::sgl::detail::build_slang_rhi_error_message(device, #call, result_, before_message_count_));    \
        }                                                                                                              \
    }
