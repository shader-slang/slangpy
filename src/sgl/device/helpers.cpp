// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"

#include "sgl/device/device.h"
#include "sgl/device/debug_logger.h"

#include "sgl/core/config.h"
#include "sgl/core/macros.h"
#include "sgl/core/format.h"

#include <string>

#if SGL_HAS_D3D12
#include <dxgidebug.h>
#include <dxgi1_3.h>
#endif

namespace sgl::detail {

static const char* get_slang_result_name(SlangResult result)
{
    switch (result) {
    case SLANG_OK:
        return "SLANG_OK";
    case SLANG_FAIL:
        return "SLANG_FAIL";
    case SLANG_E_NOT_IMPLEMENTED:
        return "SLANG_E_NOT_IMPLEMENTED";
    case SLANG_E_NO_INTERFACE:
        return "SLANG_E_NO_INTERFACE";
    case SLANG_E_ABORT:
        return "SLANG_E_ABORT";
    case SLANG_E_INVALID_HANDLE:
        return "SLANG_E_INVALID_HANDLE";
    case SLANG_E_INVALID_ARG:
        return "SLANG_E_INVALID_ARG";
    case SLANG_E_OUT_OF_MEMORY:
        return "SLANG_E_OUT_OF_MEMORY";
    case SLANG_E_BUFFER_TOO_SMALL:
        return "SLANG_E_BUFFER_TOO_SMALL";
    case SLANG_E_UNINITIALIZED:
        return "SLANG_E_UNINITIALIZED";
    case SLANG_E_PENDING:
        return "SLANG_E_PENDING";
    case SLANG_E_CANNOT_OPEN:
        return "SLANG_E_CANNOT_OPEN";
    case SLANG_E_NOT_FOUND:
        return "SLANG_E_NOT_FOUND";
    case SLANG_E_INTERNAL_FAIL:
        return "SLANG_E_INTERNAL_FAIL";
    case SLANG_E_NOT_AVAILABLE:
        return "SLANG_E_NOT_AVAILABLE";
    case SLANG_E_TIME_OUT:
        return "SLANG_E_TIME_OUT";
    default:
        return "unknown";
    }
}

// Reads last error from graphics layer.
static std::string get_last_graphics_errors()
{
#if SGL_HAS_D3D12
    IDXGIDebug* dxgiDebug = nullptr;
    DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiDebug));
    if (!dxgiDebug)
        return "";

    IDXGIInfoQueue* dxgiInfoQueue = nullptr;
    dxgiDebug->QueryInterface(IID_PPV_ARGS(&dxgiInfoQueue));
    if (!dxgiInfoQueue)
        return "";

    UINT64 messageCount = dxgiInfoQueue->GetNumStoredMessages(DXGI_DEBUG_ALL);
    if (messageCount == 0)
        return "";

    SIZE_T messageLength = 0;
    dxgiInfoQueue->GetMessage(DXGI_DEBUG_ALL, messageCount - 1, nullptr, &messageLength);
    DXGI_INFO_QUEUE_MESSAGE* pMessage = (DXGI_INFO_QUEUE_MESSAGE*)malloc(messageLength);
    dxgiInfoQueue->GetMessage(DXGI_DEBUG_ALL, messageCount - 1, pMessage, &messageLength);
    auto res = std::string(pMessage->pDescription);
    free(pMessage);
    return res;
#else
    // TODO: Get useful error information for other platforms if possible
    return "";
#endif
}

/// Called when a slang call fails.
std::string build_slang_error_message(const char* call, SlangResult result)
{
    return fmt::format("{} failed with error: {} ({})\n", call, result, get_slang_result_name(result));
}

size_t get_slang_rhi_message_count(const Device* device)
{
    return device->debug_logger()->message_count();
}

/// Called when a slang-rhi call fails.
std::string
build_slang_rhi_error_message(const Device* device, const char* call, rhi::Result result, size_t before_message_count)
{
    size_t after_message_count = get_slang_rhi_message_count(device);
    auto msg = fmt::format("{} failed with error: {} ({})\n", call, result, get_slang_result_name(result));
    if (after_message_count > before_message_count) {
        msg += "\nRHI messages:\n";
        msg += device->debug_logger()->get_messages(before_message_count, after_message_count);
    }
    if (static_cast<uint32_t>(result) >= 0x80000000U) {
        std::string graphics_errors = get_last_graphics_errors();
        if (!graphics_errors.empty()) {
            msg += "\nLast graphics layer error:\n";
            msg += graphics_errors;
        }
    }
    return msg;
}

} // namespace sgl::detail
