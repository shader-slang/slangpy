// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "renderdoc.h"

#include "sgl/core/error.h"
#include "sgl/core/platform.h"
#include "sgl/core/window.h"

#include "sgl/device/device.h"
#include "sgl/device/native_handle_traits.h"

#include "renderdoc_app.h"

#include <vector>

namespace sgl::renderdoc {

class API {
public:
    API()
    {
#if SGL_WINDOWS
        const char* library_name = "renderdoc.dll";
#elif SGL_LINUX
        const char* library_name = "librenderdoc.so";
#else
        const char* library_name = nullptr;
#endif
        if (library_name) {
            m_library = platform::load_shared_library(library_name);
            if (m_library) {
                pRENDERDOC_GetAPI RENDERDOC_GetAPI
                    = (pRENDERDOC_GetAPI)platform::get_proc_address(m_library, "RENDERDOC_GetAPI");
                if (RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&m_api) != 1) {
                    platform::release_shared_library(m_library);
                    m_library = nullptr;
                    m_api = nullptr;
                }
            }
            if (!m_api)
                log_warn("RenderDoc API is not available.");
        } else {
            log_warn("RenderDoc is not available on this platform.");
        }
    }

    ~API()
    {
        if (m_library)
            platform::release_shared_library(m_library);
    }

    bool is_available() const { return m_api != nullptr; }

    bool start_frame_capture(ref<Device> device, ref<Window> window)
    {
        SGL_CHECK_NOT_NULL(device);

        if (!is_available())
            return false;

        if (m_frame_capture) {
            log_warn("RenderDoc frame capture is already in progress.");
            return false;
        }

        RENDERDOC_DevicePointer device_handle;
        RENDERDOC_WindowHandle window_handle;

        switch (device->type()) {
#if SGL_HAS_D3D12
        case DeviceType::d3d12:
            device_handle = device->native_handles()[0].as<ID3D12Device*>();
            break;
#endif
#if SGL_HAS_VULKAN
        case DeviceType::vulkan:
            device_handle = device->native_handles()[0].as<VkInstance>();
            break;
#endif
        default:
            log_warn("RenderDoc is not supported on device type {}.", device->type());
            return false;
        }

#if SGL_WINDOWS
        window_handle = window ? window->window_handle().hwnd : nullptr;
#elif SGL_LINUX
        window_handle = window ? (void*)uintptr_t(window->window_handle().xwindow) : nullptr;
#else
        return false;
#endif

        m_frame_capture.reset(new FrameCapture{
            .device = device,
            .window = window,
            .device_handle = device_handle,
            .window_handle = window_handle,
        });

        m_api->StartFrameCapture(m_frame_capture->device_handle, m_frame_capture->window_handle);

        return true;
    }

    bool end_frame_capture()
    {
        if (!is_available())
            return false;

        if (!m_frame_capture) {
            log_warn("Frame capture is not in progress.");
            return false;
        }

        m_api->EndFrameCapture(m_frame_capture->device_handle, m_frame_capture->window_handle);
        m_frame_capture.reset();

        return true;
    }

    bool is_frame_capturing() const
    {
        if (!is_available())
            return false;

        return m_api->IsFrameCapturing();
    }

    static API& get()
    {
        static API api;
        return api;
    }

private:
    SharedLibraryHandle m_library{nullptr};
    RENDERDOC_API_1_6_0* m_api{nullptr};

    struct FrameCapture {
        ref<Device> device;
        ref<Window> window;
        RENDERDOC_DevicePointer device_handle{nullptr};
        RENDERDOC_WindowHandle window_handle{nullptr};
    };

    std::unique_ptr<FrameCapture> m_frame_capture;
};

bool is_available()
{
    return API::get().is_available();
}

bool start_frame_capture(ref<Device> device, ref<Window> window)
{
    return API::get().start_frame_capture(device, window);
}

bool end_frame_capture()
{
    return API::get().end_frame_capture();
}

bool is_frame_capturing()
{
    return API::get().is_frame_capturing();
}

} // namespace sgl::renderdoc
