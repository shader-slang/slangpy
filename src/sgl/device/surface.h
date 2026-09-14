// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/formats.h"
#include "sgl/device/resource.h"

#include "sgl/core/macros.h"
#include "sgl/core/platform.h"
#include "sgl/core/object.h"

#include <slang-rhi.h>

namespace sgl {

class Window;

struct SurfaceInfo {
    /// Preferred format for the surface.
    Format preferred_format;
    /// Supported texture usages.
    TextureUsage supported_usage;
    /// Supported texture formats.
    std::vector<Format> formats;
};

struct SurfaceConfig {
    /// Surface texture format.
    Format format = Format::undefined;
    /// Surface texture usage.
    TextureUsage usage = TextureUsage::none;
    /// Surface texture width.
    uint32_t width = 0;
    /// Surface texture height.
    uint32_t height = 0;
    /// Desired number of images.
    uint32_t desired_image_count = 3;
    /// Enable/disable vertical synchronization.
    bool vsync = true;
};

class SGL_API Surface : public Object {
    SGL_OBJECT(Surface)
public:
    Surface(WindowHandle window_handle, ref<Device> device);
    Surface(Window* window, ref<Device> device);
    ~Surface();

    /// Returns the surface info.
    const SurfaceInfo& info() const { return m_info; };

    /// Returns the surface config.
    const std::optional<SurfaceConfig>& config() const { return m_config; };

    /// Configure the surface.
    void configure(const SurfaceConfig& config);

    /// Unconfigure the surface.
    void unconfigure();

    /// Acquries the next surface image.
    ref<Texture> acquire_next_image();

    /// Present the previously acquire image.
    void present();

    /// Non-throwing variant of acquire_next_image() for the render loop's
    /// resize-recovery path: a recoverable swapchain invalidation must not
    /// terminate the app, and SGL_THROW would additionally log-fatal and
    /// debug-break. Returns the RHI result; sets \p out_texture on success.
    rhi::Result try_acquire_next_image(ref<Texture>& out_texture);

    /// Non-throwing variant of present() for the render loop's resize-recovery
    /// path (see try_acquire_next_image). Returns the RHI result.
    rhi::Result try_present();

private:
    void get_images();

    SurfaceInfo m_info;
    std::optional<SurfaceConfig> m_config;
    ref<Device> m_device;
    Slang::ComPtr<rhi::ISurface> m_rhi_surface;
};
} // namespace sgl
