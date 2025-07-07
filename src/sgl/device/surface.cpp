// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "surface.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"
#include "sgl/device/helpers.h"

#include "sgl/core/error.h"
#include "sgl/core/window.h"

namespace sgl {

Surface::Surface(WindowHandle window_handle, ref<Device> device)
    : m_device(std::move(device))
{
    SGL_ASSERT(m_device);

#if SGL_WINDOWS
    rhi::WindowHandle rhi_window_handle = rhi::WindowHandle::fromHwnd(window_handle.hwnd);
#elif SGL_LINUX
    rhi::WindowHandle rhi_window_handle
        = rhi::WindowHandle::fromXlibWindow(window_handle.xdisplay, window_handle.xwindow);
#elif SGL_MACOS
    rhi::WindowHandle rhi_window_handle = rhi::WindowHandle::fromNSWindow(window_handle.nswindow);
#endif

    SLANG_RHI_CALL(m_device->rhi_device()->createSurface(rhi_window_handle, m_rhi_surface.writeRef()));

    const rhi::SurfaceInfo& rhi_info = m_rhi_surface->getInfo();
    m_info.preferred_format = static_cast<Format>(rhi_info.preferredFormat);
    m_info.supported_usage = static_cast<TextureUsage>(rhi_info.supportedUsage);
    m_info.formats.resize(rhi_info.formatCount);
    for (size_t i = 0; i < rhi_info.formatCount; ++i)
        m_info.formats[i] = static_cast<Format>(rhi_info.formats[i]);
}

Surface::Surface(Window* window, ref<Device> device)
    : Surface(window->window_handle(), std::move(device))
{
}

Surface::~Surface() { }

void Surface::configure(const SurfaceConfig& config)
{
    rhi::SurfaceConfig rhi_config{
        .format = static_cast<rhi::Format>(config.format),
        .usage = static_cast<rhi::TextureUsage>(config.usage),
        .width = config.width,
        .height = config.height,
        .desiredImageCount = config.desired_image_count,
        .vsync = config.vsync,
    };

    SLANG_RHI_CALL(m_rhi_surface->configure(rhi_config));

    rhi_config = *m_rhi_surface->getConfig();
    m_config = {
        .format = static_cast<Format>(rhi_config.format),
        .usage = static_cast<TextureUsage>(rhi_config.usage),
        .width = rhi_config.width,
        .height = rhi_config.height,
        .desired_image_count = rhi_config.desiredImageCount,
        .vsync = rhi_config.vsync,
    };
}

void Surface::unconfigure()
{
    SLANG_RHI_CALL(m_rhi_surface->unconfigure());

    m_config.reset();
}

ref<Texture> Surface::acquire_next_image()
{
    Slang::ComPtr<rhi::ITexture> texture;
    SLANG_RHI_CALL(m_rhi_surface->acquireNextImage(texture.writeRef()));
    rhi::TextureDesc texture_desc = texture->getDesc();
    return m_device->create_texture_from_resource(
        {
            .type = TextureType::texture_2d,
            .format = static_cast<Format>(texture_desc.format),
            .width = narrow_cast<uint32_t>(texture_desc.size.width),
            .height = narrow_cast<uint32_t>(texture_desc.size.height),
            .mip_count = texture_desc.mipCount,
            .usage = static_cast<TextureUsage>(texture_desc.usage),
        },
        texture
    );
}

void Surface::present()
{
    SLANG_RHI_CALL(m_rhi_surface->present());
}

} // namespace sgl
