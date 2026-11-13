// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "platform.h"
#include "sgl/core/config.h"
#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/glfw.h"

#if SGL_HAS_GLFW
#include <GLFW/glfw3.h>
#if SGL_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
#endif

#include <algorithm>
#include <cstdio>

namespace sgl::platform {

static bool s_is_python_active = false;

bool is_python_active()
{
    return s_is_python_active;
}

void set_python_active(bool active)
{
    s_is_python_active = active;
}

float display_scale_factor()
{
#if SGL_HAS_GLFW
    try {
        detail::ensure_glfw_initialized();
        float xscale = 1.f;
        float yscale = 1.f;
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        if (monitor)
            glfwGetMonitorContentScale(monitor, &xscale, &yscale);
        return 0.5f * (xscale + yscale);
    } catch (const std::exception&) {
        // Offscreen UI contexts also use this best-effort query.
    }
#endif
    return 1.f;
}

std::vector<MonitorInfo> get_monitors()
{
#if SGL_HAS_GLFW
    std::vector<MonitorInfo> result;
    detail::ensure_glfw_initialized();
    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    for (int i = 0; i < count; ++i) {
        GLFWmonitor* monitor = monitors[i];
        MonitorInfo info{};
        info.name = glfwGetMonitorName(monitor);
#if SGL_WINDOWS
        if (const char* identifier = glfwGetWin32Monitor(monitor))
            info.identifier = identifier;
#endif
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        SGL_CHECK(mode, "Failed to query monitor video mode");
        glfwGetMonitorPos(monitor, &info.bounds.x, &info.bounds.y);
        info.bounds.z = mode->width;
        info.bounds.w = mode->height;
        glfwGetMonitorWorkarea(monitor, &info.work_area.x, &info.work_area.y, &info.work_area.z, &info.work_area.w);
        glfwGetMonitorContentScale(monitor, &info.content_scale.x, &info.content_scale.y);
        info.primary = monitor == primary;
        result.push_back(std::move(info));
    }
    return result;
#else
    SGL_THROW("sgl was built without GLFW window support");
#endif
}

// -------------------------------------------------------------------------------------------------
// Filesystem
// -------------------------------------------------------------------------------------------------

bool is_same_path(const std::filesystem::path& lhs, const std::filesystem::path& rhs)
{
    return lhs.lexically_normal() == rhs.lexically_normal();
}

bool has_extension(const std::filesystem::path& path, std::string_view ext)
{
    // Remove leading '.' from ext.
    if (ext.size() > 0 && ext[0] == '.')
        ext.remove_prefix(1);

    std::string pathExt = get_extension_from_path(path);

    if (ext.size() != pathExt.size())
        return false;

    return std::equal(
        ext.rbegin(),
        ext.rend(),
        pathExt.rbegin(),
        [](char a, char b)
        {
            return std::tolower(a) == std::tolower(b);
        }
    );
}

std::string get_extension_from_path(const std::filesystem::path& path)
{
    std::string ext;
    if (path.has_extension()) {
        ext = path.extension().string();
        // Remove the leading '.' from ext.
        if (ext.size() > 0 && ext[0] == '.')
            ext.erase(0, 1);
        // Convert to lower-case.
        std::transform(
            ext.begin(),
            ext.end(),
            ext.begin(),
            [](char c)
            {
                return (char)std::tolower(c);
            }
        );
    }
    return ext;
}

const std::filesystem::path& executable_directory()
{
    static std::filesystem::path directory{executable_path().parent_path()};
    return directory;
}

const std::string& executable_name()
{
    static std::string name{executable_path().filename().string()};
    return name;
}

const std::filesystem::path& project_directory()
{
    static std::filesystem::path path = []()
    {
        std::filesystem::path path_ = std::filesystem::path{SGL_PROJECT_DIR}.lexically_normal();
        if (path_.empty())
            return runtime_directory();
        return path_;
    }();
    return path;
}

// -------------------------------------------------------------------------------------------------
// Stacktrace
// -------------------------------------------------------------------------------------------------

std::string format_stacktrace(std::span<const ResolvedStackFrame> trace, size_t max_frames)
{
    std::string result;
    for (size_t index = 0; const auto& item : trace) {
        if (item.source.empty()) {
            result += fmt::format("{:08x}: {}+{:#x} in {}\n", item.address, item.symbol, item.offset, item.module);
        } else {
            result += fmt::format(
                "{}({}): {}+{:#x} in {}\n",
                item.source,
                item.line,
                item.symbol,
                item.offset,
                item.module
            );
        }
        if (max_frames > 0 && ++index >= max_frames) {
            if (index < trace.size())
                result += fmt::format("... {} more\n", trace.size() - index);
            break;
        }
    }
    return result;
}

std::string format_stacktrace(std::span<const StackFrame> trace, size_t max_frames)
{
    return format_stacktrace(resolve_stacktrace(trace), max_frames);
}

} // namespace sgl::platform
