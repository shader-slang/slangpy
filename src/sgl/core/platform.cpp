// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "platform.h"
#include "sgl/core/error.h"
#include "sgl/core/format.h"

#include <GLFW/glfw3.h>

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
    float xscale = 1.f;
    float yscale = 1.f;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (monitor)
        glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    return 0.5f * (xscale + yscale);
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
        [](char a, char b) { return std::tolower(a) == std::tolower(b); }
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
        std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) { return (char)std::tolower(c); });
    }
    return ext;
}

std::filesystem::path get_temp_file_path()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    SGL_DIAGNOSTIC_PUSH
    SGL_DISABLE_MSVC_WARNING(4996)
    SGL_DISABLE_CLANG_WARNING("-Wdeprecated-declarations")
    char* name = std::tmpnam(nullptr);
    SGL_DIAGNOSTIC_POP
    if (name == nullptr)
        SGL_THROW("Failed to create temporary file path.");
    return name;
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
