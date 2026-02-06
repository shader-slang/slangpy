// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "platform.h"

#if SGL_EMSCRIPTEN

#include "sgl/core/error.h"
#include <iostream>

namespace sgl::platform {

void static_init() { }

void static_shutdown() { }

void set_window_icon(WindowHandle handle, const std::filesystem::path& path)
{
    SGL_UNUSED(handle);
    SGL_UNUSED(path);
}

void set_keyboard_interrupt_handler(std::function<void()> handler)
{
    SGL_UNUSED(handler);
}

std::optional<std::filesystem::path> open_file_dialog(std::span<const FileDialogFilter> filters)
{
    SGL_UNUSED(filters);
    return {};
}

std::optional<std::filesystem::path> save_file_dialog(std::span<const FileDialogFilter> filters)
{
    SGL_UNUSED(filters);
    return {};
}

std::optional<std::filesystem::path> choose_folder_dialog()
{
    return {};
}

bool create_junction(const std::filesystem::path& link, const std::filesystem::path& target)
{
    SGL_UNUSED(link);
    SGL_UNUSED(target);
    return false;
}

bool delete_junction(const std::filesystem::path& link)
{
    SGL_UNUSED(link);
    return false;
}

const std::filesystem::path& executable_path()
{
    static std::filesystem::path path("/");
    return path;
}

const std::filesystem::path& app_data_directory()
{
    static std::filesystem::path path("/");
    return path;
}

const std::filesystem::path& home_directory()
{
    static std::filesystem::path path("/");
    return path;
}

const std::filesystem::path& runtime_directory()
{
    static std::filesystem::path path("/");
    return path;
}

std::optional<std::string> get_environment_variable(const char* name)
{
    SGL_UNUSED(name);
    return std::nullopt;
}

ProcessID current_process_id()
{
    return 0;
}

size_t page_size()
{
    return 4096;
}

MemoryStats memory_stats()
{
    return {0, 0};
}

SharedLibraryHandle load_shared_library(const std::filesystem::path& path)
{
    SGL_UNUSED(path);
    return nullptr;
}

void release_shared_library(SharedLibraryHandle library)
{
    SGL_UNUSED(library);
}

void* get_proc_address(SharedLibraryHandle library, const char* proc_name)
{
    SGL_UNUSED(library);
    SGL_UNUSED(proc_name);
    return nullptr;
}

bool is_debugger_present()
{
    return false;
}

void debug_break() { }

void print_to_debug_window(const char* str)
{
    std::cerr << str;
}

StackTrace backtrace(size_t skip_frames)
{
    SGL_UNUSED(skip_frames);
    return {};
}

ResolvedStackTrace resolve_stacktrace(std::span<const StackFrame> trace)
{
    SGL_UNUSED(trace);
    return {};
}

} // namespace sgl::platform

#endif // SGL_EMSCRIPTEN
