// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "platform.h"

#if SGL_EMSCRIPTEN

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/logger.h"

#include <emscripten/emscripten.h>
#include <emscripten/heap.h>

#include <cstdlib>
#include <iostream>
#include <unistd.h>

namespace sgl::platform {

void static_init() { }

void static_shutdown() { }

void set_window_icon(WindowHandle handle, const std::filesystem::path& path)
{
    SGL_UNUSED(handle);
    SGL_UNUSED(path);
    SGL_UNIMPLEMENTED();
}

void set_keyboard_interrupt_handler(std::function<void()> handler)
{
    // There are no process signals in a browser.
    SGL_UNUSED(handler);
}

// -------------------------------------------------------------------------------------------------
// File dialogs
// -------------------------------------------------------------------------------------------------

std::optional<std::filesystem::path> open_file_dialog(std::span<const FileDialogFilter> filters)
{
    SGL_UNUSED(filters);
    SGL_UNIMPLEMENTED();
}

std::optional<std::filesystem::path> save_file_dialog(std::span<const FileDialogFilter> filters)
{
    SGL_UNUSED(filters);
    SGL_UNIMPLEMENTED();
}

std::optional<std::filesystem::path> choose_folder_dialog()
{
    SGL_UNIMPLEMENTED();
}

// -------------------------------------------------------------------------------------------------
// Filesystem
// -------------------------------------------------------------------------------------------------

bool create_junction(const std::filesystem::path& link, const std::filesystem::path& target)
{
    std::error_code ec;
    std::filesystem::create_directory_symlink(target, link, ec);
    if (ec)
        log_warn("Failed to create symlink {} to {}: {}", link, target, ec.message());
    return !ec;
}

bool delete_junction(const std::filesystem::path& link)
{
    std::error_code ec;
    std::filesystem::remove(link, ec);
    if (ec)
        log_warn("Failed to remove symlink {}: {}", link, ec.message());
    return !ec;
}

// -------------------------------------------------------------------------------------------------
// System paths
// -------------------------------------------------------------------------------------------------

// There is no executable or shared library file in a WebAssembly module. Paths that are derived
// from them resolve to the root of the virtual file system, so applications provide runtime files
// (such as the sgl shaders in "/shaders") by mounting or preloading them there.

const std::filesystem::path& executable_path()
{
    static std::filesystem::path path("/");
    return path;
}

const std::filesystem::path& app_data_directory()
{
    static std::filesystem::path path(
        []()
        {
            return home_directory() / ".sgl";
        }()
    );
    return path;
}

const std::filesystem::path& home_directory()
{
    static std::filesystem::path path(
        []()
        {
            const char* path_str = ::getenv("HOME");
            return std::filesystem::path(path_str ? path_str : "/");
        }()
    );
    return path;
}

const std::filesystem::path& runtime_directory()
{
    static std::filesystem::path path("/");
    return path;
}

// -------------------------------------------------------------------------------------------------
// Environment
// -------------------------------------------------------------------------------------------------

std::optional<std::string> get_environment_variable(const char* name)
{
    const char* value = ::getenv(name);
    return value != nullptr ? std::string(value) : std::optional<std::string>{};
}

// -------------------------------------------------------------------------------------------------
// Processes
// -------------------------------------------------------------------------------------------------

ProcessID current_process_id()
{
    return static_cast<ProcessID>(getpid());
}

// -------------------------------------------------------------------------------------------------
// Threads
// -------------------------------------------------------------------------------------------------

ThreadID current_thread_id()
{
    // Emscripten builds are single threaded.
    return 0;
}

bool set_current_thread_name(std::string_view name)
{
    SGL_UNUSED(name);
    return false;
}

// -------------------------------------------------------------------------------------------------
// Memory
// -------------------------------------------------------------------------------------------------

size_t page_size()
{
    return sysconf(_SC_PAGESIZE);
}

MemoryStats memory_stats()
{
    // WebAssembly memory only grows, so its current size is also the peak.
    size_t heap_size = emscripten_get_heap_size();
    return {.rss = heap_size, .peak_rss = heap_size};
}

// -------------------------------------------------------------------------------------------------
// Shared libraries
// -------------------------------------------------------------------------------------------------

SharedLibraryHandle load_shared_library(const std::filesystem::path& path)
{
    // Modules are linked statically.
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

// -------------------------------------------------------------------------------------------------
// Debugger
// -------------------------------------------------------------------------------------------------

bool is_debugger_present() noexcept
{
    return false;
}

void debug_break()
{
    emscripten_debugger();
}

void print_to_debug_window(const char* str)
{
    std::cerr << str;
}

// -------------------------------------------------------------------------------------------------
// Stacktrace
// -------------------------------------------------------------------------------------------------

StackTrace backtrace(size_t skip_frames)
{
    // Emscripten's public API only provides formatted JavaScript call stacks, not frame addresses.
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
