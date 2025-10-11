// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "platform.h"

#if SGL_LINUX

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/logger.h"

#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <signal.h>
#include <sys/types.h>
#include <pwd.h>
#include <unistd.h>
#include <linux/limits.h>

#include <regex>
#include <iostream>
#include <fstream>

namespace sgl::platform {

void static_init() { }

void static_shutdown() { }

void set_window_icon(WindowHandle handle, const std::filesystem::path& path)
{
    SGL_UNUSED(handle);
    SGL_UNUSED(path);
    SGL_UNIMPLEMENTED();
}

struct KeyboardInterruptData {
    std::mutex mutex;
    std::function<void()> handler;

    static KeyboardInterruptData& get()
    {
        static KeyboardInterruptData data;
        return data;
    }
};

static void signal_handler(int sig)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();

    if (sig == SIGINT) {
        std::lock_guard<std::mutex> lock(data.mutex);
        if (data.handler) {
            data.handler();
        }
    }
}

void set_keyboard_interrupt_handler(std::function<void()> handler)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();
    std::lock_guard<std::mutex> lock(data.mutex);

    if (handler && !data.handler) {
        struct sigaction action;
        action.sa_handler = signal_handler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        if (sigaction(SIGINT, &action, nullptr) != 0)
            SGL_THROW("Failed to register keyboard interrupt handler");
    } else if (!handler && data.handler) {
        struct sigaction action;
        action.sa_handler = SIG_DFL;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        if (sigaction(SIGINT, &action, nullptr) != 0)
            SGL_THROW("Failed to unregister keyboard interrupt handler");
    }
    data.handler = handler;
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

const std::filesystem::path& executable_path()
{
    static std::filesystem::path path(
        []()
        {
            char path_str[PATH_MAX] = {0};
            if (::readlink("/proc/self/exe", path_str, PATH_MAX) == -1)
                SGL_THROW("Failed to get the executable path.");
            return std::filesystem::path(path_str);
        }()
    );
    return path;
}

const std::filesystem::path& app_data_directory()
{
    static std::filesystem::path path([]() { return home_directory() / ".sgl"; }());
    return path;
}

const std::filesystem::path& home_directory()
{
    static std::filesystem::path path(
        []()
        {
            const char* path_str;
            if ((path_str = ::getenv("HOME")) == NULL)
                path_str = ::getpwuid(getuid())->pw_dir;
            return std::filesystem::path(path_str);
        }()
    );
    return path;
}

const std::filesystem::path& runtime_directory()
{
    static std::filesystem::path path(
        []()
        {
            Dl_info info;
            if (dladdr((void*)&runtime_directory, &info) == 0)
                SGL_THROW("Failed to get the runtime directory. dladdr() failed.");
            return std::filesystem::path(info.dli_fname).parent_path();
        }()
    );
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
// Memory
// -------------------------------------------------------------------------------------------------

size_t page_size()
{
    return sysconf(_SC_PAGESIZE);
}

MemoryStats memory_stats()
{
    MemoryStats stats = {};
    std::ifstream stat_file("/proc/self/stat");
    if (stat_file) {
        std::string line;
        std::getline(stat_file, line);
        std::istringstream iss(line);
        std::vector<std::string> tokens(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{});
        if (tokens.size() >= 24) {
            stats.rss = std::stoull(tokens[23]) * page_size();
            stats.peak_rss = std::stoull(tokens[22]);
            // stats.peak_rss = std::stoull(tokens[24]) * page_size();
        }
    }
    return stats;
}

// -------------------------------------------------------------------------------------------------
// Shared libraries
// -------------------------------------------------------------------------------------------------

SharedLibraryHandle load_shared_library(const std::filesystem::path& path)
{
    return dlopen(path.c_str(), RTLD_LAZY);
}

void release_shared_library(SharedLibraryHandle library)
{
    dlclose(library);
}

void* get_proc_address(SharedLibraryHandle library, const char* proc_name)
{
    return dlsym(library, proc_name);
}

// -------------------------------------------------------------------------------------------------
// Debugger
// -------------------------------------------------------------------------------------------------

bool is_debugger_present()
{
    std::ifstream status_file("/proc/self/status");
    std::string s;
    while (status_file >> s) {
        if (s == "TracerPid:") {
            int pid;
            status_file >> pid;
            return pid != 0;
        }
        std::getline(status_file, s);
    }
    return false;
}

void debug_break()
{
    raise(SIGTRAP);
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
    uintptr_t raw_trace[1024];
    int count = ::backtrace(reinterpret_cast<void**>(raw_trace), 1024);
    if (skip_frames >= count)
        return {};
    return StackTrace{raw_trace + skip_frames, raw_trace + skip_frames + (count - skip_frames)};
}

ResolvedStackTrace resolve_stacktrace(std::span<const StackFrame> trace)
{
    auto demangle = [](const char* name)
    {
        int status = 0;
        char* buffer = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        std::string demangled{buffer ? buffer : name};
        free(buffer);
        return demangled;
    };

    char** info = ::backtrace_symbols(reinterpret_cast<void* const*>(trace.data()), trace.size());

    std::regex re("(\\S+)\\((\\S*)\\+(0x[0-9a-f]*)\\)\\s+\\[(0x[0-9a-f]+)\\].*");

    ResolvedStackTrace resolved_trace(trace.size());
    for (size_t i = 0; i < trace.size(); i++) {
        ResolvedStackFrame& resolved = resolved_trace[i];
        resolved.address = trace[i];
        resolved.offset = 0ull;

        std::cmatch m;
        if (std::regex_match(info[i], m, re)) {
            resolved.module = m[1];
            resolved.symbol = m[2];
            resolved.symbol = demangle(resolved.symbol.c_str());
            resolved.offset = std::stoul(m[3], nullptr, 16);
        } else {
            resolved.symbol = info[i];
        }
    }

    free(info);

    return resolved_trace;
}

} // namespace sgl::platform

#endif // SGL_LINUX
