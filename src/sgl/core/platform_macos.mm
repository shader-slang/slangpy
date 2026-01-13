// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "platform.h"

#if SGL_MACOS

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/logger.h"
#include "sgl/core/string.h"

#include <signal.h>
#include <limits.h>
#include <dlfcn.h>
#include <unistd.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <regex>
#include <iostream>
#include <fstream>

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <dispatch/dispatch.h>
#import <Availability.h>

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

static void sigint_handler(int)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();
    std::lock_guard<std::mutex> lock(data.mutex);

    if (data.handler)
        data.handler();
}

void set_keyboard_interrupt_handler(std::function<void()> handler)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();
    std::lock_guard<std::mutex> lock(data.mutex);

    if (handler) {
        struct sigaction action = {};
        action.sa_handler = sigint_handler;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, nullptr) != 0)
            SGL_THROW("Failed to register keyboard interrupt handler");
    } else {
        struct sigaction action = {};
        action.sa_handler = SIG_DFL;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, nullptr) != 0)
            SGL_THROW("Failed to unregister keyboard interrupt handler");
    }
    data.handler = handler;
}

// -------------------------------------------------------------------------------------------------
// File dialogs
// -------------------------------------------------------------------------------------------------

/*
 Parse file dialog filters like:
   "*.png;*.jpg" or "*.txt"

   - Split on ';' (Windows-style multi pattern separator)
   - Trim whitespace
   - Strip leading '*' and '.'
   - Lowercase result
   - Collect unique extensions (no dot)
*/
static std::vector<std::string> parse_extensions_from_filters(std::span<const FileDialogFilter> filters)
{
    std::vector<std::string> exts;
    for (const auto& filter : filters) {
        for (const auto& pattern : string::split(filter.pattern, ";")) {
            std::string ext = string::to_lower(string::remove_leading_whitespace(pattern, " \n\r\t.*"));
            if (std::find(exts.begin(), exts.end(), ext) == exts.end())
                exts.push_back(std::move(ext));
        }
    }
    return exts;
}

static void ensure_app_activation()
{
    if (NSApp == nil) {
        [NSApplication sharedApplication];
    }
    // Use accessory policy to avoid creating a full dock menu for library usage.
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    [NSApp activateIgnoringOtherApps:YES];
    NSRunningApplication* app = [NSRunningApplication currentApplication];
    [app activateWithOptions:NSApplicationActivateIgnoringOtherApps];
}

std::optional<std::filesystem::path> open_file_dialog(std::span<const FileDialogFilter> filters)
{
    __block std::optional<std::filesystem::path> resultPath;
    auto work = ^{
        @autoreleasepool {
            ensure_app_activation();
            NSOpenPanel* panel = [NSOpenPanel openPanel];
            [panel setCanChooseFiles:YES];
            [panel setCanChooseDirectories:NO];
            [panel setAllowsMultipleSelection:NO];

            auto exts = parse_extensions_from_filters(filters);
            if (!exts.empty()) {
                NSMutableArray<UTType*>* types = [NSMutableArray arrayWithCapacity:exts.size()];
                for (const auto& e : exts) {
                    NSString* ext = [NSString stringWithUTF8String:e.c_str()];
                    UTType* ut = [UTType typeWithFilenameExtension:ext];
                    if (ut)
                        [types addObject:ut];
                }
                [panel setAllowedContentTypes:types];
            }

            NSInteger response = [panel runModal];
            if (response == NSModalResponseOK) {
                NSURL* url = [panel URL];
                if (url) {
                    std::string path = [[url path] UTF8String];
                    resultPath = std::filesystem::path(path);
                }
            }
        }
    };
    if ([NSThread isMainThread]) {
        work();
    } else {
        dispatch_sync(dispatch_get_main_queue(), work);
    }
    return resultPath;
}

std::optional<std::filesystem::path> save_file_dialog(std::span<const FileDialogFilter> filters)
{
    __block std::optional<std::filesystem::path> resultPath;
    auto work = ^{
        @autoreleasepool {
            ensure_app_activation();
            NSSavePanel* panel = [NSSavePanel savePanel];

            auto exts = parse_extensions_from_filters(filters);
            if (!exts.empty()) {
                NSMutableArray<UTType*>* types = [NSMutableArray arrayWithCapacity:exts.size()];
                for (const auto& e : exts) {
                    NSString* ext = [NSString stringWithUTF8String:e.c_str()];
                    UTType* ut = [UTType typeWithFilenameExtension:ext];
                    if (ut)
                        [types addObject:ut];
                }
                [panel setAllowedContentTypes:types];
            }

            NSInteger response = [panel runModal];
            if (response == NSModalResponseOK) {
                NSURL* url = [panel URL];
                if (url) {
                    std::string path = [[url path] UTF8String];
                    resultPath = std::filesystem::path(path);
                }
            }
        }
    };
    if ([NSThread isMainThread]) {
        work();
    } else {
        dispatch_sync(dispatch_get_main_queue(), work);
    }
    return resultPath;
}

std::optional<std::filesystem::path> choose_folder_dialog()
{
    __block std::optional<std::filesystem::path> resultPath;
    auto work = ^{
        @autoreleasepool {
            ensure_app_activation();
            NSOpenPanel* panel = [NSOpenPanel openPanel];
            [panel setCanChooseFiles:NO];
            [panel setCanChooseDirectories:YES];
            [panel setAllowsMultipleSelection:NO];

            NSInteger response = [panel runModal];
            if (response == NSModalResponseOK) {
                NSURL* url = [panel URL];
                if (url) {
                    std::string path = [[url path] UTF8String];
                    resultPath = std::filesystem::path(path);
                }
            }
        }
    };
    if ([NSThread isMainThread]) {
        work();
    } else {
        dispatch_sync(dispatch_get_main_queue(), work);
    }
    return resultPath;
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
            uint32_t path_len = PATH_MAX;
            if (_NSGetExecutablePath(path_str, &path_len) == -1)
                SGL_THROW("Failed to get the executable path.");
            return std::filesystem::path(path_str);
        }()
    );
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
            const char* path_str;
            if ((path_str = ::getenv("HOME")) == NULL)
                SGL_THROW("Failed to get the home directory path.");
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
    return ::getpagesize();
}

MemoryStats memory_stats()
{
    MemoryStats stats = {};
    struct mach_task_basic_info info;
    mach_msg_type_number_t info_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &info_count) == KERN_SUCCESS) {
        stats.rss = info.resident_size;
        stats.peak_rss = info.resident_size_max;
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

    std::regex re("([0-9]+)\\s+(\\S+)\\s+(0x[0-9a-f]+)\\s+(\\S+)\\s\\+\\s([0-9]*)");

    ResolvedStackTrace resolved_trace(trace.size());
    for (size_t i = 0; i < trace.size(); i++) {
        ResolvedStackFrame& resolved = resolved_trace[i];
        resolved.address = trace[i];
        resolved.offset = 0ull;

        std::cmatch m;
        if (std::regex_match(info[i], m, re)) {
            resolved.address = std::stoull(m[3], nullptr, 16);
            resolved.module = m[2];
            resolved.symbol = m[4];
            resolved.symbol = demangle(resolved.symbol.c_str());
            resolved.offset = std::stoull(m[5], nullptr, 10);
        } else {
            resolved.symbol = info[i];
        }
    }

    free(info);

    return resolved_trace;
}

// -------------------------------------------------------------------------------------------------
// Crash handling
// -------------------------------------------------------------------------------------------------

static CrashHandlerCallback s_crash_handler_callback;

static void crash_signal_handler(int sig)
{
    if (s_crash_handler_callback) {
        CrashContext ctx;
        ctx.code = sig;
        ctx.stack_trace = backtrace();
        s_crash_handler_callback(ctx);
    }
    std::abort();
}

void set_crash_handler(CrashHandlerCallback callback)
{
    s_crash_handler_callback = callback;

    struct sigaction action = {};
    action.sa_handler = callback ? crash_signal_handler : SIG_DFL;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
    sigaction(SIGFPE, &action, nullptr);
}

} // namespace sgl::platform

#endif // SGL_MACOS
