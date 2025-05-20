// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <string>
#include <vector>

#if SGL_WINDOWS
#define DECLARE_HANDLE(name)                                                                                           \
    struct name##__;                                                                                                   \
    typedef struct name##__* name
DECLARE_HANDLE(HWND);
#undef DECLARE_HANDLE
#endif

namespace sgl {
/// Native window handle.
struct WindowHandle {
#if SGL_WINDOWS
    HWND hwnd;
#elif SGL_LINUX
    void* xdisplay;
    uint32_t xwindow;
#elif SGL_MACOS
    void* nswindow;
#endif
};

/// Shared library handle.
using SharedLibraryHandle = void*;
} // namespace sgl

namespace sgl::platform {


/// Initialize the platform layer.
SGL_API void static_init();

/// Shutdown the platform layer.
SGL_API void static_shutdown();

/// Inform the platform that the library is loaded from python.
SGL_API void set_python_active(bool active);

/// Check if the library is loaded from python.
[[nodiscard]] SGL_API bool is_python_active();


/// Set the window icon.
SGL_API void set_window_icon(WindowHandle handle, const std::filesystem::path& path);

/// The pixel scale factor of the primary display.
[[nodiscard]] SGL_API float display_scale_factor();

/// Setup a callback function to be called when Ctrl-C is detected.
/// Use nullptr to remove handler.
SGL_API void set_keyboard_interrupt_handler(std::function<void()> handler);

// -------------------------------------------------------------------------------------------------
// File dialogs
// -------------------------------------------------------------------------------------------------

struct FileDialogFilter {
    /// Readable name (e.g. "JPEG").
    std::string name;
    /// File extension pattern (e.g. "*.jpg" or "*.jpg,*.jpeg").
    std::string pattern;

    FileDialogFilter() = default;
    FileDialogFilter(std::string_view name, std::string_view pattern)
        : name(name)
        , pattern(pattern)
    {
    }
    FileDialogFilter(std::pair<std::string_view, std::string_view> pair)
        : name(pair.first)
        , pattern(pair.second)
    {
    }
};
using FileDialogFilterList = std::vector<FileDialogFilter>;

/// Show a file open dialog.
/// \param filters List of file filters.
/// \return The selected file path or nothing if the dialog was cancelled.
[[nodiscard]] SGL_API std::optional<std::filesystem::path>
open_file_dialog(std::span<const FileDialogFilter> filters = {});

/// Show a file save dialog.
/// \param filters List of file filters.
/// \return The selected file path or nothing if the dialog was cancelled.
[[nodiscard]] SGL_API std::optional<std::filesystem::path>
save_file_dialog(std::span<const FileDialogFilter> filters = {});

/// Show a folder selection dialog.
/// \return The selected folder path or nothing if the dialog was cancelled.
[[nodiscard]] SGL_API std::optional<std::filesystem::path> choose_folder_dialog();

// -------------------------------------------------------------------------------------------------
// Filesystem
// -------------------------------------------------------------------------------------------------

/// Compares two paths in their weakly canonical form, returning true if they match.
/// Operator == on path does a string comparison, ignoring the fact that windows paths are case insensitive.
/// STL's equivalent comparator throws when either of the paths does not exist.
/// This complements the two by allowing comparing non-existent paths, but at the same time ignoring case on windows.
[[nodiscard]] SGL_API bool is_same_path(const std::filesystem::path& lhs, const std::filesystem::path& rhs);

/// Check if a file path has a given file extension. Does a case-insensitive comparison.
[[nodiscard]] SGL_API bool has_extension(const std::filesystem::path& path, const std::string_view ext);

/// Get the file extension from a path in lower-case and without the leading '.' character.
[[nodiscard]] SGL_API std::string get_extension_from_path(const std::filesystem::path& path);

/// Create a junction (soft link).
[[nodiscard]] SGL_API bool create_junction(const std::filesystem::path& link, const std::filesystem::path& target);

/// Delete a junction (sof link).
[[nodiscard]] SGL_API bool delete_junction(const std::filesystem::path& link);

// -------------------------------------------------------------------------------------------------
// System paths
// -------------------------------------------------------------------------------------------------

/// The full path to the current executable.
[[nodiscard]] SGL_API const std::filesystem::path& executable_path();

/// The current executable directory.
[[nodiscard]] SGL_API const std::filesystem::path& executable_directory();

/// The current executable name.
[[nodiscard]] SGL_API const std::string& executable_name();

/// The application data directory.
[[nodiscard]] SGL_API const std::filesystem::path& app_data_directory();

/// The home directory.
[[nodiscard]] SGL_API const std::filesystem::path& home_directory();

/// The project source directory. Note that this is only valid during development.
[[nodiscard]] SGL_API const std::filesystem::path& project_directory();

/// The runtime directory. This is the path where the sgl runtime library
/// (sgl.dll, libsgl.so or libsgl.dynlib) resides.
[[nodiscard]] SGL_API const std::filesystem::path& runtime_directory();

// -------------------------------------------------------------------------------------------------
// Environment
// -------------------------------------------------------------------------------------------------

/// Get the content of a system environment variable.
[[nodiscard]] SGL_API std::optional<std::string> get_environment_variable(const char* name);

// -------------------------------------------------------------------------------------------------
// Memory
// -------------------------------------------------------------------------------------------------

/// The page size of the system.
[[nodiscard]] SGL_API size_t page_size();

struct MemoryStats {
    /// Current resident/working set size in bytes.
    uint64_t rss;
    /// Peak resident/working set size in bytes.
    uint64_t peak_rss;
};

/// Get the current memory stats.
[[nodiscard]] SGL_API MemoryStats memory_stats();

// -------------------------------------------------------------------------------------------------
// Shared libraries
// -------------------------------------------------------------------------------------------------

/// Load a shared library.
[[nodiscard]] SGL_API SharedLibraryHandle load_shared_library(const std::filesystem::path& path);

/// Release a shared library.
SGL_API void release_shared_library(SharedLibraryHandle library);

/// Get a function pointer from a library.
SGL_API void* get_proc_address(SharedLibraryHandle library, const char* proc_name);

// -------------------------------------------------------------------------------------------------
// Debugger
// -------------------------------------------------------------------------------------------------

/// Check if a debugger session is attached.
[[nodiscard]] SGL_API bool is_debugger_present();

/// Breaks in debugger (int 3 functionality).
SGL_API void debug_break();

/// Print a message into the debug window.
SGL_API void print_to_debug_window(const char* str);

// -------------------------------------------------------------------------------------------------
// Stacktrace
// -------------------------------------------------------------------------------------------------

using StackFrame = uintptr_t;
using StackTrace = std::vector<StackFrame>;

struct ResolvedStackFrame {
    uintptr_t address;
    std::string module;
    std::string symbol;
    size_t offset;
    std::string source;
    uint32_t line;
};

using ResolvedStackTrace = std::vector<ResolvedStackFrame>;

/// Generate a backtrace.
[[nodiscard]] SGL_API StackTrace backtrace(size_t skip_frames = 1);

/// Resolve a stack trace with symbol information.
[[nodiscard]] SGL_API ResolvedStackTrace resolve_stacktrace(std::span<const StackFrame> trace);

/// Convert resolved stack trace to a human readable string.
[[nodiscard]] SGL_API std::string format_stacktrace(std::span<const ResolvedStackFrame> trace, size_t max_frames = 0);

/// Convert resolved stack trace to a human readable string.
[[nodiscard]] SGL_API std::string format_stacktrace(std::span<const StackFrame> trace, size_t max_frames = 0);


} // namespace sgl::platform
