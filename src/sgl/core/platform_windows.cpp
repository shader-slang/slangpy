// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "platform.h"

#if SGL_WINDOWS

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/string.h"

#ifndef WINDOWS_LEAN_AND_MEAN
#define WINDOWS_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <CommCtrl.h>
#include <commdlg.h>
#include <comutil.h>
#include <Psapi.h>
#include <ShellScalingApi.h>
#include <ShlObj_core.h>
#include <winioctl.h>
#include <DbgHelp.h>

#include <mutex>

#define WINDOWS_CALL(a)                                                                                                \
    {                                                                                                                  \
        auto result_ = a;                                                                                              \
        if (FAILED(result_))                                                                                           \
            SGL_THROW("Windows API call failed: {} (result={})", #a, int(result_));                                    \
    }

namespace sgl::platform {

static bool s_initialized;

void static_init()
{
    WINDOWS_CALL(CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE));
    s_initialized = true;
}

void static_shutdown()
{
    s_initialized = false;
    CoUninitialize();
}

void set_window_icon(WindowHandle handle, const std::filesystem::path& path)
{
    HANDLE hicon = LoadImageW(GetModuleHandleW(NULL), path.c_str(), IMAGE_ICON, 0, 0, LR_DEFAULTSIZE | LR_LOADFROMFILE);
    if (!hicon)
        SGL_THROW("Failed to load icon from \"{}\".", path);
    HWND hwnd = handle.hwnd ? handle.hwnd : GetActiveWindow();
    SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hicon);
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

static BOOL WINAPI console_ctrl_handler(_In_ DWORD dwCtrlType)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();

    if (dwCtrlType == CTRL_C_EVENT) {
        std::lock_guard<std::mutex> lock(data.mutex);
        if (data.handler) {
            data.handler();
            return TRUE;
        }
    }

    return FALSE;
}

void set_keyboard_interrupt_handler(std::function<void()> handler)
{
    KeyboardInterruptData& data = KeyboardInterruptData::get();
    std::lock_guard<std::mutex> lock(data.mutex);

    if (handler && !data.handler) {
        if (SetConsoleCtrlHandler(console_ctrl_handler, TRUE) == 0)
            SGL_THROW("Failed to register keyboard interrupt handler");
    } else if (!handler && data.handler) {
        if (SetConsoleCtrlHandler(console_ctrl_handler, FALSE) == 0)
            SGL_THROW("Failed to unregister keyboard interrupt handler");
    }
    data.handler = handler;
}

// -------------------------------------------------------------------------------------------------
// File dialogs
// -------------------------------------------------------------------------------------------------

struct FilterSpec {
    FilterSpec(std::span<const FileDialogFilter> filters, bool for_open)
    {
        if (filters.empty()) {
            com_dlg.push_back({.pszName = L"All files", .pszSpec = L"*.*"});
            return;
        }

        names.reserve(filters.size() + 1);
        specs.reserve(filters.size() + 1);

        if (for_open)
            com_dlg.push_back({});

        std::wstring all;
        for (const auto& f : filters) {
            names.push_back(string::to_wstring(f.name));
            specs.push_back(string::to_wstring(f.pattern));
            for (auto& c : specs.back())
                if (c == L',')
                    c = L';';
            com_dlg.push_back({.pszName = names.back().c_str(), .pszSpec = specs.back().c_str()});
            if (!all.empty())
                all += L';';
            all += specs.back();
        }

        if (for_open) {
            names.push_back(L"Supported files");
            specs.push_back(all);
            com_dlg[0] = {names.back().c_str(), specs.back().c_str()};
        }
    }

    size_t size() const { return com_dlg.size(); }
    const COMDLG_FILTERSPEC* data() const { return com_dlg.data(); }

private:
    std::vector<COMDLG_FILTERSPEC> com_dlg;
    std::vector<std::wstring> names;
    std::vector<std::wstring> specs;
};

template<typename DialogType>
static std::optional<std::filesystem::path>
file_dialog_common(std::span<const FileDialogFilter> filters, DWORD options, const CLSID clsid)
{
    SGL_CHECK(s_initialized, "Platform not initialized.");

    FilterSpec fs(filters, typeid(DialogType) == typeid(IFileOpenDialog));

    DialogType* dialog;
    WINDOWS_CALL(CoCreateInstance(clsid, NULL, CLSCTX_ALL, IID_PPV_ARGS(&dialog)));
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM);
    dialog->SetFileTypes((uint32_t)fs.size(), fs.data());
    dialog->SetDefaultExtension(fs.data()->pszSpec);

    if (dialog->Show(NULL) == S_OK) {
        IShellItem* pItem;
        if (dialog->GetResult(&pItem) == S_OK) {
            PWSTR path_str;
            if (pItem->GetDisplayName(SIGDN_FILESYSPATH, &path_str) == S_OK) {
                std::filesystem::path path = path_str;
                CoTaskMemFree(path_str);
                return path;
            }
        }
    }

    return {};
}

std::optional<std::filesystem::path> open_file_dialog(std::span<const FileDialogFilter> filters)
{
    return file_dialog_common<IFileOpenDialog>(filters, FOS_FILEMUSTEXIST, CLSID_FileOpenDialog);
}

std::optional<std::filesystem::path> save_file_dialog(std::span<const FileDialogFilter> filters)
{
    return file_dialog_common<IFileSaveDialog>(filters, FOS_OVERWRITEPROMPT, CLSID_FileSaveDialog);
}

std::optional<std::filesystem::path> choose_folder_dialog()
{
    return file_dialog_common<IFileOpenDialog>({}, FOS_PICKFOLDERS | FOS_PATHMUSTEXIST, CLSID_FileOpenDialog);
}

// -------------------------------------------------------------------------------------------------
// Filesystem
// -------------------------------------------------------------------------------------------------

// As defined in ntifs.h
typedef struct _REPARSE_DATA_BUFFER {
    ULONG ReparseTag;
    USHORT ReparseDataLength;
    USHORT Reserved;
    union {
        struct {
            USHORT SubstituteNameOffset;
            USHORT SubstituteNameLength;
            USHORT PrintNameOffset;
            USHORT PrintNameLength;
            ULONG Flags;
            WCHAR PathBuffer[1];
        } SymbolicLinkReparseBuffer;
        struct {
            USHORT SubstituteNameOffset;
            USHORT SubstituteNameLength;
            USHORT PrintNameOffset;
            USHORT PrintNameLength;
            WCHAR PathBuffer[1];
        } MountPointReparseBuffer;
        struct {
            UCHAR DataBuffer[1];
        } GenericReparseBuffer;
    } DUMMYUNIONNAME;
} REPARSE_DATA_BUFFER;

// As defined in wdm.h
#define FILE_DIRECTORY_FILE 0x00000001
#define FILE_WRITE_THROUGH 0x00000002
#define FILE_SEQUENTIAL_ONLY 0x00000004
#define FILE_NO_INTERMEDIATE_BUFFERING 0x00000008

#define FILE_SYNCHRONOUS_IO_ALERT 0x00000010
#define FILE_SYNCHRONOUS_IO_NONALERT 0x00000020
#define FILE_NON_DIRECTORY_FILE 0x00000040
#define FILE_CREATE_TREE_CONNECTION 0x00000080

#define FILE_COMPLETE_IF_OPLOCKED 0x00000100
#define FILE_NO_EA_KNOWLEDGE 0x00000200
#define FILE_OPEN_REMOTE_INSTANCE 0x00000400
#define FILE_RANDOM_ACCESS 0x00000800

#define FILE_DELETE_ON_CLOSE 0x00001000
#define FILE_OPEN_BY_FILE_ID 0x00002000
#define FILE_OPEN_FOR_BACKUP_INTENT 0x00004000
#define FILE_NO_COMPRESSION 0x00008000

bool create_junction(const std::filesystem::path& link, const std::filesystem::path& target)
{
    // Get absolute target.
    WCHAR absolute_target[MAX_PATH];
    if (GetFullPathNameW(target.native().c_str(), MAX_PATH, absolute_target, NULL) == 0)
        return false;

    // Create junction as normal directory.
    if (!CreateDirectoryW(link.native().c_str(), NULL))
        return false;

    // Open file handle to junction directory.
    HANDLE handle = CreateFileW(
        link.native().c_str(),
        GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        NULL
    );
    if (handle == INVALID_HANDLE_VALUE)
        return false;

    // Allocate reparse data.
    char tmp[MAXIMUM_REPARSE_DATA_BUFFER_SIZE] = {0};
    REPARSE_DATA_BUFFER& reparse_data = *reinterpret_cast<REPARSE_DATA_BUFFER*>(tmp);

    // Setup reparse data.
    std::wstring substituteName = std::wstring(L"\\??\\") + absolute_target;
    std::wstring printName = absolute_target;

    reparse_data.ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
    reparse_data.Reserved = 0;
    reparse_data.MountPointReparseBuffer.SubstituteNameOffset = (USHORT)0;
    reparse_data.MountPointReparseBuffer.SubstituteNameLength = (USHORT)(substituteName.length() * sizeof(wchar_t));
    reparse_data.MountPointReparseBuffer.PrintNameOffset = (USHORT)((substituteName.length() + 1) * sizeof(wchar_t));
    reparse_data.MountPointReparseBuffer.PrintNameLength = (USHORT)(printName.length() * sizeof(wchar_t));
    reparse_data.ReparseDataLength = USHORT(
        sizeof(reparse_data.MountPointReparseBuffer) + reparse_data.MountPointReparseBuffer.PrintNameOffset
        + reparse_data.MountPointReparseBuffer.PrintNameLength
    );
    if (reparse_data.ReparseDataLength > MAXIMUM_REPARSE_DATA_BUFFER_SIZE)
        return false;

    std::memcpy(
        &reparse_data.MountPointReparseBuffer.PathBuffer[0],
        substituteName.data(),
        substituteName.length() * sizeof(wchar_t)
    );
    std::memcpy(
        &reparse_data.MountPointReparseBuffer.PathBuffer[substituteName.length() + 1],
        printName.data(),
        printName.length() * sizeof(wchar_t)
    );

    // Set the total size of the data buffer.

    // Use DeviceIoControl to set the reparse point data on the file handle.
    size_t header_size = FIELD_OFFSET(REPARSE_DATA_BUFFER, MountPointReparseBuffer);
    bool success = DeviceIoControl(
        handle,
        FSCTL_SET_REPARSE_POINT,
        &reparse_data,
        (DWORD)header_size + reparse_data.ReparseDataLength,
        NULL,
        0,
        NULL,
        0
    );

    // Close file handle.
    CloseHandle(handle);

    return success;
}

bool delete_junction(const std::filesystem::path& link)
{
    // Open file handle to junction directory.
    HANDLE handle = CreateFileW(
        link.native().c_str(),
        GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        NULL
    );
    if (handle == INVALID_HANDLE_VALUE)
        return false;

    REPARSE_GUID_DATA_BUFFER rgdb = {0};
    rgdb.ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;

    // Delete reparse point data.
    bool success = DeviceIoControl(
        handle,
        FSCTL_DELETE_REPARSE_POINT,
        &rgdb,
        REPARSE_GUID_DATA_BUFFER_HEADER_SIZE,
        NULL,
        0,
        NULL,
        0
    );

    // Close file handle.
    CloseHandle(handle);

    return success;
}

// -------------------------------------------------------------------------------------------------
// System paths
// -------------------------------------------------------------------------------------------------

const std::filesystem::path& executable_path()
{
    static std::filesystem::path path(
        []()
        {
            CHAR path_str[1024];
            if (GetModuleFileNameA(nullptr, path_str, ARRAYSIZE(path_str)) == 0)
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
            PWSTR path_str;
            if (!SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &path_str)))
                SGL_THROW("Failed to get the application data directory.");
            return std::filesystem::path(path_str);
        }()
    );
    return path;
}

const std::filesystem::path& home_directory()
{
    static std::filesystem::path path(
        []()
        {
            if (auto value = get_environment_variable("USERPROFILE"))
                return std::filesystem::path(*value);
            return std::filesystem::path();
        }()
    );
    return path;
}

const std::filesystem::path& runtime_directory()
{
    static std::filesystem::path path(
        []()
        {
            HMODULE hm = NULL;
            if (GetModuleHandleExA(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    (LPCSTR)&runtime_directory,
                    &hm
                )
                == 0)
                SGL_THROW("Failed to get the runtime directory. GetModuleHandle failed: {}", GetLastError());
            CHAR path_str[1024];
            if (GetModuleFileNameA(hm, path_str, sizeof(path_str)) == 0)
                SGL_THROW("Failed to get the runtime directory. GetModuleFileNameA failed: {}", GetLastError());
            return std::filesystem::path(path_str).parent_path();
        }()
    );
    return path;
}

// -------------------------------------------------------------------------------------------------
// Environment
// -------------------------------------------------------------------------------------------------

std::optional<std::string> get_environment_variable(const char* name)
{
    static char buf[4096];
    DWORD len = GetEnvironmentVariableA(name, buf, (DWORD)std::size(buf));
    SGL_ASSERT(len < (DWORD)std::size(buf));
    return len > 0 ? std::string(buf) : std::optional<std::string>{};
}

// -------------------------------------------------------------------------------------------------
// Processes
// -------------------------------------------------------------------------------------------------

ProcessID current_process_id()
{
    return static_cast<ProcessID>(GetCurrentProcessId());
}

// -------------------------------------------------------------------------------------------------
// Memory
// -------------------------------------------------------------------------------------------------

size_t page_size()
{
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwAllocationGranularity;
}

MemoryStats memory_stats()
{
    MemoryStats stats = {};
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(PROCESS_MEMORY_COUNTERS))) {
        stats.rss = pmc.WorkingSetSize;
        stats.peak_rss = pmc.PeakWorkingSetSize;
    }
    return stats;
}

// -------------------------------------------------------------------------------------------------
// Shared libraries
// -------------------------------------------------------------------------------------------------

SharedLibraryHandle load_shared_library(const std::filesystem::path& path)
{
    /// Comments taken from slang-platform
    // We try to search the DLL in two different attempts.
    // First attempt - LoadLibraryExW()
    // If it failed to find one, we will use LoadLibraryW() to search over all PATH.
    // Search order: 1) The directory that contains the DLL (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR).
    //                  This directory is searched only for dependencies of the DLL being loaded.
    //               2) Application directory
    //               3) User directories (AddDllDirectory/SetDllDirectory)
    //               4) System32
    //               5) PATH environment variable (by the 2nd attempt with LoadLibraryW())
    // https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexw
    // https://docs.microsoft.com/en-us/windows/desktop/api/libloaderapi/nf-libloaderapi-loadlibraryw
    SharedLibraryHandle handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!handle)
        handle = LoadLibraryW(path.c_str());

    return handle;
}

void release_shared_library(SharedLibraryHandle library)
{
    FreeLibrary(static_cast<HMODULE>(library));
}

void* get_proc_address(SharedLibraryHandle library, const char* proc_name)
{
    return GetProcAddress(static_cast<HMODULE>(library), proc_name);
}

// -------------------------------------------------------------------------------------------------
// Debugger
// -------------------------------------------------------------------------------------------------

bool is_debugger_present()
{
    return ::IsDebuggerPresent() == TRUE;
}

void debug_break()
{
    __debugbreak();
}

void print_to_debug_window(const char* str)
{
    OutputDebugStringA(str);
}

// -------------------------------------------------------------------------------------------------
// Stacktrace
// -------------------------------------------------------------------------------------------------

StackTrace backtrace(size_t skip_frames)
{
    StackFrame stack[1024];
    size_t frame_count = CaptureStackBackTrace(DWORD(skip_frames), 1024, reinterpret_cast<PVOID*>(stack), nullptr);
    return {stack, stack + frame_count};
}

ResolvedStackTrace resolve_stacktrace(std::span<const StackFrame> trace)
{
    HANDLE process = GetCurrentProcess();
    SymInitialize(process, nullptr, true);

    struct Symbol : SYMBOL_INFO {
        char name_storage[1023];
    } symbol{};
    symbol.MaxNameLen = 1024;
    symbol.SizeOfStruct = sizeof(SYMBOL_INFO);

    IMAGEHLP_MODULE64 module{};
    module.SizeOfStruct = sizeof(IMAGEHLP_MODULE64);

    IMAGEHLP_LINE line{};
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE);

    ResolvedStackTrace resolved_trace(trace.size());
    for (size_t i = 0; i < trace.size(); i++) {
        ResolvedStackFrame& resolved = resolved_trace[i];
        resolved.address = trace[i];
        resolved.offset = 0ull;
        if (SymFromAddr(process, resolved.address, &resolved.offset, &symbol))
            resolved.symbol = symbol.Name;
        if (SymGetModuleInfo64(process, symbol.ModBase, &module))
            resolved.module = module.ModuleName;
        DWORD displacement;
        if (SymGetLineFromAddr64(process, resolved.address, &displacement, &line)) {
            resolved.source = line.FileName ? line.FileName : "";
            resolved.line = line.LineNumber;
        }
    }

    return resolved_trace;
}

} // namespace sgl::platform

#endif // SGL_WINDOWS
