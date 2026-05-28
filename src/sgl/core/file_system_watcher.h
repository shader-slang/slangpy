// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/fwd.h"
#include "sgl/core/stream.h"
#include "sgl/core/enum.h"
#include "sgl/core/platform.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

namespace sgl {

/// Types of file system event that can be reported.
enum class FileSystemWatcherChange {
    invalid,
    added,
    removed,
    modified,
};

/// Init options for FileSystemWatcher.
struct FileSystemWatchDesc {

    /// Directory to monitor.
    std::filesystem::path directory;
};

/// Data reported on a given file system event to a file monitored
/// by FileSystemWatcher.
struct FileSystemWatchEvent {

    /// Path of file that has changed relative to watcher.
    std::filesystem::path path;

    /// Absolute path of file in file system
    std::filesystem::path absolute_path;

    /// Change type.
    FileSystemWatcherChange change{FileSystemWatcherChange::invalid};

    /// System time change was recorded.
    std::chrono::system_clock::time_point time;
};

// Declare watch state (only used internally).
struct FileSystemWatchState;

/// Monitors directories for changes and calls a callback when they're detected.
/// The watcher automatically queues up changes until disk has been idle for
/// a period. Relies on regular polling of update().
class SGL_API FileSystemWatcher : public Object {
    SGL_OBJECT(FileSystemWatcher)
public:
    using ChangeCallback = std::function<void(std::span<FileSystemWatchEvent>)>;

    FileSystemWatcher();
    ~FileSystemWatcher();

    /// FileSystemWatch move constructor is deleted to allow for map of unique ptrs.
    FileSystemWatcher(FileSystemWatcher&& other) = delete;

    /// Add watch of a new directory.
    uint32_t add_watch(const FileSystemWatchDesc& desc);

    /// Remove existing watch.
    void remove_watch(uint32_t id);

    /// Remove existing watch.
    void remove_watch(const std::filesystem::path& directory);

    /// Get number of active watches
    size_t watch_count() const { return m_watches.size(); }

    /// Get callback for file system events.
    ChangeCallback on_change() const { return m_on_change; }

    /// Set callback for file system events.
    void set_on_change(ChangeCallback on_change);

    /// Update function to poll the watcher + report events.
    void update();

    /// Delay period before queued events are output.
    uint32_t delay() { return m_output_delay_ms; }

    /// Set delay period before queued events are output.
    void set_delay(uint32_t milliseconds) { m_output_delay_ms = milliseconds; }

    /// Internal function called when OS reports an event.
    void _notify_change(FileSystemWatchState* state, const std::filesystem::path& path, FileSystemWatcherChange change);

private:
    /// Releases OS monitoring for a given watch.
    void stop_watch(const std::unique_ptr<FileSystemWatchState>&);

    /// Next unique id to be assigned to a given watch.
    uint32_t m_next_id{1};

    /// Delay before outputting events in milliseconds.
    uint32_t m_output_delay_ms{1000};

    /// Map of id->watch.
    std::map<int, std::unique_ptr<FileSystemWatchState>> m_watches;

    /// Watch event callback.
    ChangeCallback m_on_change;

    /// Events reported since last call to watch event callback.
    std::vector<FileSystemWatchEvent> m_queued_events;

    /// Time last event was recorded.
    std::chrono::system_clock::time_point m_last_event;

#if SGL_LINUX
    /// File descriptor for linux inotify watcher.
    int m_inotify_file_descriptor;
#endif

#if !SGL_LINUX
    /// Mutex to protect the watch map.
    std::mutex m_watches_mutex;

    /// Mutex to protect the queued events.
    std::mutex m_queued_events_mutex;

    /// Thread to poll for file system changes.
    std::thread m_thread;
    std::atomic<bool> m_stop_thread{false};
    void thread_func();
#endif
};


} // namespace sgl
