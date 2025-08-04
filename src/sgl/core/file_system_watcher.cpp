// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/file_system_watcher.h"

#include "sgl/core/error.h"

#include <map>

#if SGL_LINUX
#include <sys/inotify.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace sgl {

struct FileSystemWatchState {
#if SGL_LINUX
    int watch_descriptor;
#endif
#if !SGL_LINUX
    std::map<std::filesystem::path, std::filesystem::file_time_type> files;
#endif
    FileSystemWatcher* watcher;
    FileSystemWatchDesc desc;
};

#if !SGL_LINUX
static std::map<std::filesystem::path, std::filesystem::file_time_type>
get_directory_files(const std::filesystem::path& directory)
{
    if (!std::filesystem::exists(directory))
        return {};

    std::map<std::filesystem::path, std::filesystem::file_time_type> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::error_code ec;
            std::filesystem::path rel_path = std::filesystem::relative(entry.path(), directory, ec);
            if (ec) {
                log_warn("Failed to get relative path for file \"{}\"", entry.path());
                continue;
            }
            std::filesystem::file_time_type write_time = std::filesystem::last_write_time(entry.path(), ec);
            if (ec) {
                log_warn("Failed to get last write time for file \"{}\"", entry.path());
                continue;
            }
            files.emplace(rel_path, write_time);
        }
    }
    return files;
}
#endif

FileSystemWatcher::FileSystemWatcher()
{
#if SGL_LINUX
    m_inotify_file_descriptor = inotify_init();
    if (m_inotify_file_descriptor < 0) {
        SGL_THROW("Failed to initialize inotify file descriptor");
    }

    int flags = fcntl(m_inotify_file_descriptor, F_GETFL, 0);
    if (flags == -1) {
        close(m_inotify_file_descriptor);
        SGL_THROW("Failed to get inotify file descriptor flags");
    }

    flags |= O_NONBLOCK;
    if (fcntl(m_inotify_file_descriptor, F_SETFL, flags) == -1) {
        close(m_inotify_file_descriptor);
        SGL_THROW("Failed to set inotify file descriptor flags to non-blocking");
    }
#endif

#if !SGL_LINUX
    m_thread = std::thread([this]() { thread_func(); });
#endif
}

FileSystemWatcher::~FileSystemWatcher()
{
    for (const auto& pair : m_watches) {
        stop_watch(pair.second);
    }
#if SGL_LINUX
    close(m_inotify_file_descriptor);
#endif

#if !SGL_LINUX
    m_stop_thread = true;
    if (m_thread.joinable())
        m_thread.join();
#endif
}

uint32_t FileSystemWatcher::add_watch(const FileSystemWatchDesc& desc)
{
#if !SGL_LINUX
    std::lock_guard<std::mutex> lock(m_watches_mutex);
#endif

    // Check watch doesn't already exist
    for (const auto& pair : m_watches) {
        if (pair.second->desc.directory == desc.directory) {
            SGL_THROW("A watch already exists for {}.", desc.directory);
        }
    }

    // Check that directory exists.
    if (!std::filesystem::exists(desc.directory))
        SGL_THROW("Directory {} does not exist", desc.directory);

    // Init a new watcher state object
    uint32_t id = m_next_id++;
    auto state = std::make_unique<FileSystemWatchState>();
    state->desc = desc;
    state->watcher = this;

#if SGL_LINUX
    // On linux, add a watch to the inotify file descriptor.
    state->watch_descriptor = inotify_add_watch(
        m_inotify_file_descriptor,
        state->desc.directory.c_str(),
        IN_CREATE | IN_DELETE | IN_MODIFY
    );
    if (state->watch_descriptor < 0) {
        // This is bad, but issues with hot reload shouldn't stop a user being
        // able to render things. Currently tests start to struggle when creating
        // lots of monitors due to having lots of devices.
        log_error(
            "Failed to add watch for directory {}: {}. Hot reload may fail.",
            state->desc.directory,
            strerror(errno)
        );
        return 0;
    }
#endif

#if !SGL_LINUX
    state->files = get_directory_files(state->desc.directory);
#endif

    m_watches[id] = std::move(state);
    return id;
}

void FileSystemWatcher::remove_watch(uint32_t id)
{
    if (id > 0) {
#if !SGL_LINUX
        std::lock_guard<std::mutex> lock(m_watches_mutex);
#endif
        stop_watch(m_watches[id]);
        m_watches.erase(id);
    }
}

void FileSystemWatcher::remove_watch(const std::filesystem::path& directory)
{
#if !SGL_LINUX
    std::lock_guard<std::mutex> lock(m_watches_mutex);
#endif

    for (const auto& pair : m_watches) {
        if (pair.second->desc.directory == directory) {
            stop_watch(pair.second);
            m_watches.erase(pair.first);
            break;
        }
    }
}

void FileSystemWatcher::stop_watch(const std::unique_ptr<FileSystemWatchState>& state)
{
#if SGL_LINUX
    inotify_rm_watch(m_inotify_file_descriptor, state->watch_descriptor);
#endif
#if !SGL_LINUX
    SGL_UNUSED(state);
#endif
}

void FileSystemWatcher::set_on_change(ChangeCallback on_change)
{
    m_on_change = on_change;
}

void FileSystemWatcher::_notify_change(
    FileSystemWatchState* state,
    const std::filesystem::path& path,
    FileSystemWatcherChange change
)
{
    std::filesystem::path abs_path = std::filesystem::absolute(state->desc.directory / path);

    auto now = std::chrono::system_clock::now();
    FileSystemWatchEvent event{
        .path = path,
        .absolute_path = abs_path,
        .change = change,
        .time = now,
    };
    {
#if !SGL_LINUX
        std::lock_guard<std::mutex> lock(m_queued_events_mutex);
#endif
        m_queued_events.push_back(event);
    }
    m_last_event = now;
}

void FileSystemWatcher::update()
{
#if SGL_LINUX
    // Attempt non-blocking read from inotify, and return if no data.
    char buffer[4096];
    int length = read(m_inotify_file_descriptor, buffer, sizeof(buffer));
    if (length >= 0) {
        // Iterate over the inotify events and call '_notify_change' on the watcher for each one.
        int offset = 0;
        while (offset < length) {
            auto event = reinterpret_cast<inotify_event*>(buffer + offset);
            std::filesystem::path path{event->name};
            FileSystemWatcherChange change = FileSystemWatcherChange::invalid;

            if (event->mask & IN_CREATE) {
                change = FileSystemWatcherChange::added;
            } else if (event->mask & IN_DELETE) {
                change = FileSystemWatcherChange::removed;
            } else if (event->mask & IN_MODIFY) {
                change = FileSystemWatcherChange::modified;
            }

            if (change != FileSystemWatcherChange::invalid) {
                auto it = std::find_if(
                    m_watches.begin(),
                    m_watches.end(),
                    [event](const auto& pair) { return pair.second->watch_descriptor == event->wd; }
                );
                if (it != m_watches.end()) {
                    _notify_change(it->second.get(), path, change);
                }
            }
            offset += sizeof(struct inotify_event) + event->len;
        }
    } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EBADF) {
        SGL_THROW("Failed to read from inotify file descriptor");
    }
#endif

    // Process queued events.
    {
#if !SGL_LINUX
        std::lock_guard<std::mutex> lock(m_queued_events_mutex);
#endif
        if (m_queued_events.size() > 0) {
            auto duration = std::chrono::system_clock::now() - m_last_event;
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            if (millis > m_output_delay_ms) {
                m_on_change(m_queued_events);
                m_queued_events.clear();
            }
        }
    }
}

#if !SGL_LINUX
void FileSystemWatcher::thread_func()
{
    uint32_t counter = 0;
    while (!m_stop_thread) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (counter++ % 10 != 0)
            continue;

        std::lock_guard<std::mutex> lock(m_watches_mutex);
        for (const auto& [_, state] : m_watches) {
            std::map<std::filesystem::path, std::filesystem::file_time_type> files
                = get_directory_files(state->desc.directory);

            // Detect added and modified files.
            for (const auto& [path, write_time] : files) {
                auto it = state->files.find(path);
                if (it == state->files.end()) {
                    _notify_change(state.get(), path, FileSystemWatcherChange::added);
                } else if (it->second != write_time) {
                    _notify_change(state.get(), path, FileSystemWatcherChange::modified);
                }
            }
            // Detect removed files.
            for (const auto& [path, write_time] : state->files) {
                auto it = files.find(path);
                if (it == files.end()) {
                    _notify_change(state.get(), path, FileSystemWatcherChange::removed);
                }
            }

            state->files = std::move(files);
        }
    }
}
#endif

} // namespace sgl
