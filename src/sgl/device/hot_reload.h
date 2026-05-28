// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"
#include "sgl/device/reflection.h"
#include "sgl/device/device_child.h"

#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"

#include <slang.h>

#include <exception>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace sgl {

/// Shader hot reload management, detects when relevant slang files
/// have been editor and triggers session recreates as necessary.
class SGL_API HotReload : public Object {
    SGL_OBJECT(HotReload)
public:
    HotReload(ref<Device> device);

    /// Force immediate recreation of all registered sessions and
    /// any modules/programs they've loaded/linked.
    void recreate_all_sessions();

    /// Updates internal file system monitor for change detection.
    void update();

    // Enable/disable auto rebuild in response to file system events.
    bool auto_detect_changes() const { return m_auto_detect_changes; }
    void set_auto_detect_changes(bool val) { m_auto_detect_changes = val; }

    // Adjust delay used by internal fs monitor before reponding to file system events.
    uint32_t auto_detect_delay() const;
    void set_auto_detect_delay(uint32_t delay_ms);

    /// Return true if last attempt to recreate sessions failed with exception.
    bool last_build_failed() const { return m_last_build_failed; }

    // Internal functions called from session constructor/destructor
    // to register sessions with hot reload system.
    void _register_slang_session(SlangSession* session);
    void _unregister_slang_session(SlangSession* session);

    // Called from session when modules have updated, meaning dependencies may have changed
    void _on_session_modules_changed(SlangSession* session);

    /// Exclusively for testing, erase all existing file watches
    void _clear_file_watches();
    void _reset_reloaded() { m_has_reloaded = false; }
    bool _has_reloaded() const { return m_has_reloaded; }

private:
    void on_file_system_event(std::span<FileSystemWatchEvent> events);
    void update_watched_paths_for_session(SlangSession* session);

    Device* m_device;
    bool m_auto_detect_changes{true};
    ref<FileSystemWatcher> m_file_system_watcher;
    std::set<SlangSession*> m_all_slang_sessions;
    bool m_last_build_failed{false};
    std::set<std::filesystem::path> m_watched_paths;
    bool m_has_reloaded;
};

} // namespace sgl
