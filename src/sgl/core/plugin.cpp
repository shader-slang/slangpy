// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "plugin.h"
#include "sgl/core/logger.h"
#include "sgl/core/timer.h"

#include <chrono>

namespace sgl {

PluginManager& PluginManager::instance()
{
    static PluginManager sInstance;
    return sInstance;
}

bool PluginManager::load_plugin_by_name(const std::filesystem::path& plugin_dir, std::string_view name)
{
    auto path = plugin_dir / std::string(name);
#if SGL_WINDOWS
    path.replace_extension(".dll");
#elif SGL_LINUX
    path.replace_extension(".so");
#endif
    return load_plugin(path);
}

void PluginManager::load_plugins_by_name(const std::filesystem::path& plugin_dir, std::span<const std::string> names)
{
    Timer timer;

    size_t loaded_count = 0;
    for (const auto& name : names) {
        if (load_plugin_by_name(plugin_dir, name))
            loaded_count++;
    }

    if (loaded_count > 0)
        log_info("Loaded {} plugin(s) in {:.3}s", loaded_count, timer.elapsed_s());
}

bool PluginManager::load_plugin(const std::filesystem::path& path)
{
    // Early exit if plugin is already loaded.
    {
        std::lock_guard<std::mutex> lock(m_libraries_mutex);
        if (m_libraries.find(path) != m_libraries.end())
            return false;
    }

    if (!std::filesystem::exists(path))
        SGL_THROW("Failed to load plugin library from {}. File not found.", path);

    SharedLibraryHandle library = platform::load_shared_library(path);
    if (library == nullptr)
        SGL_THROW("Failed to load plugin library from {}. Cannot load shared library.", path);

    using RegisterPluginProc = void (*)(PluginRegistry&);

    auto register_plugin_proc = (RegisterPluginProc)platform::get_proc_address(library, "register_plugin");
    if (register_plugin_proc == nullptr)
        SGL_THROW("Failed to load plugin library from {}. Symbol 'register_plugin' not found.", path);

    // Register plugin library.
    {
        std::lock_guard<std::mutex> lock(m_libraries_mutex);
        m_libraries[path] = library;
    }

    // Call plugin library to register plugin classes.
    {
        PluginRegistry registry(*this, library);
        register_plugin_proc(registry);
    }

    return true;
}

bool PluginManager::release_plugin(const std::filesystem::path& path)
{
    std::lock_guard<std::mutex> libraries_lock(m_libraries_mutex);

    auto library_it = m_libraries.find(path);
    if (library_it == m_libraries.end()) {
        log_warn("Failed to release plugin library {}. The library isn't loaded.", path);
        return false;
    }

    // Get library handle.
    SharedLibraryHandle library = library_it->second;

    // Delete all the classes that were owned by the library.
    {
        std::lock_guard<std::mutex> class_descs_lock(m_class_descs_mutex);
        for (auto it = m_class_descs.begin(); it != m_class_descs.end();) {
            if (it->second->library == library)
                it = m_class_descs.erase(it);
            else
                ++it;
        }
    }

    platform::release_shared_library(library);
    m_libraries.erase(library_it);

    return true;
}

void PluginManager::release_all_plugins()
{
    while (true) {
        std::filesystem::path path;
        {
            std::lock_guard<std::mutex> lock(m_libraries_mutex);
            if (m_libraries.empty())
                return;
            path = m_libraries.begin()->first;
        }
        release_plugin(path);
    }
}

} // namespace sgl
