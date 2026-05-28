// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/error.h"
#include "sgl/core/platform.h"

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace sgl {

/**
 * \brief Plugin manager for loading plugin libraries and creating plugin instances.
 *
 * The plugin system is based around the following principles:
 * - Plugins are compiled into shared libraries known as _plugin libraries_.
 * - A _plugin library_ registers one or more _plugin classes_ when being loaded.
 * - A _plugin class_ needs to inherit from a known _plugin base class_, implementing it's interface.
 * - A _plugin base class_ defines a `PluginInfo` struct, describing meta information about each
 *   _plugin class_ as well as a `PluginCreate` typedef used for instantiating plugin classes.
 *
 * A _plugin base class_ can be any class that is exported from the main sgl library.
 * It needs to define a `PluginInfo` struct as well as a `PluginCreate` typedef.
 * The SGL_PLUGIN_BASE_CLASS macro extends the class with the required members for the plugin system.
 * For example:
 *
 * \code
 * class SGL_API PluginBase
 * {
 * public:
 *     struct PluginInfo { const char* desc };
 *     using PluginCreate = (PluginBase *)(*)();
 *     SGL_PLUGIN_BASE_CLASS(PluginBase);
 *
 *     virtual void do_struff() = 0;
 * };
 * \endcode
 *
 * To implement a _plugin class_ type, we simply inherit from the base class.
 * The SGL_PLUGIN_CLASS macro extends the class with the required members for the plugin system.
 * The class also needs to implement a static create function, matching the `PluginCreate` type of the base class.
 * For example:
 *
 * \code
 * class PluginA : public PluginBase
 * {
 * public:
 *     SGL_PLUGIN_CLASS(PluginA, "PluginA", PluginInfo({"plugin description string"}));
 *
 *     static PluginBase* create() { return new PluginA(); }
 *
 *     void do_struff() override { ... }
 * };
 * \endcode
 *
 * The _plugin library_ must export a `register_plugin` function for registering all the plugin class types when called.
 *
 * \code
 * extern "C" SGL_API_EXPORT void register_plugin(sgl::PluginRegistry& registry)
 * {
 *     registry.register_class<PluginBase, PluginA>();
 * }
 * \endcode
 *
 * At runtime, the plugin manager can load plugin libraries using `load_plugin_by_name` and `load_plugin`.
 * New plugin instances can be created using `create_class`, for example:
 *
 * \code
 * PluginBase* p = PluginManager::instance().create_class<PluginBase>("PluginA");
 * \endcode
 *
 * The `get_infos` function returns a list of plugin infos for all loaded plugin types of a given plugin base class.
 */
class SGL_API PluginManager {
public:
    /// Singleton accessor.
    static PluginManager& instance();

    /**
     * \brief Create an instance of a plugin class.
     *
     * \tparam BaseT The plugin base class.
     * \tparam Args The argument type pack used for construction.
     * \param type The plugin type name.
     * \param args Additional arguments passed on construction.
     * \return Returns a new instance of the requested plugin type or nullptr if not registered.
     */
    template<typename BaseT, typename... Args>
    std::invoke_result_t<typename BaseT::PluginCreate, Args...> create_class(std::string_view type, Args... args) const
    {
        std::lock_guard<std::mutex> lock(m_class_descs_mutex);
        const ClassDesc<BaseT>* class_desc = find_class_desc<BaseT>(type);
        return class_desc ? class_desc->create(args...)
                          : std::invoke_result_t<typename BaseT::PluginCreate, Args...>{nullptr};
    }

    /**
     * \brief Check if a given type of a plugin is available.
     *
     * \tparam BaseT The plugin base class.
     * \param type The plugin type name.
     * \return True if plugin type is available.
     */
    template<typename BaseT>
    bool has_class(std::string_view type) const
    {
        std::lock_guard<std::mutex> lock(m_class_descs_mutex);
        const ClassDesc<BaseT>* class_desc = find_class_desc<BaseT>(type);
        return class_desc != nullptr;
    }

    /**
     * \brief Get infos for all registered plugin types for a given plugin base class.
     *
     * \tparam BaseT The plugin base class.
     * \return A list of infos.
     */
    template<typename BaseT>
    std::vector<std::pair<std::string, typename BaseT::PluginInfo>> get_infos() const
    {
        std::lock_guard<std::mutex> lock(m_class_descs_mutex);
        std::vector<std::pair<std::string, typename BaseT::PluginInfo>> result;

        for (const auto& [name, desc] : m_class_descs)
            if (auto match_desc = dynamic_cast<const ClassDesc<BaseT>*>(desc.get()))
                result.push_back(std::make_pair(match_desc->type, match_desc->info));

        return result;
    }

    /**
     * \brief Load a plugin library by name.
     * This will automatically determine the plugin path and file extension.
     * \param plugin_dir Path to plugin directory.
     * \param name Name of the plugin library.
     * \return True if successful.
     */
    bool load_plugin_by_name(const std::filesystem::path& plugin_dir, std::string_view name);

    /**
     * Load a list of plugin libraries.
     * \param plugin_dir Path to plugin directory.
     * \param names Names of the plugin libraries.
     */
    void load_plugins_by_name(const std::filesystem::path& plugin_dir, std::span<const std::string> names);

    /**
     * Load a plugin library.
     * \param path File path of the plugin library.
     * \return True if successful.
     */
    bool load_plugin(const std::filesystem::path& path);

    /**
     * Release a previously loaded plugin library.
     * \param path File path of the plugin library.
     * \return True if successful.
     */
    bool release_plugin(const std::filesystem::path& path);

    /**
     * Release all loaded plugin libraries.
     */
    void release_all_plugins();

private:
    struct ClassDescBase {
        ClassDescBase(SharedLibraryHandle library, std::string_view type)
            : library(library)
            , type(type)
        {
        }
        virtual ~ClassDescBase() { }

        SharedLibraryHandle library;
        std::string type;
    };

    template<typename BaseT>
    struct ClassDesc : public ClassDescBase {
        typename BaseT::PluginInfo info;
        typename BaseT::PluginCreate create;

        ClassDesc(
            SharedLibraryHandle library,
            std::string_view type,
            typename BaseT::PluginInfo info,
            typename BaseT::PluginCreate create
        )
            : ClassDescBase(library, type)
            , info(info)
            , create(create)
        {
        }
    };

    template<typename BaseT>
    void register_class(
        SharedLibraryHandle library,
        std::string_view type,
        typename BaseT::PluginInfo info,
        typename BaseT::PluginCreate create
    )
    {
        std::lock_guard<std::mutex> lock(m_class_descs_mutex);

        if (find_class_desc<BaseT>(type) != nullptr)
            SGL_THROW(
                "A plugin class with type name \"{}\" (base class type \"{}\") has already been registered.",
                type,
                BaseT::get_plugin_base_type()
            );

        auto desc = std::make_shared<ClassDesc<BaseT>>(library, type, info, create);
        m_class_descs.emplace(type, std::move(desc));
    }

    template<typename BaseT>
    const ClassDesc<BaseT>* find_class_desc(std::string_view type) const
    {
        if (auto it = m_class_descs.find(std::string(type)); it != m_class_descs.end())
            if (auto desc = dynamic_cast<const ClassDesc<BaseT>*>(it->second.get()); desc && desc->type == type)
                return desc;

        return nullptr;
    }

    std::map<std::filesystem::path, SharedLibraryHandle> m_libraries;
    std::map<std::string, std::shared_ptr<ClassDescBase>> m_class_descs;

    mutable std::mutex m_libraries_mutex;
    mutable std::mutex m_class_descs_mutex;

    friend class PluginRegistry;
};

/**
 * \brief Helper class passed to plugin libraries to register plugin classes.
 */
class PluginRegistry {
public:
    PluginRegistry(PluginManager& plugin_manager, SharedLibraryHandle library)
        : m_plugin_manager(plugin_manager)
        , m_library(library)
    {
    }
    PluginRegistry(const PluginRegistry&) = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;

    /**
     * \brief Register a plugin class.
     * Throws if a class with the same type name (and same base class) has already been registered.
     *
     * \tparam BaseT The plugin base class.
     * \param type The plugin type name.
     * \param info The plugin info (type of BaseT::PluginInfo).
     * \param create The plugin create function (type of BaseT::PluginCreate).
     */
    template<typename BaseT>
    void register_class(std::string_view type, typename BaseT::PluginInfo info, typename BaseT::PluginCreate create)
    {
        m_plugin_manager.register_class<BaseT>(m_library, type, info, create);
    }

    /**
     * \brief Register a plugin class.
     * This helper assumes that the plugin class has a `create` function matching the BaseT::PluginCreate type.
     *
     * \tparam BaseT The plugin base class.
     * \tparam T The plugin class.
     */
    template<typename BaseT, typename T>
    void register_class()
    {
        register_class<BaseT>(T::k_plugin_type, T::k_plugin_info, T::create);
    }

private:
    PluginManager& m_plugin_manager;
    SharedLibraryHandle m_library;
};

/**
 * \brief Macro for extending a class to be used as a plugin base class.
 *
 * This macro must be applied in the class declaration of the plugin base class.
 * It assumes that both a public PluginInfo struct type and a PluginCreate typedef
 * are defined when invoked.
 */
#define SGL_PLUGIN_BASE_CLASS(cls)                                                                                     \
public:                                                                                                                \
    static_assert(std::is_class_v<cls> == true);                                                                       \
    /* TODO: check for existance of cls::PluginCreate */                                                               \
    static_assert(std::is_class_v<cls::PluginInfo> == true);                                                           \
    static const std::string& get_plugin_base_type()                                                                   \
    {                                                                                                                  \
        static std::string type{#cls};                                                                                 \
        return type;                                                                                                   \
    }                                                                                                                  \
    virtual const std::string& get_plugin_type() const = 0;                                                            \
    virtual const PluginInfo& get_plugin_info() const = 0;

/**
 * \brief Macro for extending a class to be used as a plugin class.
 *
 * This macro must be applied in the class declaration of the plugin class.
 */
#define SGL_PLUGIN_CLASS(cls, type, info)                                                                              \
public:                                                                                                                \
    static_assert(std::is_class_v<cls> == true);                                                                       \
    /* TODO: check inheritance of base plugin class */                                                                 \
    static inline const std::string k_plugin_type{type};                                                               \
    static inline const PluginInfo k_plugin_info{info};                                                                \
    virtual const std::string& get_plugin_type() const final                                                           \
    {                                                                                                                  \
        return k_plugin_type;                                                                                          \
    }                                                                                                                  \
    virtual const PluginInfo& get_plugin_info() const final                                                            \
    {                                                                                                                  \
        return k_plugin_info;                                                                                          \
    }

} // namespace sgl
