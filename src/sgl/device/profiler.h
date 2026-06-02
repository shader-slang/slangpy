// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"

#include "sgl/core/enum.h"
#include "sgl/core/error.h"
#include "sgl/core/macros.h"
#include "sgl/core/object.h"

#include <atomic>
#include <string>
#include <string_view>
#include <memory>
#include <span>
#include <vector>

namespace sgl {

enum class ProfilerZoneFlags : uint32_t {
    none = 0,
    debug_group = 1 << 0,
    copy_name = 1 << 1,
};
SGL_ENUM_CLASS_OPERATORS(ProfilerZoneFlags);
SGL_ENUM_FLAGS_INFO(
    ProfilerZoneFlags,
    {
        {ProfilerZoneFlags::none, "none"},
        {ProfilerZoneFlags::debug_group, "debug_group"},
        {ProfilerZoneFlags::copy_name, "copy_name"},
    }
);
SGL_ENUM_REGISTER(ProfilerZoneFlags);

enum class ProfilerTimelineType : uint32_t {
    cpu,
    gpu,
};
SGL_ENUM_INFO(
    ProfilerTimelineType,
    {
        {ProfilerTimelineType::cpu, "cpu"},
        {ProfilerTimelineType::gpu, "gpu"},
    }
);
SGL_ENUM_REGISTER(ProfilerTimelineType);

/// Descriptor for creating a Profiler.
struct ProfilerDesc { };

/// Stable metadata for a profiler source callsite.
struct ProfilerSourceLocation {
    const char* file;
    uint32_t line;
    const char* function;
};

/// Metadata for a profiler timeline/lane.
struct ProfilerTimelineInfo {
    ProfilerTimelineType type{ProfilerTimelineType::cpu};
    std::string name;
    uint64_t thread_id{0};
    uint64_t device_id{0};
    CommandQueueType queue{CommandQueueType::graphics};
};

struct ProfilerZone {
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    const ProfilerSourceLocation* source_location;
    const char* name;
    const ProfilerZone* parent;
    std::span<const ProfilerZone*> children;
};

struct ProfilerFrame {
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    const ProfilerSourceLocation* source_location;
    const char* name;
    uint32_t frame_id;
};

class ProfilerTraceStorage;

class SGL_API ProfilerTrace : public Object {
    SGL_OBJECT(ProfilerTrace)
public:
    struct Timeline {
        ProfilerTimelineInfo info;
        std::vector<const ProfilerZone*> zones;
    };

    void write_to_json(const std::filesystem::path& path) const;

private:
    std::shared_ptr<ProfilerTraceStorage> m_trace_storage;
    std::vector<Timeline> m_timelines;

    friend class Profiler;
};

struct ProfilerImpl;

/// Hierarchical CPU/GPU application profiler.
///
/// A profiler becomes the application-wide current profiler when constructed and removes itself from the current
/// stack when destroyed.
class SGL_API Profiler : public Object {
    SGL_OBJECT(Profiler)
public:
    explicit Profiler(ProfilerDesc desc = {});
    ~Profiler();

    /// Intern or reuse a dynamic source location.
    static const ProfilerSourceLocation*
    intern_source_location(std::string_view file, uint32_t line, std::string_view function);

    /// Intern or reuse a dynamic profiler zone name.
    static const char* intern_name(std::string_view name);

    /// Whether profiling is enabled. When disabled, recording calls are no-ops.
    bool enabled() const { return m_enabled; }

    /// Enable or disable profiling.
    void set_enabled(bool enabled) { m_enabled = enabled; }

    /// Whether SlangPy functional calls automatically insert profiler zones.
    bool auto_zones_enabled() const { return m_auto_zones_enabled; }

    /// Enable or disable automatic SlangPy functional call zones.
    void set_auto_zones_enabled(bool enabled) { m_auto_zones_enabled = enabled; }

    /// Whether auto GPU zones also emit command debug groups.
    bool debug_groups_enabled() const { return m_debug_groups_enabled; }

    void set_debug_groups_enabled(bool enabled) { m_debug_groups_enabled = enabled; }

    const ProfilerDesc& desc() const { return m_desc; }

    /// Get a snapshot of the current profiler trace data.
    /// The returned trace is a copy of the data recorded so far, and is not updated with future profiling events.
    ref<ProfilerTrace> trace_snapshot();

    bool begin_zone(
        const ProfilerSourceLocation* source_location,
        const char* name,
        CommandEncoder* encoder,
        ProfilerZoneFlags flags
    ) noexcept;
    void end_zone(CommandEncoder* encoder, ProfilerZoneFlags flags) noexcept;

    bool begin_frame(const ProfilerSourceLocation* source_location, const char* name) noexcept;
    void end_frame() noexcept;

    void tick();

    std::string to_string() const override;

private:
    ProfilerDesc m_desc;
    std::atomic<bool> m_enabled{true};
    std::atomic<bool> m_auto_zones_enabled{true};
    std::atomic<bool> m_debug_groups_enabled{false};

    ProfilerImpl* m_impl;

    friend class ProfilerZoneScope;
};

// ---------------------------------------------------------------------------
// Application-wide current profiler stack.
// ---------------------------------------------------------------------------

// TODO:
// The profiler stack stores raw pointers so live profiler construction can automatically push without keeping
// profilers alive forever. Destruction removes all stack entries for that profiler, but this does not prevent a
// racing thread from observing a profiler pointer while the profiler is being destroyed. Stronger lifetime
// guarantees for live switching/destruction require an activation model with retained ownership.

/// Push a profiler onto the application-wide current profiler stack.
/// \param profiler Profiler to push (must not be null).
SGL_API void push_current_profiler(Profiler* profiler);

/// Pop the top profiler from the application-wide current profiler stack.
/// Throws if the stack is empty.
/// \return The popped profiler.
SGL_API Profiler* pop_current_profiler();

/// Get the current profiler from the top of the application-wide profiler stack.
/// Throws if the stack is empty.
/// \return The current profiler.
SGL_API Profiler* current_profiler();

/// Return the application-wide current profiler, or null.
SGL_API Profiler* current_profiler_or_null();

/// RAII helper that pushes a profiler as application-wide current profiler.
class SGL_API ProfilerScope {
public:
    explicit ProfilerScope(Profiler* profiler)
        : m_profiler(profiler)
    {
        push_current_profiler(profiler);
    }

    ~ProfilerScope()
    {
        if (m_active) {
            SGL_ASSERT(current_profiler() == m_profiler);
            pop_current_profiler();
        }
    }

    // Non-copyable.
    ProfilerScope(const ProfilerScope&) = delete;
    ProfilerScope& operator=(const ProfilerScope&) = delete;

    // Movable.
    ProfilerScope(ProfilerScope&& other) noexcept
        : m_profiler(other.m_profiler)
        , m_active(other.m_active)
    {
        other.m_profiler = nullptr;
        other.m_active = false;
    }

    ProfilerScope& operator=(ProfilerScope&& other) noexcept
    {
        if (this != &other) {
            if (m_active) {
                SGL_ASSERT(current_profiler() == m_profiler);
                pop_current_profiler();
            }
            m_profiler = other.m_profiler;
            m_active = other.m_active;
            other.m_profiler = nullptr;
            other.m_active = false;
        }
        return *this;
    }

private:
    Profiler* m_profiler{nullptr};
    bool m_active{true};
};

namespace detail {

    /// RAII helper for profiling zones on the current profiler.
    class SGL_API ProfilerZoneScope {
    public:
        explicit ProfilerZoneScope(
            const ProfilerSourceLocation* source_location,
            const char* name = nullptr,
            CommandEncoder* encoder = nullptr,
            ProfilerZoneFlags flags = ProfilerZoneFlags::none
        ) noexcept
            : m_profiler(nullptr)
            , m_encoder(encoder)
            , m_flags(flags)
        {
            Profiler* profiler = current_profiler_or_null();
            if (profiler && profiler->begin_zone(source_location, name, encoder, flags)) {
                m_profiler = profiler;
            }
        }

        ~ProfilerZoneScope() noexcept
        {
            if (m_profiler)
                m_profiler->end_zone(m_encoder, m_flags);
        }

    private:
        Profiler* m_profiler;
        CommandEncoder* m_encoder;
        ProfilerZoneFlags m_flags;

        SGL_NON_COPYABLE_AND_MOVABLE(ProfilerZoneScope);
    };

    /// RAII helper for profiler frames on the current profiler.
    class SGL_API ProfilerFrameScope {
    public:
        explicit ProfilerFrameScope(const ProfilerSourceLocation* source_location, const char* name = nullptr) noexcept
        {
            Profiler* profiler = current_profiler_or_null();
            if (profiler && profiler->begin_frame(source_location, name))
                m_profiler = profiler;
        }

        ~ProfilerFrameScope() noexcept
        {
            if (m_profiler)
                m_profiler->end_frame();
        }

    private:
        Profiler* m_profiler{nullptr};

        SGL_NON_COPYABLE_AND_MOVABLE(ProfilerFrameScope);
    };

} // namespace detail
} // namespace sgl

#if SGL_MSVC
#define SGL_PRETTY_FUNC __FUNCSIG__
#elif SGL_GCC || SGL_CLANG
#define SGL_PRETTY_FUNC __PRETTY_FUNCTION__
#else
#define SGL_PRETTY_FUNC __func__
#endif


/// Start a profiler zone for the current C++ scope.
///
/// The macro always records the callsite source location (`__FILE__`, `__LINE__`, and `__func__`).
/// Optional arguments are forwarded in fixed order:
/// - `SGL_PROFILER_ZONE()`
/// - `SGL_PROFILER_ZONE(name)`
/// - `SGL_PROFILER_ZONE(name, encoder)`
/// - `SGL_PROFILER_ZONE(name, encoder, flags)`
///
/// `name` must point to stable storage, such as a string literal or a pointer returned by
/// `Profiler::intern_name()`, unless `flags` includes `ProfilerZoneFlags::copy_name`.
/// Use explicit `nullptr` placeholders to pass later arguments, e.g.
/// `SGL_PROFILER_ZONE(nullptr, nullptr, flags)`.
///
/// The implementation uses the common `, ##__VA_ARGS__` extension so empty macro arguments work with
/// the MSVC/GCC-style preprocessors used by this project.
#define SGL_PROFILER_ZONE(...) SGL_PROFILER_ZONE_IMPL(__COUNTER__, __VA_ARGS__)
#define SGL_PROFILER_ZONE_IMPL(counter, ...) SGL_PROFILER_ZONE_IMPL2(counter, __VA_ARGS__)
#define SGL_PROFILER_ZONE_IMPL2(counter, ...)                                                                          \
    static const ::sgl::ProfilerSourceLocation SGL_CONCAT_STRINGS(sgl_profiler_source_location_, counter)              \
        = {__FILE__, __LINE__, SGL_PRETTY_FUNC};                                                                       \
    ::sgl::detail::ProfilerZoneScope SGL_CONCAT_STRINGS(sgl_profiler_zone_, counter)(                                  \
        &SGL_CONCAT_STRINGS(sgl_profiler_source_location_, counter),                                                   \
        ##__VA_ARGS__                                                                                                  \
    )

/// Start a profiler frame for the current C++ scope.
///
/// The macro records the callsite source location and optionally accepts a stable frame name:
/// - `SGL_PROFILER_FRAME()`
/// - `SGL_PROFILER_FRAME(name)`
///
/// Like zone names, `name` must point to stable storage.
#define SGL_PROFILER_FRAME(...) SGL_PROFILER_FRAME_IMPL(__COUNTER__, __VA_ARGS__)
#define SGL_PROFILER_FRAME_IMPL(counter, ...) SGL_PROFILER_FRAME_IMPL2(counter, __VA_ARGS__)
#define SGL_PROFILER_FRAME_IMPL2(counter, ...)                                                                         \
    static const ::sgl::ProfilerSourceLocation SGL_CONCAT_STRINGS(sgl_profiler_source_location_, counter)              \
        = {__FILE__, __LINE__, SGL_PRETTY_FUNC};                                                                       \
    ::sgl::detail::ProfilerFrameScope SGL_CONCAT_STRINGS(sgl_profiler_frame_, counter)(                                \
        &SGL_CONCAT_STRINGS(sgl_profiler_source_location_, counter),                                                   \
        ##__VA_ARGS__                                                                                                  \
    )
