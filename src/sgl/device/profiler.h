// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"

#include "sgl/core/enum.h"
#include "sgl/core/error.h"
#include "sgl/core/macros.h"
#include "sgl/core/object.h"

#include <atomic>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <memory>
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
struct ProfilerDesc {
    bool frame_stats_enabled{true};
    bool trace_enabled_on_start{false};
    uint32_t stats_window_size{120};
    uint32_t gpu_query_pool_size{64 * 1024};
    uint32_t gpu_query_block_size{256};
    bool auto_zones_enabled{true};
    bool debug_groups_enabled{false};
};

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

struct ProfilerTimelineRecord {
    uint32_t id{0};
    ProfilerTimelineType type{ProfilerTimelineType::cpu};
    std::string name;
    uint64_t thread_id{0};
    uint64_t device_id{0};
    CommandQueueType queue{CommandQueueType::graphics};
};

struct ProfilerSourceRecord {
    uint32_t id{0};
    std::string file;
    uint32_t line{0};
    std::string original_function;
    std::string display_function;
};

struct ProfilerNameRecord {
    uint32_t id{0};
    std::string name;
};

struct ProfilerFrameRecord {
    uint32_t id{0};
    uint32_t name_id{0};
    uint32_t source_id{0};
    uint64_t start_timestamp{0};
    uint64_t end_timestamp{0};
};

struct ProfilerZoneRecord {
    uint32_t id{0};
    uint32_t event_id{0};
    uint32_t child_index_begin{0};
    uint32_t child_index_count{0};
    uint32_t timeline_id{0};
    uint32_t frame_id{0};
    uint32_t source_id{0};
    uint32_t name_id{0};
    uint64_t start_timestamp{0};
    uint64_t end_timestamp{0};
};

class SGL_API ProfilerTrace : public Object {
    SGL_OBJECT(ProfilerTrace)
public:
    void write_to_json(const std::filesystem::path& path) const;

    const std::vector<ProfilerTimelineRecord>& timelines() const { return m_timelines; }
    const std::vector<ProfilerSourceRecord>& sources() const { return m_sources; }
    const std::vector<ProfilerNameRecord>& names() const { return m_names; }
    const std::vector<ProfilerFrameRecord>& frames() const { return m_frames; }
    const std::vector<ProfilerZoneRecord>& zones() const { return m_zones; }
    const std::vector<uint32_t>& child_indices() const { return m_child_indices; }
    const std::vector<uint32_t>& root_indices() const { return m_root_indices; }

private:
    std::vector<ProfilerTimelineRecord> m_timelines;
    std::vector<ProfilerSourceRecord> m_sources;
    std::vector<ProfilerNameRecord> m_names;
    std::vector<ProfilerFrameRecord> m_frames;
    std::vector<ProfilerZoneRecord> m_zones;
    std::vector<uint32_t> m_child_indices;
    std::vector<uint32_t> m_root_indices;

    friend class Profiler;
    friend struct ProfilerImpl;
};

struct ProfilerStatValue {
    bool valid{false};
    double last_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
    double average_ms{0.0};
    double stddev_ms{0.0};
    uint32_t sample_count{0};
};

struct ProfilerStatsNode {
    uint32_t id{0};
    uint32_t parent_id{0};
    uint32_t child_index_begin{0};
    uint32_t child_index_count{0};
    uint32_t source_id{0};
    uint32_t name_id{0};
    uint32_t pending_gpu_sample_count{0};
    ProfilerStatValue cpu;
    ProfilerStatValue gpu;
};

class SGL_API ProfilerStats : public Object {
    SGL_OBJECT(ProfilerStats)
public:
    const std::vector<ProfilerSourceRecord>& sources() const { return m_sources; }
    const std::vector<ProfilerNameRecord>& names() const { return m_names; }
    const std::vector<ProfilerStatsNode>& nodes() const { return m_nodes; }
    const std::vector<uint32_t>& child_indices() const { return m_child_indices; }
    uint32_t completed_frame_count() const { return m_completed_frame_count; }
    uint32_t window_size() const { return m_window_size; }

private:
    std::vector<ProfilerSourceRecord> m_sources;
    std::vector<ProfilerNameRecord> m_names;
    std::vector<ProfilerStatsNode> m_nodes;
    std::vector<uint32_t> m_child_indices;
    uint32_t m_completed_frame_count{0};
    uint32_t m_window_size{0};

    friend class Profiler;
    friend struct ProfilerImpl;
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

    /// Whether completed frames update rolling frame statistics.
    bool frame_stats_enabled() const { return m_frame_stats_enabled; }
    void set_frame_stats_enabled(bool enabled) { m_frame_stats_enabled = enabled; }

    uint32_t stats_window_size() const { return m_stats_window_size; }
    void set_stats_window_size(uint32_t size);

    const ProfilerDesc& desc() const { return m_desc; }

    void start_trace(bool clear = true);
    void stop_trace();
    void clear_trace();

    /// Get a snapshot of the current profiler trace data.
    /// The returned trace is a copy of the data recorded so far, and is not updated with future profiling events.
    ref<ProfilerTrace> trace_snapshot();
    ref<ProfilerStats> stats_snapshot();

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
    void flush();

    std::string to_string() const override;

private:
    ProfilerDesc m_desc;
    std::atomic<bool> m_enabled{true};
    std::atomic<bool> m_auto_zones_enabled{true};
    std::atomic<bool> m_debug_groups_enabled{false};
    std::atomic<bool> m_frame_stats_enabled{true};
    std::atomic<uint32_t> m_stats_window_size{120};
    std::atomic<uint32_t> m_next_event_id{1};
    std::atomic<uint32_t> m_next_frame_id{0};
    std::atomic<uint32_t> m_active_frame_id{std::numeric_limits<uint32_t>::max()};

    ProfilerImpl* m_impl;

    friend struct ProfilerImpl;
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
    class SGL_API ZoneGuard {
    public:
        explicit ZoneGuard(
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

        ~ZoneGuard() noexcept
        {
            if (m_profiler)
                m_profiler->end_zone(m_encoder, m_flags);
        }

    private:
        Profiler* m_profiler;
        CommandEncoder* m_encoder;
        ProfilerZoneFlags m_flags;

        SGL_NON_COPYABLE_AND_MOVABLE(ZoneGuard);
    };

    /// RAII helper for profiler frames on the current profiler.
    class SGL_API FrameGuard {
    public:
        explicit FrameGuard(const ProfilerSourceLocation* source_location, const char* name = nullptr) noexcept
        {
            Profiler* profiler = current_profiler_or_null();
            if (profiler && profiler->begin_frame(source_location, name))
                m_profiler = profiler;
        }

        ~FrameGuard() noexcept
        {
            if (m_profiler)
                m_profiler->end_frame();
        }

    private:
        Profiler* m_profiler{nullptr};

        SGL_NON_COPYABLE_AND_MOVABLE(FrameGuard);
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
    ::sgl::detail::ZoneGuard SGL_CONCAT_STRINGS(sgl_profiler_zone_, counter)(                                          \
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
    ::sgl::detail::FrameGuard SGL_CONCAT_STRINGS(sgl_profiler_frame_, counter)(                                        \
        &SGL_CONCAT_STRINGS(sgl_profiler_source_location_, counter),                                                   \
        ##__VA_ARGS__                                                                                                  \
    )
