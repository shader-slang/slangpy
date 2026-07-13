// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/enum.h"
#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/fwd.h"
#include "sgl/device/types.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sgl {

/// Type of execution timeline stored in a profiler trace.
enum class ProfilerTimelineType : uint32_t {
    cpu, ///< CPU thread timeline.
    gpu, ///< GPU device queue timeline.
};
SGL_ENUM_INFO(
    ProfilerTimelineType,
    {
        {ProfilerTimelineType::cpu, "cpu"},
        {ProfilerTimelineType::gpu, "gpu"},
    }
);
SGL_ENUM_REGISTER(ProfilerTimelineType);

/// Reason why a bounded profiler capture stopped recording.
enum class ProfilerCaptureStopReason : uint32_t {
    user,        ///< The capture was stopped explicitly by the user.
    memory_limit ///< The capture reached ProfilerCaptureDesc::max_memory_bytes.
};
SGL_ENUM_INFO(
    ProfilerCaptureStopReason,
    {
        {ProfilerCaptureStopReason::user, "user"},
        {ProfilerCaptureStopReason::memory_limit, "memory_limit"},
    }
);
SGL_ENUM_REGISTER(ProfilerCaptureStopReason);

/// Configuration used to construct a Profiler.
struct ProfilerDesc {
    /// Maximum number of unread CPU events in each producer thread's queue. Must be a positive power of two.
    uint32_t thread_event_capacity{8192};
    /// Maximum number of completed frames retained in live trace snapshots.
    uint32_t live_frame_count{120};
    /// Maximum number of CPU and GPU zones retained in live trace snapshots.
    uint32_t live_event_capacity{100000};
    /// Maximum completed and ended GPU-pending frames retained in the global frame stream.
    uint32_t frame_stats_window_size{120};
    /// Number of timestamp queries shared by GPU zones. Each GPU zone uses two queries; the count must be positive and
    /// even.
    uint32_t gpu_query_pool_size{16384};
    /// Enable automatic zones around SlangPy functional dispatch recording.
    bool enable_auto_zones{true};
    /// Mirror command-encoder profiling zones into backend debug groups.
    bool enable_debug_groups{false};
};

/// Configuration for a bounded profiler capture.
struct ProfilerCaptureDesc {
    /// Maximum capture-owned zone and frame storage in bytes.
    uint64_t max_memory_bytes{256ull * 1024ull * 1024ull};
};

/// Profiling-site metadata backed by process-lifetime interned strings.
/// String views remain valid for the lifetime of the process.
struct ProfilerSite {
    /// One-based site identifier referenced by zones and frames.
    uint32_t id{0};
    /// Source filename supplied when the site was registered.
    std::string_view file;
    /// One-based source line number, or zero when unavailable.
    uint32_t line{0};
    /// Compact qualified function name.
    std::string_view function;
    /// User-facing zone or frame name.
    std::string_view name;
};

/// Metadata for one CPU thread or GPU queue timeline.
struct ProfilerTimeline {
    /// Execution domain represented by the timeline.
    ProfilerTimelineType type{ProfilerTimelineType::cpu};
    /// Human-readable timeline name.
    std::string name;
    /// Platform thread identifier for CPU timelines, otherwise zero.
    uint64_t thread_id{0};
    /// Device identity for GPU timelines, otherwise zero.
    uint64_t device_id{0};
    /// Command queue type for GPU timelines.
    CommandQueueType queue{CommandQueueType::graphics};
};

/// Completed frame boundary stored in a profiler trace.
struct ProfilerFrame {
    /// Monotonically increasing profiler-wide frame index.
    uint32_t index{0};
    /// Profiling site that names this frame.
    uint32_t site_id{0};
    /// CPU timeline on which the frame was recorded.
    uint32_t timeline_id{0};
    /// Frame start time in profiler-clock nanoseconds.
    uint64_t start_ns{0};
    /// Frame duration in nanoseconds.
    uint64_t duration_ns{0};
};

/// Distribution of duration samples in milliseconds.
struct ProfilerDurationStatistics {
    /// Number of samples.
    uint64_t count{0};
    /// Sum of sample durations in milliseconds.
    double total_ms{0.0};
    /// Minimum sample duration in milliseconds.
    double minimum_ms{0.0};
    /// Maximum sample duration in milliseconds.
    double maximum_ms{0.0};
    /// Arithmetic mean duration in milliseconds.
    double mean_ms{0.0};
    /// Population standard deviation in milliseconds.
    double standard_deviation_ms{0.0};
    /// Linearly interpolated 50th percentile in milliseconds.
    double p50_ms{0.0};
    /// Linearly interpolated 90th percentile in milliseconds.
    double p90_ms{0.0};
    /// Linearly interpolated 95th percentile in milliseconds.
    double p95_ms{0.0};
    /// Linearly interpolated 99th percentile in milliseconds.
    double p99_ms{0.0};
};

/// Mergeable duration summary that does not retain individual call samples.
struct ProfilerCallStatistics {
    /// Number of calls.
    uint64_t count{0};
    /// Sum of call durations in milliseconds.
    double total_ms{0.0};
    /// Minimum call duration in milliseconds.
    double minimum_ms{0.0};
    /// Maximum call duration in milliseconds.
    double maximum_ms{0.0};
    /// Arithmetic mean call duration in milliseconds.
    double mean_ms{0.0};
    /// Population standard deviation in milliseconds.
    double standard_deviation_ms{0.0};
};

/// Diagnostic counters shared by immutable profiler snapshots.
struct ProfilerDiagnostics {
    /// Number of producer events dropped because a per-thread event queue or zone stack was full.
    uint64_t producer_drop_count{0};
    /// Maximum number of unread events observed in any per-thread producer queue.
    uint64_t thread_event_queue_high_water_mark{0};
    /// Number of statistics nodes promoted to roots because their parent event was unavailable.
    uint64_t hierarchy_loss_count{0};
    /// Number of GPU zones recorded without GPU timing because no timestamp-query pair was available.
    uint64_t gpu_query_exhaustion_count{0};
    /// Number of allocated GPU zones still awaiting submission or timestamp resolution.
    uint64_t pending_gpu_zone_count{0};
};

/// Availability of one entry's GPU duration in one retained frame.
enum class ProfilerFrameGpuStatus : uint8_t {
    absent,      ///< The entry had no occurrences; its zero duration is valid.
    unavailable, ///< The entry occurred but did not request GPU timing.
    complete,    ///< All requested GPU timings resolved successfully.
    incomplete,  ///< At least one requested GPU timing could not be resolved.
};
SGL_ENUM_INFO(
    ProfilerFrameGpuStatus,
    {
        {ProfilerFrameGpuStatus::absent, "absent"},
        {ProfilerFrameGpuStatus::unavailable, "unavailable"},
        {ProfilerFrameGpuStatus::complete, "complete"},
        {ProfilerFrameGpuStatus::incomplete, "incomplete"},
    }
);
SGL_ENUM_REGISTER(ProfilerFrameGpuStatus);

/// Frame-aligned statistics for one hierarchical CPU zone path.
///
/// CPU time per frame is the sum of inclusive occurrences at this path. Parent and child entries are therefore not
/// additive. GPU time is the sum of directly measured occurrences and can exceed elapsed frame time when work
/// overlaps.
struct ProfilerFrameStatsEntry {
    /// Index of the parent entry in the same frame statistics list, or -1 for a root.
    int32_t parent_index{-1};
    /// Profiling site represented by this hierarchical path.
    uint32_t site_id{0};
    /// User-facing profiling-site name.
    std::string name;
    /// Distribution of summed CPU duration per retained frame.
    ProfilerDurationStatistics cpu_time_per_frame;
    /// Distribution of summed GPU duration for frames with complete GPU measurements.
    ProfilerDurationStatistics gpu_time_per_frame;
    /// Mergeable statistics over individual CPU occurrences.
    ProfilerCallStatistics cpu_time_per_call;
    /// Mergeable statistics over individual resolved GPU occurrences.
    ProfilerCallStatistics gpu_time_per_call;
};

/// Non-owning view of one retained completed frame.
struct ProfilerFrameStatsSampleView {
    /// Monotonically increasing profiler-wide frame index.
    uint32_t frame_index{0};
    /// Completed frame duration in milliseconds.
    double frame_time_ms{0.0};
    /// Number of occurrences for each entry.
    std::span<const uint32_t> call_count;
    /// Summed inclusive CPU duration in milliseconds for each entry.
    std::span<const double> cpu_time_ms;
    /// Summed directly measured GPU duration in milliseconds for each entry.
    std::span<const double> gpu_time_ms;
    /// GPU timing availability for each entry.
    std::span<const ProfilerFrameGpuStatus> gpu_status;
};

/// Immutable structure-of-arrays storage for at most 4096 zones.
///
/// Bounded chunks let trace queries and exports process large captures without flattening all zones into one
/// allocation. The Python bindings expose each column as a zero-copy, read-only NumPy array whose lifetime is tied to
/// the chunk.
class SGL_API ProfilerZoneChunk : public Object {
    SGL_OBJECT(ProfilerZoneChunk)
public:
    /// Number of zones in this chunk.
    size_t size() const { return start_ns.size(); }

    /// Zone start times in profiler-clock nanoseconds.
    std::vector<uint64_t> start_ns;
    /// Zone durations in nanoseconds.
    std::vector<uint64_t> duration_ns;
    /// CPU/GPU correlation identifiers.
    std::vector<uint64_t> correlation_id;
    /// Indices into ProfilerTrace::timelines().
    std::vector<uint32_t> timeline_id;
    /// One-based identifiers into ProfilerTrace::sites().
    std::vector<uint32_t> site_id;
    /// Global zone indices of parents on the same timeline, or -1 for roots.
    std::vector<int32_t> parent_index;
    /// Profiler-wide frame indices, or -1 for zones outside a frame.
    std::vector<int32_t> frame_index;
};

class ProfilerTrace;

/// Immutable set of global zone indices selected from one ProfilerTrace.
class SGL_API ProfilerZoneSelection : public Object {
    SGL_OBJECT(ProfilerZoneSelection)
public:
    /// Number of selected zones.
    uint64_t count() const { return m_indices.size(); }
    /// Sorted global zone indices into the source trace.
    const std::vector<uint32_t>& indices() const { return m_indices; }
    /// Calculate duration statistics for the selected zones.
    ProfilerDurationStatistics statistics() const;

private:
    ref<ProfilerTrace> m_trace;
    std::vector<uint32_t> m_indices;
    friend class ProfilerTrace;
};

/// Immutable profiler trace containing metadata, frame boundaries, and chunked CPU/GPU zones.
class SGL_API ProfilerTrace : public Object {
    SGL_OBJECT(ProfilerTrace)
public:
    /// Capture or live-history start time in profiler-clock nanoseconds.
    uint64_t start_ns() const { return m_start_ns; }
    /// Capture or live-history stop time in profiler-clock nanoseconds.
    uint64_t stop_ns() const { return m_stop_ns; }
    /// Reason why capture recording stopped.
    ProfilerCaptureStopReason stop_reason() const { return m_stop_reason; }
    /// Whether a bounded capture stopped at its memory limit.
    bool truncated() const { return m_truncated; }
    /// Capture-owned zone and frame storage in bytes.
    uint64_t memory_bytes() const { return m_memory_bytes; }
    /// Diagnostic counters captured with this snapshot.
    const ProfilerDiagnostics& diagnostics() const { return m_diagnostics; }

    /// Process-global profiling sites visible when the snapshot was built.
    const std::vector<ProfilerSite>& sites() const { return m_sites; }
    /// CPU and GPU timelines referenced by this trace.
    const std::vector<ProfilerTimeline>& timelines() const { return m_timelines; }
    /// Completed frame boundaries retained by this trace.
    const std::vector<ProfilerFrame>& frames() const { return m_frames; }
    /// Immutable structure-of-arrays zone chunks.
    const std::vector<ref<ProfilerZoneChunk>>& zone_chunks() const { return m_zone_chunks; }
    /// Total number of zones across all chunks.
    uint64_t zone_count() const;

    /// Select zones using exact-name, timeline, frame, and timestamp filters.
    /// Frame and timestamp ranges are half-open. Zones overlap the requested timestamp range rather than requiring
    /// their complete duration to be contained in it. Zones outside frames are excluded whenever a frame bound is set.
    ref<ProfilerZoneSelection> query_zones(
        std::optional<std::string> name = {},
        std::optional<ProfilerTimelineType> timeline_type = {},
        std::optional<uint32_t> frame_begin = {},
        std::optional<uint32_t> frame_end = {},
        std::optional<uint64_t> start_ns = {},
        std::optional<uint64_t> end_ns = {}
    ) const;
    /// Stream this trace as Chrome trace-event JSON suitable for Perfetto.
    void write_to_json(const std::filesystem::path& path) const;

private:
    uint64_t m_start_ns{0};
    uint64_t m_stop_ns{0};
    ProfilerCaptureStopReason m_stop_reason{ProfilerCaptureStopReason::user};
    bool m_truncated{false};
    uint64_t m_memory_bytes{0};
    ProfilerDiagnostics m_diagnostics;
    std::vector<ProfilerSite> m_sites;
    std::vector<ProfilerTimeline> m_timelines;
    std::vector<ProfilerFrame> m_frames;
    std::vector<ref<ProfilerZoneChunk>> m_zone_chunks;
    friend class Profiler;
    friend class ProfilerZoneSelection;
    friend struct ProfilerImpl;
};

/// Immutable snapshot of rolling statistics for the profiler's global frame stream.
///
/// Sample matrices are row-major with shape (sample_count(), entry_count()). All columns are ordered oldest to newest.
class SGL_API ProfilerFrameStats : public Object {
    SGL_OBJECT(ProfilerFrameStats)
public:
    /// Number of retained completed frames.
    size_t sample_count() const { return m_sample_frame_index.size(); }
    /// Number of hierarchical entries.
    size_t entry_count() const { return m_entries.size(); }
    /// Total ended frames still waiting for GPU measurements.
    uint64_t pending_frame_count() const { return m_pending_frame_count; }
    /// Duration of the most recently completed frame in milliseconds.
    double latest_frame_ms() const { return m_sample_frame_time_ms.empty() ? 0.0 : m_sample_frame_time_ms.back(); }
    /// Distribution of completed frame durations.
    const ProfilerDurationStatistics& frame_time() const { return m_frame_time; }
    /// Hierarchical CPU-zone paths in deterministic parent-before-child preorder.
    const std::vector<ProfilerFrameStatsEntry>& entries() const { return m_entries; }
    /// Profiler-wide frame index for each retained sample.
    const std::vector<uint32_t>& sample_frame_index() const { return m_sample_frame_index; }
    /// Completed frame duration in milliseconds for each retained sample.
    const std::vector<double>& sample_frame_time_ms() const { return m_sample_frame_time_ms; }
    /// Row-major occurrence-count matrix with shape (sample_count(), entry_count()).
    const std::vector<uint32_t>& sample_call_count() const { return m_sample_call_count; }
    /// Row-major CPU-duration matrix with shape (sample_count(), entry_count()).
    const std::vector<double>& sample_cpu_time_ms() const { return m_sample_cpu_time_ms; }
    /// Row-major GPU-duration matrix with shape (sample_count(), entry_count()).
    const std::vector<double>& sample_gpu_time_ms() const { return m_sample_gpu_time_ms; }
    /// Row-major GPU-status matrix with shape (sample_count(), entry_count()).
    const std::vector<ProfilerFrameGpuStatus>& sample_gpu_status() const { return m_sample_gpu_status; }
    /// Return a non-owning entry-aligned view of one retained sample.
    ProfilerFrameStatsSampleView sample(size_t index) const;
    /// Diagnostic counters captured with this snapshot.
    const ProfilerDiagnostics& diagnostics() const { return m_diagnostics; }

    /// Return a terminal-oriented representation of the hierarchy, timings, and diagnostic counters.
    std::string to_string() const override;

private:
    uint64_t m_pending_frame_count{0};
    ProfilerDurationStatistics m_frame_time;
    std::vector<ProfilerFrameStatsEntry> m_entries;
    std::vector<uint32_t> m_sample_frame_index;
    std::vector<double> m_sample_frame_time_ms;
    std::vector<uint32_t> m_sample_call_count;
    std::vector<double> m_sample_cpu_time_ms;
    std::vector<double> m_sample_gpu_time_ms;
    std::vector<ProfilerFrameGpuStatus> m_sample_gpu_status;
    ProfilerDiagnostics m_diagnostics;
    friend struct ProfilerImpl;
};

struct ProfilerImpl;
class Profiler;

struct ProfilerZoneToken {
    Profiler* profiler{nullptr};
    void* thread_data{nullptr};
    void* gpu_state{nullptr};
    CommandEncoder* command_encoder{nullptr};
    uint64_t start_ns{0};
    uint64_t correlation_id{0};
    uint64_t parent_correlation_id{0};
    uint32_t site_id{0};
    uint32_t frame_index{~0u};
    uint32_t stack_index{0};
    uint32_t gpu_begin_query{~0u};
    bool debug_group_active{false};
};

struct ProfilerFrameToken {
    Profiler* profiler{nullptr};
    uint64_t start_ns{0};
    uint32_t frame_index{~0u};
};

/// Bounded CPU/GPU instrumentation profiler with immutable trace and frame-statistics snapshots.
class SGL_API Profiler : public Object {
    SGL_OBJECT(Profiler)
public:
    /// Create a profiler. If the current thread has no current profiler, this profiler becomes its default.
    /// The destructor removes that default entry when it remains the stack's sole entry. A default profiler must be
    /// destroyed on its constructing thread after temporary profiler scopes on that thread have ended. Destruction
    /// must not race device submission, command-buffer discard, or device close operations associated with GPU zones.
    explicit Profiler(ProfilerDesc desc = {});
    ~Profiler();

    /// Register a profiling site in the process-global site registry.
    /// Metadata is interned and remains valid for the lifetime of the process.
    static uint32_t
    register_site(std::string_view file, uint32_t line, std::string_view function, std::string_view name);

    /// Whether manual and automatic profiling zones are recorded.
    bool enabled() const { return m_enabled.load(std::memory_order_relaxed); }
    /// Enable or disable all zone and frame recording.
    void set_enabled(bool value) { m_enabled.store(value, std::memory_order_relaxed); }
    /// Whether SlangPy functional dispatches create automatic zones.
    bool enable_auto_zones() const { return m_enable_auto_zones.load(std::memory_order_relaxed); }
    /// Enable or disable automatic SlangPy functional dispatch zones.
    void set_enable_auto_zones(bool value) { m_enable_auto_zones.store(value, std::memory_order_relaxed); }
    /// Whether command-encoder profiling zones also emit backend debug groups.
    bool enable_debug_groups() const { return m_enable_debug_groups.load(std::memory_order_relaxed); }
    /// Enable or disable backend debug groups for command-encoder zones.
    void set_enable_debug_groups(bool value) { m_enable_debug_groups.store(value, std::memory_order_relaxed); }
    /// Whether a bounded capture is currently recording. A capture stopped by its memory limit is ready but inactive.
    bool capture_active() const;
    /// Constructor configuration. Python exposes this as a copy because changing it does not reconfigure the profiler.
    const ProfilerDesc& desc() const { return m_desc; }

    /// Start a bounded capture after flushing events completed before this call.
    void start_capture(ProfilerCaptureDesc desc = {});
    /// Stop and return the current or memory-limited capture.
    ///
    /// This performs one tick and flushes completed CPU producer events. It does not deliberately wait for submitted
    /// GPU work or unresolved timestamp queries, but a backend timestamp-calibration refresh performed by tick() may
    /// synchronize queued GPU work. Call Device::wait(), then tick(), before stopping when complete GPU data is
    /// required.
    ref<ProfilerTrace> stop_capture();
    /// Discard the active or completed capture without producing a trace.
    void discard_capture();
    /// Return the most recently published immutable live trace and request periodic live-trace publication.
    ref<ProfilerTrace> live_snapshot() const;
    /// Return the most recently published immutable frame statistics and request periodic statistics publication.
    ref<ProfilerFrameStats> frame_stats_snapshot() const;
    /// Release profiler-owned references to snapshots previously handed to Python.
    /// Python bindings call this while holding the GIL before acquiring a new snapshot.
    void release_retired_snapshots();
    /// Clear completed and pending frame statistics after flushing current producer events.
    void clear_frame_stats();
    /// Poll submitted GPU timestamp queries and queue resolved measurements for the collector.
    ///
    /// The poll does not deliberately wait for profiled submissions or unresolved queries. Refreshing timestamp
    /// calibration can nevertheless synchronize queued GPU work on some backends, notably CUDA.
    void tick();
    /// Block until the collector consumes events published before this call and publishes both snapshot products.
    void flush();

    /// The noexcept instrumentation entry points below intentionally call std::terminate if an internal allocation or
    /// backend operation throws. Such failures indicate violated profiler or backend invariants.

    /// Begin a CPU zone and optionally reserve GPU timestamp queries on a command encoder.
    ProfilerZoneToken begin_zone(uint32_t site_id, CommandEncoder* command_encoder = nullptr) noexcept;
    /// End a zone previously returned by begin_zone().
    void end_zone(const ProfilerZoneToken& token) noexcept;
    /// Begin the profiler's global frame, or return an invalid token if a frame is already open or closing.
    ProfilerFrameToken begin_frame(uint32_t site_id) noexcept;
    /// Close a frame previously returned by begin_frame(). The collector seals it after all attached zones publish.
    void end_frame(const ProfilerFrameToken& token) noexcept;

    std::string to_string() const override;

private:
    ProfilerDesc m_desc;
    std::atomic<bool> m_enabled{true};
    std::atomic<bool> m_enable_auto_zones{true};
    std::atomic<bool> m_enable_debug_groups{false};
    std::unique_ptr<ProfilerImpl> m_impl;
    friend struct ProfilerImpl;
};

/// Return the current thread's profiler or throw if none is active.
SGL_API Profiler* current_profiler();
/// Return the current thread's profiler, or nullptr if none is active.
SGL_API Profiler* current_profiler_or_null() noexcept;
SGL_API void push_current_profiler(Profiler* profiler);
SGL_API Profiler* pop_current_profiler();

class SGL_API ProfilerScope {
public:
    explicit ProfilerScope(Profiler* profiler);
    ~ProfilerScope();
    ProfilerScope(const ProfilerScope&) = delete;
    ProfilerScope& operator=(const ProfilerScope&) = delete;

private:
    Profiler* m_profiler;
};

namespace detail {
    inline CommandEncoder* profiler_command_encoder() noexcept
    {
        return nullptr;
    }
    inline CommandEncoder* profiler_command_encoder(CommandEncoder* command_encoder) noexcept
    {
        return command_encoder;
    }

    class SGL_API ProfilerZoneGuard {
    public:
        ProfilerZoneGuard(uint32_t site_id, CommandEncoder* command_encoder = nullptr) noexcept;
        ~ProfilerZoneGuard() noexcept;

    private:
        ProfilerZoneToken m_token;
        SGL_NON_COPYABLE_AND_MOVABLE(ProfilerZoneGuard);
    };

    inline const char* profiler_frame_name()
    {
        return "frame";
    }
    inline const char* profiler_frame_name(const char* name)
    {
        return name;
    }

    class SGL_API ProfilerFrameGuard {
    public:
        explicit ProfilerFrameGuard(uint32_t site_id) noexcept;
        ~ProfilerFrameGuard() noexcept;

    private:
        ProfilerFrameToken m_token;
        SGL_NON_COPYABLE_AND_MOVABLE(ProfilerFrameGuard);
    };
} // namespace detail

} // namespace sgl

#if SGL_MSVC
#define SGL_PROFILER_FUNCTION_NAME __FUNCSIG__
#elif SGL_GCC || SGL_CLANG
#define SGL_PROFILER_FUNCTION_NAME __PRETTY_FUNCTION__
#else
#define SGL_PROFILER_FUNCTION_NAME __func__
#endif

#define SGL_PROFILE_FUNCTION(...) SGL_PROFILE_FUNCTION_IMPL(__COUNTER__, __VA_ARGS__)
#define SGL_PROFILE_FUNCTION_IMPL(counter, ...) SGL_PROFILE_FUNCTION_IMPL2(counter, __VA_ARGS__)
#define SGL_PROFILE_FUNCTION_IMPL2(counter, ...)                                                                       \
    static const uint32_t SGL_CONCAT_STRINGS(_sgl_profiler_site_, counter)                                             \
        = ::sgl::Profiler::register_site(__FILE__, __LINE__, SGL_PROFILER_FUNCTION_NAME, SGL_PROFILER_FUNCTION_NAME);  \
    ::sgl::detail::ProfilerZoneGuard SGL_CONCAT_STRINGS(_sgl_profiler_zone_, counter)(                                 \
        SGL_CONCAT_STRINGS(_sgl_profiler_site_, counter),                                                              \
        ::sgl::detail::profiler_command_encoder(__VA_ARGS__)                                                           \
    )

#define SGL_PROFILE_ZONE(name, ...) SGL_PROFILE_ZONE_IMPL(__COUNTER__, name, __VA_ARGS__)
#define SGL_PROFILE_ZONE_IMPL(counter, name, ...) SGL_PROFILE_ZONE_IMPL2(counter, name, __VA_ARGS__)
#define SGL_PROFILE_ZONE_IMPL2(counter, name, ...)                                                                     \
    static const uint32_t SGL_CONCAT_STRINGS(_sgl_profiler_site_, counter)                                             \
        = ::sgl::Profiler::register_site(__FILE__, __LINE__, SGL_PROFILER_FUNCTION_NAME, name);                        \
    ::sgl::detail::ProfilerZoneGuard SGL_CONCAT_STRINGS(_sgl_profiler_zone_, counter)(                                 \
        SGL_CONCAT_STRINGS(_sgl_profiler_site_, counter),                                                              \
        ::sgl::detail::profiler_command_encoder(__VA_ARGS__)                                                           \
    )

#define SGL_PROFILE_FRAME(...) SGL_PROFILE_FRAME_IMPL(__COUNTER__, __VA_ARGS__)
#define SGL_PROFILE_FRAME_IMPL(counter, ...) SGL_PROFILE_FRAME_IMPL2(counter, __VA_ARGS__)
#define SGL_PROFILE_FRAME_IMPL2(counter, ...)                                                                          \
    static const uint32_t SGL_CONCAT_STRINGS(_sgl_profiler_site_, counter) = ::sgl::Profiler::register_site(           \
        __FILE__,                                                                                                      \
        __LINE__,                                                                                                      \
        SGL_PROFILER_FUNCTION_NAME,                                                                                    \
        ::sgl::detail::profiler_frame_name(__VA_ARGS__)                                                                \
    );                                                                                                                 \
    ::sgl::detail::ProfilerFrameGuard SGL_CONCAT_STRINGS(_sgl_profiler_frame_, counter)(                               \
        SGL_CONCAT_STRINGS(_sgl_profiler_site_, counter)                                                               \
    )
