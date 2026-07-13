// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler.h"

#include "sgl/core/error.h"
#include "sgl/core/platform.h"
#include "sgl/core/thread.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/query.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace sgl {

namespace {

    constexpr uint32_t ZONE_CHUNK_SIZE = 4096;
    constexpr uint32_t MAX_ZONE_DEPTH = 64;
    constexpr uint32_t GPU_ZONES_PER_CHUNK = 16;
    constexpr uint32_t INVALID_INDEX = ~0u;
    constexpr uint64_t GPU_CALIBRATION_INTERVAL_NS = 2'000'000'000ull;

    uint64_t now_ns() noexcept
    {
        return uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count()
        );
    }

    struct SiteKey {
        uint32_t file_id{0};
        uint32_t line{0};
        uint32_t function_signature_id{0};
        uint32_t name_id{0};
        bool operator==(const SiteKey&) const = default;
    };

    struct SiteKeyHash {
        size_t operator()(const SiteKey& value) const noexcept
        {
            size_t h = std::hash<uint32_t>()(value.file_id);
            h ^= std::hash<uint32_t>()(value.line) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>()(value.function_signature_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>()(value.name_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::string_view trim(std::string_view value)
    {
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
            value.remove_prefix(1);
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
            value.remove_suffix(1);
        return value;
    }

    std::string compact_function_name(std::string_view signature)
    {
        signature = trim(signature);
        if (const size_t with = signature.find(" [with "); with != std::string_view::npos)
            signature = trim(signature.substr(0, with));

        size_t parameter_open = std::string_view::npos;
        size_t open = std::string_view::npos;
        uint32_t depth = 0;
        for (size_t i = 0; i < signature.size(); ++i) {
            if (signature[i] == '(') {
                if (depth == 0)
                    open = i;
                ++depth;
            } else if (signature[i] == ')' && depth > 0) {
                --depth;
                if (depth == 0)
                    parameter_open = open;
            }
        }
        if (parameter_open == std::string_view::npos)
            return std::string(signature);

        const std::string_view prefix = trim(signature.substr(0, parameter_open));
        if (prefix.empty())
            return std::string(signature);

        const size_t operator_pos = prefix.rfind("operator");
        size_t start = operator_pos == std::string_view::npos ? prefix.size() : operator_pos;
        uint32_t template_depth = 0;
        uint32_t group_depth = 0;
        while (start > 0) {
            const char c = prefix[start - 1];
            if (c == '>') {
                ++template_depth;
            } else if (c == '<' && template_depth > 0) {
                --template_depth;
            } else if (c == ')' || c == ']') {
                ++group_depth;
            } else if ((c == '(' || c == '[') && group_depth > 0) {
                --group_depth;
            } else if (std::isspace(static_cast<unsigned char>(c)) && template_depth == 0 && group_depth == 0) {
                break;
            }
            --start;
        }
        std::string result(prefix.substr(start));
        constexpr std::string_view anonymous_namespace = "(anonymous namespace)::";
        for (size_t pos = result.find(anonymous_namespace); pos != std::string::npos;
             pos = result.find(anonymous_namespace)) {
            result.erase(pos, anonymous_namespace.size());
        }
        return result;
    }

    struct StringInterner {
        uint32_t intern(std::string_view value)
        {
            if (auto it = ids.find(value); it != ids.end())
                return it->second;
            const uint32_t id = uint32_t(strings.size());
            strings.emplace_back(value);
            ids.emplace(std::string_view(strings.back()), id);
            return id;
        }

        std::string_view get(uint32_t id) const
        {
            SGL_ASSERT(id < strings.size());
            return strings[id];
        }

        std::deque<std::string> strings;
        std::unordered_map<std::string_view, uint32_t> ids;
    };

    struct SiteRegistry {
        uint32_t compact_function_id(uint32_t signature_id)
        {
            if (auto it = compact_function_ids.find(signature_id); it != compact_function_ids.end())
                return it->second;
            const uint32_t id = strings.intern(compact_function_name(strings.get(signature_id)));
            compact_function_ids.emplace(signature_id, id);
            return id;
        }

        std::mutex mutex;
        StringInterner strings;
        std::unordered_map<uint32_t, uint32_t> compact_function_ids;
        std::unordered_map<SiteKey, uint32_t, SiteKeyHash> ids;
        std::vector<ProfilerSite> sites;
    };

    SiteRegistry& site_registry()
    {
        static SiteRegistry registry;
        return registry;
    }

    std::vector<ProfilerSite> site_snapshot()
    {
        SiteRegistry& registry = site_registry();
        std::lock_guard lock(registry.mutex);
        return registry.sites;
    }

    std::string_view site_name(uint32_t id)
    {
        SiteRegistry& registry = site_registry();
        std::lock_guard lock(registry.mutex);
        if (id == 0 || id > registry.sites.size())
            return {};
        return registry.sites[id - 1].name;
    }

    const ProfilerSite* find_site(const std::vector<ProfilerSite>& sites, uint32_t id)
    {
        if (id == 0 || id > sites.size())
            return nullptr;
        return &sites[id - 1];
    }

    template<typename T>
    ProfilerDurationStatistics calculate_statistics(std::vector<T> durations, double unit_to_ms)
    {
        ProfilerDurationStatistics result;
        result.count = durations.size();
        if (durations.empty())
            return result;

        std::sort(durations.begin(), durations.end());
        const long double total = std::accumulate(durations.begin(), durations.end(), 0.0L);
        const long double mean = total / durations.size();
        long double variance = 0.0;
        for (T value : durations) {
            const long double delta = value - mean;
            variance += delta * delta;
        }
        variance /= durations.size();

        auto percentile = [&](double p)
        {
            const double position = p * double(durations.size() - 1);
            const size_t lo = size_t(position);
            const size_t hi = std::min(lo + 1, durations.size() - 1);
            return double(durations[lo]) + (double(durations[hi]) - double(durations[lo])) * (position - lo);
        };
        result.total_ms = double(total) * unit_to_ms;
        result.minimum_ms = double(durations.front()) * unit_to_ms;
        result.maximum_ms = double(durations.back()) * unit_to_ms;
        result.mean_ms = double(mean) * unit_to_ms;
        result.standard_deviation_ms = std::sqrt(double(variance)) * unit_to_ms;
        result.p50_ms = percentile(0.50) * unit_to_ms;
        result.p90_ms = percentile(0.90) * unit_to_ms;
        result.p95_ms = percentile(0.95) * unit_to_ms;
        result.p99_ms = percentile(0.99) * unit_to_ms;
        return result;
    }

    ProfilerDurationStatistics calculate_statistics(std::vector<uint64_t> durations)
    {
        return calculate_statistics(std::move(durations), 1e-6);
    }

    ProfilerDurationStatistics calculate_statistics_ms(std::vector<double> durations)
    {
        return calculate_statistics(std::move(durations), 1.0);
    }

    void write_json_string(std::ostream& stream, std::string_view value)
    {
        stream << '"';
        for (unsigned char c : value) {
            switch (c) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\b':
                stream << "\\b";
                break;
            case '\f':
                stream << "\\f";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                if (c < 0x20)
                    stream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
                else
                    stream << char(c);
            }
        }
        stream << '"';
    }

    enum class GpuTimingStatus : uint8_t {
        unavailable,
        pending,
        complete,
        missing,
    };

    struct CpuEvent {
        enum class Type : uint8_t { zone, frame } type{Type::zone};
        uint64_t start_ns{0};
        uint64_t duration_ns{0};
        uint64_t correlation_id{0};
        uint64_t parent_correlation_id{0};
        uint32_t timeline_id{0};
        uint32_t site_id{0};
        uint32_t frame_index{INVALID_INDEX};
        GpuTimingStatus gpu_timing_status{GpuTimingStatus::unavailable};
    };

    struct StoredZone {
        uint64_t start_ns{0};
        uint64_t duration_ns{0};
        uint64_t correlation_id{0};
        uint64_t parent_correlation_id{0};
        uint32_t timeline_id{0};
        uint32_t site_id{0};
        int32_t frame_index{-1};
    };

#if SGL_MSVC
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
    struct alignas(64) RingIndex {
        std::atomic<uint64_t> value{0};
    };

    struct ThreadData {
        explicit ThreadData(uint32_t capacity)
            : events(capacity)
            , mask(capacity - 1)
        {
        }

        bool push(const CpuEvent& event) noexcept
        {
            const uint64_t write = write_index.value.load(std::memory_order_relaxed);
            const uint64_t read = read_index.value.load(std::memory_order_acquire);
            if (write - read >= events.size()) {
                drop_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            events[write & mask] = event;
            write_index.value.store(write + 1, std::memory_order_release);
            uint64_t high = write + 1 - read;
            uint64_t old = high_water_mark.load(std::memory_order_relaxed);
            while (old < high && !high_water_mark.compare_exchange_weak(old, high, std::memory_order_relaxed)) { }
            return true;
        }

        std::vector<CpuEvent> events;
        uint64_t mask{0};
        RingIndex write_index;
        RingIndex read_index;
        std::array<uint64_t, MAX_ZONE_DEPTH> zone_stack{};
        std::array<uint64_t, MAX_ZONE_DEPTH> zone_gpu_recording_stack{};
        uint32_t zone_depth{0};
        uint32_t local_sequence{0};
        uint32_t timeline_id{0};
        uint64_t gpu_recording_id{0};
        void* gpu_context{nullptr};
        void* gpu_recording{nullptr};
        std::atomic<uint64_t> drop_count{0};
        std::atomic<uint64_t> stack_overflow_count{0};
        std::atomic<uint64_t> high_water_mark{0};
    };
#if SGL_MSVC
#pragma warning(pop)
#endif

    struct TlsCacheEntry {
        Profiler* profiler{nullptr};
        uint64_t instance_id{0};
        ThreadData* thread_data{nullptr};
        std::weak_ptr<void> lifetime;
    };

    thread_local std::vector<Profiler*> tls_profiler_stack;
    thread_local std::vector<TlsCacheEntry> tls_thread_cache;
    std::atomic<uint64_t> next_profiler_instance_id{1};

} // namespace

struct ProfilerImpl {
    enum class CaptureState : uint8_t {
        idle,
        recording,
        ready,
    };

    struct CallAccumulator {
        uint64_t count{0};
        long double total_ns{0.0};
        long double squared_total_ns{0.0};
        uint64_t minimum_ns{~0ull};
        uint64_t maximum_ns{0};

        void add(uint64_t duration_ns)
        {
            ++count;
            total_ns += duration_ns;
            squared_total_ns += static_cast<long double>(duration_ns) * duration_ns;
            minimum_ns = std::min(minimum_ns, duration_ns);
            maximum_ns = std::max(maximum_ns, duration_ns);
        }

        void merge(const CallAccumulator& other)
        {
            count += other.count;
            total_ns += other.total_ns;
            squared_total_ns += other.squared_total_ns;
            minimum_ns = std::min(minimum_ns, other.minimum_ns);
            maximum_ns = std::max(maximum_ns, other.maximum_ns);
        }

        ProfilerCallStatistics statistics() const
        {
            ProfilerCallStatistics result;
            result.count = count;
            if (count == 0)
                return result;
            constexpr double ns_to_ms = 1e-6;
            const long double mean = total_ns / count;
            const long double variance = std::max(0.0L, squared_total_ns / count - mean * mean);
            result.total_ms = double(total_ns) * ns_to_ms;
            result.minimum_ms = double(minimum_ns) * ns_to_ms;
            result.maximum_ms = double(maximum_ns) * ns_to_ms;
            result.mean_ms = double(mean) * ns_to_ms;
            result.standard_deviation_ms = std::sqrt(double(variance)) * ns_to_ms;
            return result;
        }
    };

    struct PendingFrameZone {
        uint32_t site_id{0};
        uint64_t duration_ns{0};
        uint64_t parent_correlation_id{0};
        GpuTimingStatus gpu_timing_status{GpuTimingStatus::unavailable};
        uint64_t gpu_duration_ns{0};
    };

    struct FrameNodeData {
        int32_t parent_index{-1};
        uint32_t site_id{0};
        CallAccumulator cpu;
        CallAccumulator gpu;
        uint64_t gpu_requested_count{0};
        uint64_t missing_gpu_call_count{0};
    };

    struct FrameRecord {
        ProfilerFrame frame;
        std::vector<FrameNodeData> nodes;
    };

    struct PendingFrameData {
        ProfilerFrame frame;
        bool ended{false};
        uint64_t pending_gpu_count{0};
        std::unordered_map<uint64_t, PendingFrameZone> zones;
    };

    struct GpuResult {
        StoredZone zone;
        bool missing{false};
    };

    struct GpuSlot {
        uint64_t correlation_id{0};
        uint64_t parent_correlation_id{0};
        uint32_t site_id{0};
        uint32_t frame_index{INVALID_INDEX};
        bool ended{false};
    };

    struct GpuRecording {
        std::vector<uint32_t> chunks;
        uint32_t current_chunk{INVALID_INDEX};
        uint32_t chunk_cursor{GPU_ZONES_PER_CHUNK};
        uint64_t submit_id{0};
        bool submitted{false};
    };

    struct GpuCalibration {
        TimestampCalibration value;
        uint64_t anchor_ns{0};
        uint64_t refreshed_ns{0};
        bool valid{false};
    };

    struct GpuContext {
        ref<Device> device;
        CommandQueueType queue{CommandQueueType::graphics};
        ref<QueryPool> query_pool;
        std::vector<GpuSlot> slots;
        std::vector<uint32_t> free_chunks;
        std::unordered_map<uint64_t, std::unique_ptr<GpuRecording>> recordings;
        uint32_t timeline_id{0};
        DeviceCallbackID close_callback{0};
        DeviceCallbackID submitted_callback{0};
        DeviceCallbackID discarded_callback{0};
        GpuCalibration calibration;
        std::atomic<bool> closed{false};

        uint32_t chunk_capacity(uint32_t chunk) const
        {
            const uint32_t first_slot = chunk * GPU_ZONES_PER_CHUNK;
            return std::min<uint32_t>(GPU_ZONES_PER_CHUNK, uint32_t(slots.size()) - first_slot);
        }

        uint64_t recording_zone_count(const GpuRecording& recording) const
        {
            if (recording.chunks.empty())
                return 0;
            uint64_t result = 0;
            for (uint32_t chunk : recording.chunks)
                result += chunk == recording.current_chunk ? recording.chunk_cursor : chunk_capacity(chunk);
            return result;
        }

        void recycle_recording(const GpuRecording& recording)
        {
            for (uint32_t chunk : recording.chunks)
                free_chunks.push_back(chunk);
        }
    };

    explicit ProfilerImpl(Profiler* owner, const ProfilerDesc& desc)
        : owner(owner)
        , desc(desc)
        , instance_id(next_profiler_instance_id.fetch_add(1, std::memory_order_relaxed))
    {
        live_published = ref<ProfilerTrace>(new ProfilerTrace());
        frame_stats_published = ref<ProfilerFrameStats>(new ProfilerFrameStats());
        worker = std::thread(
            [this]
            {
                worker_main();
            }
        );
    }

    ~ProfilerImpl()
    {
        shutdown_gpu();
        {
            std::lock_guard lock(control_mutex);
            stopping = true;
        }
        control_cv.notify_all();
        if (worker.joinable())
            worker.join();
    }

    ThreadData* thread_data()
    {
        for (auto it = tls_thread_cache.begin(); it != tls_thread_cache.end();) {
            if (it->profiler == owner && it->instance_id == instance_id)
                return it->thread_data;
            if (it->lifetime.expired()) {
                it = tls_thread_cache.erase(it);
            } else {
                ++it;
            }
        }

        auto data = std::make_unique<ThreadData>(desc.thread_event_capacity);
        const ThreadID thread_id = platform::current_thread_id();
        ThreadData* result = data.get();
        {
            std::lock_guard lock(thread_mutex);
            data->timeline_id = uint32_t(timelines.size());
            ProfilerTimeline timeline;
            timeline.type = ProfilerTimelineType::cpu;
            timeline.thread_id = thread_id;
            timeline.name = fmt::format("CPU thread {}", thread_id);
            timelines.push_back(std::move(timeline));
            threads.push_back(std::move(data));
        }
        tls_thread_cache.push_back({owner, instance_id, result, lifetime});
        return result;
    }

    uint64_t producer_drop_count() const
    {
        uint64_t result = 0;
        std::lock_guard lock(thread_mutex);
        for (const auto& thread : threads)
            result += thread->drop_count.load(std::memory_order_relaxed)
                + thread->stack_overflow_count.load(std::memory_order_relaxed);
        return result;
    }

    uint64_t thread_event_queue_high_water_mark() const
    {
        uint64_t result = 0;
        std::lock_guard lock(thread_mutex);
        for (const auto& thread : threads)
            result = std::max(result, thread->high_water_mark.load(std::memory_order_relaxed));
        return result;
    }

    ProfilerDiagnostics make_diagnostics() const
    {
        ProfilerDiagnostics result;
        result.producer_drop_count = producer_drop_count();
        result.thread_event_queue_high_water_mark = thread_event_queue_high_water_mark();
        result.hierarchy_loss_count = hierarchy_loss_count;
        result.gpu_query_exhaustion_count = gpu_query_exhaustion_count.load(std::memory_order_relaxed);
        result.pending_gpu_zone_count = pending_gpu_zone_count.load(std::memory_order_relaxed);
        return result;
    }

    enum class GlobalFrameStatus : uint64_t {
        inactive = 0,
        open = 1,
        closed = 2,
        finalizing = 3,
    };

    static constexpr uint64_t GLOBAL_FRAME_STATUS_MASK = 3;
    static constexpr uint64_t GLOBAL_FRAME_ZONE_COUNT_INCREMENT = 1ull << 2;
    static constexpr uint64_t GLOBAL_FRAME_ZONE_COUNT_MASK = 0x3fffffffull << 2;

    static uint64_t global_frame_state(uint32_t frame_index, GlobalFrameStatus status, uint32_t zone_count = 0)
    {
        return uint64_t(frame_index) << 32 | uint64_t(zone_count) << 2 | uint64_t(status);
    }

    static GlobalFrameStatus global_frame_status(uint64_t state)
    {
        return GlobalFrameStatus(state & GLOBAL_FRAME_STATUS_MASK);
    }

    static uint32_t global_frame_index(uint64_t state) { return uint32_t(state >> 32); }

    static uint32_t global_frame_zone_count(uint64_t state)
    {
        return uint32_t((state & GLOBAL_FRAME_ZONE_COUNT_MASK) >> 2);
    }

    bool begin_global_frame(ProfilerFrameToken& token, ThreadData* data, uint32_t site_id, uint64_t start_ns)
    {
        std::lock_guard lock(global_frame_mutex);
        if (global_frame.load(std::memory_order_acquire) != uint64_t(GlobalFrameStatus::inactive))
            return false;
        const uint32_t frame_index = next_frame_index.fetch_add(1, std::memory_order_relaxed);
        global_frame_event = {};
        global_frame_event.type = CpuEvent::Type::frame;
        global_frame_event.start_ns = start_ns;
        global_frame_event.timeline_id = data->timeline_id;
        global_frame_event.site_id = site_id;
        global_frame_event.frame_index = frame_index;
        global_frame.store(global_frame_state(frame_index, GlobalFrameStatus::open), std::memory_order_release);
        token.profiler = owner;
        token.start_ns = start_ns;
        token.frame_index = frame_index;
        return true;
    }

    void attach_zone_to_global_frame(ProfilerZoneToken& token)
    {
        uint64_t state = global_frame.load(std::memory_order_acquire);
        while (global_frame_status(state) == GlobalFrameStatus::open) {
            if (global_frame_zone_count(state) == 0x3fffffffu)
                return;
            if (global_frame.compare_exchange_weak(
                    state,
                    state + GLOBAL_FRAME_ZONE_COUNT_INCREMENT,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire
                )) {
                token.frame_index = global_frame_index(state);
                return;
            }
        }
    }

    void end_global_frame(const ProfilerFrameToken& token, uint64_t end_ns)
    {
        {
            std::lock_guard lock(global_frame_mutex);
            uint64_t state = global_frame.load(std::memory_order_acquire);
            for (;;) {
                if (global_frame_status(state) != GlobalFrameStatus::open
                    || global_frame_index(state) != token.frame_index)
                    return;
                const uint64_t closed_state = (state & ~GLOBAL_FRAME_STATUS_MASK) | uint64_t(GlobalFrameStatus::closed);
                if (global_frame.compare_exchange_weak(
                        state,
                        closed_state,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire
                    ))
                    break;
            }
            global_frame_event.duration_ns = end_ns - token.start_ns;
        }
        try_finalize_global_frame(token.frame_index);
    }

    void release_zone_from_global_frame(uint32_t frame_index)
    {
        const uint64_t previous = global_frame.fetch_sub(GLOBAL_FRAME_ZONE_COUNT_INCREMENT, std::memory_order_acq_rel);
        SGL_ASSERT(global_frame_index(previous) == frame_index && global_frame_zone_count(previous) > 0);
        if (global_frame_status(previous) == GlobalFrameStatus::closed && global_frame_zone_count(previous) == 1)
            try_finalize_global_frame(frame_index);
    }

    void try_finalize_global_frame(uint32_t frame_index)
    {
        uint64_t expected = global_frame_state(frame_index, GlobalFrameStatus::closed);
        if (!global_frame.compare_exchange_strong(
                expected,
                global_frame_state(frame_index, GlobalFrameStatus::finalizing),
                std::memory_order_acq_rel
            ))
            return;
        CpuEvent event;
        {
            std::lock_guard lock(global_frame_mutex);
            event = global_frame_event;
        }
        {
            std::lock_guard lock(sealed_frame_mutex);
            sealed_frame_events.push_back(event);
        }
        global_frame.store(uint64_t(GlobalFrameStatus::inactive), std::memory_order_release);
        control_cv.notify_all();
    }

    bool drain()
    {
        bool dirty = false;
        drained_cpu_events.clear();
        {
            std::lock_guard thread_lock(thread_mutex);
            for (const auto& thread : threads) {
                uint64_t read = thread->read_index.value.load(std::memory_order_relaxed);
                const uint64_t write = thread->write_index.value.load(std::memory_order_acquire);
                while (read < write) {
                    drained_cpu_events.push_back(thread->events[read & thread->mask]);
                    ++read;
                }
                thread->read_index.value.store(read, std::memory_order_release);
            }
        }
        for (const CpuEvent& event : drained_cpu_events) {
            consume(event);
            dirty = true;
        }
        std::vector<CpuEvent> frame_events;
        {
            std::lock_guard lock(sealed_frame_mutex);
            frame_events.swap(sealed_frame_events);
        }
        for (const CpuEvent& event : frame_events) {
            consume(event);
            dirty = true;
        }
        std::vector<GpuResult> gpu_results;
        {
            std::lock_guard lock(gpu_result_mutex);
            gpu_results.swap(pending_gpu_results);
        }
        for (const GpuResult& result : gpu_results) {
            consume_gpu(result);
            dirty = true;
        }
        return dirty;
    }

    FrameRecord build_frame_record(PendingFrameData&& pending)
    {
        FrameRecord result;
        result.frame = pending.frame;
        std::unordered_map<uint64_t, uint32_t> path_nodes;
        std::unordered_map<uint64_t, uint32_t> correlation_nodes;
        auto resolve = [&](auto&& self, uint64_t correlation_id) -> int32_t
        {
            if (auto it = correlation_nodes.find(correlation_id); it != correlation_nodes.end())
                return int32_t(it->second);
            auto zone_it = pending.zones.find(correlation_id);
            if (zone_it == pending.zones.end())
                return -1;
            PendingFrameZone& zone = zone_it->second;
            int32_t parent_index = -1;
            if (zone.parent_correlation_id != 0) {
                if (pending.zones.find(zone.parent_correlation_id) != pending.zones.end())
                    parent_index = self(self, zone.parent_correlation_id);
                else
                    ++hierarchy_loss_count;
            }
            const uint64_t path_key = (uint64_t(uint32_t(parent_index + 1)) << 32) | zone.site_id;
            uint32_t node_index;
            if (auto path_it = path_nodes.find(path_key); path_it != path_nodes.end()) {
                node_index = path_it->second;
            } else {
                node_index = uint32_t(result.nodes.size());
                result.nodes.push_back({parent_index, zone.site_id});
                path_nodes.emplace(path_key, node_index);
            }
            FrameNodeData& node = result.nodes[node_index];
            node.cpu.add(zone.duration_ns);
            if (zone.gpu_timing_status != GpuTimingStatus::unavailable) {
                ++node.gpu_requested_count;
                if (zone.gpu_timing_status == GpuTimingStatus::complete)
                    node.gpu.add(zone.gpu_duration_ns);
                else
                    ++node.missing_gpu_call_count;
            }
            correlation_nodes.emplace(correlation_id, node_index);
            return int32_t(node_index);
        };
        for (const auto& [correlation_id, zone] : pending.zones) {
            SGL_UNUSED(zone);
            resolve(resolve, correlation_id);
        }
        return result;
    }

    void finalize_ready_frames()
    {
        while (!frame_stats_pending_frames.empty()) {
            const uint32_t frame_index = frame_stats_pending_frames.front();
            auto pending_it = pending_frames.find(frame_index);
            if (pending_it == pending_frames.end()) {
                frame_stats_pending_frames.pop_front();
                continue;
            }
            if (pending_it->second.pending_gpu_count != 0)
                break;
            frame_stats_completed_frames.push_back(build_frame_record(std::move(pending_it->second)));
            pending_frames.erase(pending_it);
            frame_stats_pending_frames.pop_front();
            while (frame_stats_completed_frames.size() > desc.frame_stats_window_size)
                frame_stats_completed_frames.pop_front();
        }
    }

    void bound_pending_frames()
    {
        if (frame_stats_pending_frames.size() <= desc.frame_stats_window_size)
            return;
        auto pending_it = pending_frames.find(frame_stats_pending_frames.front());
        if (pending_it != pending_frames.end()) {
            for (auto& [correlation_id, zone] : pending_it->second.zones) {
                SGL_UNUSED(correlation_id);
                if (zone.gpu_timing_status == GpuTimingStatus::pending)
                    zone.gpu_timing_status = GpuTimingStatus::missing;
            }
            pending_it->second.pending_gpu_count = 0;
        }
        finalize_ready_frames();
    }

    void retain_zone(const StoredZone& zone)
    {
        live_zones.push_back(zone);
        while (live_zones.size() > desc.live_event_capacity)
            live_zones.pop_front();

        std::lock_guard capture_lock(capture_mutex);
        if (capture_state == CaptureState::recording && zone.start_ns >= capture_start_ns)
            append_capture_zone(zone);
    }

    void consume(const CpuEvent& event)
    {
        if (event.type == CpuEvent::Type::frame) {
            ProfilerFrame frame;
            frame.index = event.frame_index;
            frame.site_id = event.site_id;
            frame.timeline_id = event.timeline_id;
            frame.start_ns = event.start_ns;
            frame.duration_ns = event.duration_ns;
            live_frames.push_back(frame);
            {
                std::lock_guard capture_lock(capture_mutex);
                if (capture_state == CaptureState::recording && frame.start_ns >= capture_start_ns)
                    append_capture_frame(frame);
            }
            if (live_frames.size() > desc.live_frame_count)
                live_frames.erase(
                    live_frames.begin(),
                    live_frames.begin() + (live_frames.size() - desc.live_frame_count)
                );
            if (event.frame_index >= frame_stats_min_index) {
                for (auto it = pending_frames.begin(); it != pending_frames.end();) {
                    if (!it->second.ended && it->first < event.frame_index
                        && it->second.frame.timeline_id == event.timeline_id) {
                        it = pending_frames.erase(it);
                    } else {
                        ++it;
                    }
                }
                PendingFrameData& pending = pending_frames[event.frame_index];
                pending.frame = frame;
                pending.ended = true;
                frame_stats_pending_frames.push_back(event.frame_index);
                finalize_ready_frames();
                bound_pending_frames();
            }
            return;
        }

        StoredZone zone;
        zone.start_ns = event.start_ns;
        zone.duration_ns = event.duration_ns;
        zone.correlation_id = event.correlation_id;
        zone.parent_correlation_id = event.parent_correlation_id;
        zone.timeline_id = event.timeline_id;
        zone.site_id = event.site_id;
        zone.frame_index = event.frame_index == INVALID_INDEX ? -1 : int32_t(event.frame_index);
        retain_zone(zone);

        if (event.frame_index != INVALID_INDEX && event.frame_index >= frame_stats_min_index) {
            PendingFrameData& pending = pending_frames[event.frame_index];
            pending.frame.index = event.frame_index;
            pending.frame.timeline_id = event.timeline_id;
            PendingFrameZone& pending_zone = pending.zones[event.correlation_id];
            pending_zone.site_id = event.site_id;
            pending_zone.duration_ns = event.duration_ns;
            pending_zone.parent_correlation_id = event.parent_correlation_id;
            pending_zone.gpu_timing_status = event.gpu_timing_status;
            if (event.gpu_timing_status == GpuTimingStatus::pending)
                ++pending.pending_gpu_count;
        }
    }

    void consume_gpu(const GpuResult& result)
    {
        const StoredZone& zone = result.zone;
        if (!result.missing)
            retain_zone(zone);
        if (zone.frame_index < 0 || uint32_t(zone.frame_index) < frame_stats_min_index)
            return;
        auto pending_it = pending_frames.find(uint32_t(zone.frame_index));
        if (pending_it == pending_frames.end())
            return;
        auto zone_it = pending_it->second.zones.find(zone.correlation_id);
        if (zone_it == pending_it->second.zones.end())
            return;
        PendingFrameZone& pending_zone = zone_it->second;
        if (pending_zone.gpu_timing_status != GpuTimingStatus::pending)
            return;
        pending_zone.gpu_timing_status = result.missing ? GpuTimingStatus::missing : GpuTimingStatus::complete;
        pending_zone.gpu_duration_ns = zone.duration_ns;
        if (pending_it->second.pending_gpu_count > 0)
            --pending_it->second.pending_gpu_count;
        if (pending_it->second.ended) {
            finalize_ready_frames();
        }
    }

    uint64_t capture_storage_bytes() const
    {
        return capture_zones.capacity() * sizeof(StoredZone) + capture_frames.capacity() * sizeof(ProfilerFrame);
    }

    void clear_capture_storage()
    {
        std::vector<StoredZone>().swap(capture_zones);
        std::vector<ProfilerFrame>().swap(capture_frames);
    }

    void stop_capture_at_memory_limit()
    {
        capture_state = CaptureState::ready;
        capture_reason = ProfilerCaptureStopReason::memory_limit;
        capture_stop_ns = now_ns();
    }

    void append_capture_zone(const StoredZone& zone)
    {
        if (capture_zones.size() == capture_zones.capacity()) {
            const uint64_t frame_bytes = capture_frames.capacity() * sizeof(ProfilerFrame);
            const uint64_t available_bytes
                = capture_desc.max_memory_bytes > frame_bytes ? capture_desc.max_memory_bytes - frame_bytes : 0;
            const size_t max_capacity = size_t(available_bytes / sizeof(StoredZone));
            if (capture_zones.size() >= max_capacity) {
                stop_capture_at_memory_limit();
                return;
            }
            const size_t next_capacity = std::min(
                max_capacity,
                std::max<size_t>(capture_zones.capacity() * 2, std::min<size_t>(4096, max_capacity))
            );
            capture_zones.reserve(next_capacity);
        }
        capture_zones.push_back(zone);
    }

    void append_capture_frame(const ProfilerFrame& frame)
    {
        if (capture_frames.size() == capture_frames.capacity()) {
            const uint64_t zone_bytes = capture_zones.capacity() * sizeof(StoredZone);
            const uint64_t available_bytes
                = capture_desc.max_memory_bytes > zone_bytes ? capture_desc.max_memory_bytes - zone_bytes : 0;
            const size_t max_capacity = size_t(available_bytes / sizeof(ProfilerFrame));
            if (capture_frames.size() >= max_capacity) {
                stop_capture_at_memory_limit();
                return;
            }
            const size_t next_capacity = std::min(max_capacity, std::max<size_t>(capture_frames.capacity() * 2, 16));
            capture_frames.reserve(next_capacity);
        }
        capture_frames.push_back(frame);
    }

    void queue_missing_gpu_results(const GpuContext& context, const GpuRecording& recording)
    {
        std::vector<GpuResult> results;
        for (uint32_t chunk : recording.chunks) {
            const uint32_t zone_count
                = chunk == recording.current_chunk ? recording.chunk_cursor : context.chunk_capacity(chunk);
            const uint32_t first_slot = chunk * GPU_ZONES_PER_CHUNK;
            for (uint32_t i = 0; i < zone_count; ++i) {
                const GpuSlot& slot = context.slots[first_slot + i];
                StoredZone zone;
                zone.correlation_id = slot.correlation_id;
                zone.parent_correlation_id = slot.parent_correlation_id;
                zone.timeline_id = context.timeline_id;
                zone.site_id = slot.site_id;
                zone.frame_index = slot.frame_index == INVALID_INDEX ? -1 : int32_t(slot.frame_index);
                results.push_back({zone, true});
            }
        }
        if (!results.empty()) {
            std::lock_guard result_lock(gpu_result_mutex);
            pending_gpu_results.insert(pending_gpu_results.end(), results.begin(), results.end());
        }
    }

    GpuContext* get_or_create_gpu_context(CommandEncoder* encoder)
    {
        Device* device = encoder->device();
        if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
            return nullptr;
        const CommandQueueType queue = encoder->queue();
        for (const auto& context : gpu_contexts) {
            if (device == context->device.get() && context->queue == queue)
                return context->closed.load(std::memory_order_acquire) ? nullptr : context.get();
        }

        auto context = std::make_unique<GpuContext>();
        context->device = ref<Device>(device);
        context->queue = queue;
        context->query_pool = device->create_query_pool({QueryType::timestamp, desc.gpu_query_pool_size});
        context->slots.resize(desc.gpu_query_pool_size / 2);
        const uint32_t chunk_count = (uint32_t(context->slots.size()) + GPU_ZONES_PER_CHUNK - 1) / GPU_ZONES_PER_CHUNK;
        context->free_chunks.reserve(chunk_count);
        for (uint32_t i = chunk_count; i > 0; --i)
            context->free_chunks.push_back(i - 1);
        {
            std::lock_guard lock(thread_mutex);
            context->timeline_id = uint32_t(timelines.size());
            timelines.push_back({
                ProfilerTimelineType::gpu,
                fmt::format("GPU {} queue {}", uintptr_t(device), uint32_t(queue)),
                0,
                uint64_t(uintptr_t(device)),
                queue,
            });
        }
        GpuContext* result = context.get();
        context->close_callback = device->register_device_close_callback(
            [this, result](Device*)
            {
                std::lock_guard lock(gpu_mutex);
                result->closed.store(true, std::memory_order_release);
                for (auto& [recording_id, recording] : result->recordings) {
                    SGL_UNUSED(recording_id);
                    queue_missing_gpu_results(*result, *recording);
                    pending_gpu_zone_count.fetch_sub(
                        result->recording_zone_count(*recording),
                        std::memory_order_relaxed
                    );
                    result->recycle_recording(*recording);
                }
                result->recordings.clear();
            }
        );
        context->submitted_callback = device->register_command_recording_submitted_callback(
            [this, result](const CommandRecordingSubmittedEvent& event)
            {
                std::lock_guard lock(gpu_mutex);
                if (auto it = result->recordings.find(event.id); it != result->recordings.end()) {
                    it->second->submitted = true;
                    it->second->submit_id = event.submit_id;
                }
            }
        );
        context->discarded_callback = device->register_command_recording_discarded_callback(
            [this, result](const CommandRecordingDiscardedEvent& event)
            {
                std::lock_guard lock(gpu_mutex);
                if (auto it = result->recordings.find(event.id); it != result->recordings.end()) {
                    queue_missing_gpu_results(*result, *it->second);
                    pending_gpu_zone_count.fetch_sub(
                        result->recording_zone_count(*it->second),
                        std::memory_order_relaxed
                    );
                    result->recycle_recording(*it->second);
                    result->recordings.erase(it);
                }
            }
        );
        gpu_contexts.push_back(std::move(context));
        return result;
    }

    GpuSlot* begin_gpu_zone(ProfilerZoneToken& token, CommandEncoder* encoder, uint64_t parent_correlation_id)
    {
        ThreadData* thread = static_cast<ThreadData*>(token.thread_data);
        const uint64_t recording_id = encoder->recording_id();
        GpuContext* context = static_cast<GpuContext*>(thread->gpu_context);
        GpuRecording* recording = static_cast<GpuRecording*>(thread->gpu_recording);

        bool needs_chunk = thread->gpu_recording_id != recording_id || !context || !recording
            || recording->chunk_cursor >= context->chunk_capacity(recording->current_chunk);
        if (needs_chunk) {
            std::lock_guard lock(gpu_mutex);
            context = get_or_create_gpu_context(encoder);
            if (!context)
                return nullptr;
            auto& recording_ptr = context->recordings[recording_id];
            if (!recording_ptr)
                recording_ptr = std::make_unique<GpuRecording>();
            recording = recording_ptr.get();
            if (recording->chunk_cursor
                >= (recording->current_chunk == INVALID_INDEX ? 0
                                                              : context->chunk_capacity(recording->current_chunk))) {
                if (context->free_chunks.empty()) {
                    gpu_query_exhaustion_count.fetch_add(1, std::memory_order_relaxed);
                    return nullptr;
                }
                recording->current_chunk = context->free_chunks.back();
                context->free_chunks.pop_back();
                recording->chunk_cursor = 0;
                recording->chunks.push_back(recording->current_chunk);
                const uint32_t first_query = recording->current_chunk * GPU_ZONES_PER_CHUNK * 2;
                context->query_pool->reset(first_query, context->chunk_capacity(recording->current_chunk) * 2);
            }
            thread->gpu_recording_id = recording_id;
            thread->gpu_context = context;
            thread->gpu_recording = recording;
        }

        const uint32_t index = recording->current_chunk * GPU_ZONES_PER_CHUNK + recording->chunk_cursor++;
        GpuSlot& slot = context->slots[index];
        slot.correlation_id = token.correlation_id;
        slot.parent_correlation_id = parent_correlation_id;
        slot.site_id = token.site_id;
        slot.frame_index = token.frame_index;
        slot.ended = false;
        encoder->write_timestamp(context->query_pool, index * 2);
        token.gpu_state = context;
        token.gpu_begin_query = index;
        pending_gpu_zone_count.fetch_add(1, std::memory_order_relaxed);
        return &slot;
    }

    void end_gpu_zone(const ProfilerZoneToken& token)
    {
        if (!token.gpu_state || token.gpu_begin_query == INVALID_INDEX)
            return;
        GpuContext* context = static_cast<GpuContext*>(token.gpu_state);
        if (context->closed.load(std::memory_order_acquire) || token.gpu_begin_query >= context->slots.size())
            return;
        GpuSlot& slot = context->slots[token.gpu_begin_query];
        token.command_encoder->write_timestamp(context->query_pool, token.gpu_begin_query * 2 + 1);
        std::atomic_ref(slot.ended).store(true, std::memory_order_release);
    }

    const GpuCalibration& get_gpu_calibration(GpuContext& context)
    {
        const uint64_t current_ns = now_ns();
        const bool expired = !context.calibration.valid || current_ns < context.calibration.refreshed_ns
            || current_ns - context.calibration.refreshed_ns >= GPU_CALIBRATION_INTERVAL_NS;
        if (expired) {
            const uint64_t before_ns = current_ns;
            context.calibration.value = context.device->get_timestamp_calibration(context.queue);
            const uint64_t after_ns = now_ns();
            SGL_CHECK(context.calibration.value.gpu_frequency != 0, "Invalid GPU timestamp calibration frequency");
            context.calibration.anchor_ns = before_ns / 2 + after_ns / 2 + ((before_ns & 1) + (after_ns & 1)) / 2;
            context.calibration.refreshed_ns = after_ns;
            context.calibration.valid = true;
        }
        return context.calibration;
    }

    void tick_gpu()
    {
        std::vector<GpuResult> results;
        std::lock_guard lock(gpu_mutex);
        for (const auto& context_ptr : gpu_contexts) {
            GpuContext& context = *context_ptr;
            if (context.closed.load(std::memory_order_acquire))
                continue;
            const GpuCalibration* calibration = nullptr;
            for (auto it = context.recordings.begin(); it != context.recordings.end();) {
                GpuRecording& recording = *it->second;
                if (!recording.submitted || !context.device->is_submit_finished(recording.submit_id)) {
                    ++it;
                    continue;
                }

                bool resolved = true;
                for (uint32_t chunk : recording.chunks) {
                    const uint32_t zone_count
                        = chunk == recording.current_chunk ? recording.chunk_cursor : context.chunk_capacity(chunk);
                    const uint32_t first_query = chunk * GPU_ZONES_PER_CHUNK * 2;
                    if (context.query_pool->get_result_state(first_query, zone_count * 2)
                        != QueryResultState::resolved) {
                        resolved = false;
                        break;
                    }
                }
                if (!resolved) {
                    ++it;
                    continue;
                }

                if (!calibration)
                    calibration = &get_gpu_calibration(context);

                const auto to_ns = [&](uint64_t tick)
                {
                    const int64_t delta = tick >= calibration->value.gpu_timestamp
                        ? int64_t(tick - calibration->value.gpu_timestamp)
                        : -int64_t(calibration->value.gpu_timestamp - tick);
                    return uint64_t(
                        int64_t(calibration->anchor_ns)
                        + int64_t((static_cast<long double>(delta) * 1.0e9L) / calibration->value.gpu_frequency)
                    );
                };
                const auto duration_ns = [&](uint64_t begin_tick, uint64_t end_tick)
                {
                    if (end_tick < begin_tick)
                        return uint64_t(0);
                    return uint64_t(
                        (static_cast<long double>(end_tick - begin_tick) * 1.0e9L) / calibration->value.gpu_frequency
                    );
                };
                for (uint32_t chunk : recording.chunks) {
                    const uint32_t zone_count
                        = chunk == recording.current_chunk ? recording.chunk_cursor : context.chunk_capacity(chunk);
                    const uint32_t first_slot = chunk * GPU_ZONES_PER_CHUNK;
                    const uint32_t first_query = first_slot * 2;
                    const std::vector<uint64_t> ticks = context.query_pool->get_results(first_query, zone_count * 2);
                    for (uint32_t i = 0; i < zone_count; ++i) {
                        GpuSlot& slot = context.slots[first_slot + i];
                        if (!std::atomic_ref(slot.ended).load(std::memory_order_acquire))
                            continue;
                        StoredZone zone;
                        zone.start_ns = to_ns(ticks[i * 2]);
                        zone.duration_ns = duration_ns(ticks[i * 2], ticks[i * 2 + 1]);
                        zone.correlation_id = slot.correlation_id;
                        zone.parent_correlation_id = slot.parent_correlation_id;
                        zone.timeline_id = context.timeline_id;
                        zone.site_id = slot.site_id;
                        zone.frame_index = slot.frame_index == INVALID_INDEX ? -1 : int32_t(slot.frame_index);
                        results.push_back({zone, false});
                    }
                }
                pending_gpu_zone_count.fetch_sub(context.recording_zone_count(recording), std::memory_order_relaxed);
                context.recycle_recording(recording);
                it = context.recordings.erase(it);
            }
        }
        if (!results.empty()) {
            std::lock_guard result_lock(gpu_result_mutex);
            pending_gpu_results.insert(pending_gpu_results.end(), results.begin(), results.end());
        }
    }

    void shutdown_gpu()
    {
        std::lock_guard lock(gpu_mutex);
        for (const auto& context : gpu_contexts) {
            if (context->closed.load(std::memory_order_acquire))
                continue;
            context->device->unregister_command_recording_discarded_callback(context->discarded_callback);
            context->device->unregister_command_recording_submitted_callback(context->submitted_callback);
            context->device->unregister_device_close_callback(context->close_callback);
        }
        gpu_contexts.clear();
    }

    template<typename ZoneContainer>
    ref<ProfilerTrace> make_trace(
        const ZoneContainer& zones,
        const std::vector<ProfilerFrame>& frames,
        uint64_t start,
        uint64_t stop,
        ProfilerCaptureStopReason reason,
        bool truncated,
        uint64_t memory_bytes
    ) const
    {
        ref<ProfilerTrace> trace(new ProfilerTrace());
        trace->m_start_ns = start;
        trace->m_stop_ns = stop;
        trace->m_stop_reason = reason;
        trace->m_truncated = truncated;
        trace->m_memory_bytes = memory_bytes;
        trace->m_diagnostics = make_diagnostics();
        trace->m_sites = site_snapshot();
        {
            std::lock_guard lock(thread_mutex);
            trace->m_timelines = timelines;
        }
        trace->m_frames = frames;
        std::unordered_map<uint32_t, std::unordered_map<uint64_t, uint32_t>> indices;
        for (uint32_t i = 0; i < zones.size(); ++i)
            indices[zones[i].timeline_id].emplace(zones[i].correlation_id, i);

        ref<ProfilerZoneChunk> chunk;
        for (uint32_t i = 0; i < zones.size(); ++i) {
            if (!chunk || chunk->size() == ZONE_CHUNK_SIZE) {
                chunk = ref<ProfilerZoneChunk>(new ProfilerZoneChunk());
                const size_t reserve_count = std::min<size_t>(ZONE_CHUNK_SIZE, zones.size() - i);
                chunk->start_ns.reserve(reserve_count);
                chunk->duration_ns.reserve(reserve_count);
                chunk->correlation_id.reserve(reserve_count);
                chunk->timeline_id.reserve(reserve_count);
                chunk->site_id.reserve(reserve_count);
                chunk->parent_index.reserve(reserve_count);
                chunk->frame_index.reserve(reserve_count);
                trace->m_zone_chunks.push_back(chunk);
            }
            const StoredZone& zone = zones[i];
            int32_t parent = -1;
            if (zone.parent_correlation_id != 0) {
                auto timeline_it = indices.find(zone.timeline_id);
                if (timeline_it != indices.end()) {
                    auto it = timeline_it->second.find(zone.parent_correlation_id);
                    if (it != timeline_it->second.end())
                        parent = int32_t(it->second);
                }
            }
            chunk->start_ns.push_back(zone.start_ns);
            chunk->duration_ns.push_back(zone.duration_ns);
            chunk->correlation_id.push_back(zone.correlation_id);
            chunk->timeline_id.push_back(zone.timeline_id);
            chunk->site_id.push_back(zone.site_id);
            chunk->parent_index.push_back(parent);
            chunk->frame_index.push_back(zone.frame_index);
        }
        return trace;
    }

    ref<ProfilerFrameStats> make_frame_stats() const
    {
        struct AggregatedNode {
            int32_t parent_index{-1};
            uint32_t site_id{0};
            CallAccumulator cpu;
            CallAccumulator gpu;
        };

        ref<ProfilerFrameStats> result(new ProfilerFrameStats());
        const std::vector<ProfilerSite> sites = site_snapshot();
        result->m_pending_frame_count = frame_stats_pending_frames.size();
        const size_t frame_count = frame_stats_completed_frames.size();
        std::vector<uint64_t> frame_durations;
        frame_durations.reserve(frame_count);
        result->m_sample_frame_index.reserve(frame_count);
        result->m_sample_frame_time_ms.reserve(frame_count);
        for (const FrameRecord& frame : frame_stats_completed_frames) {
            frame_durations.push_back(frame.frame.duration_ns);
            result->m_sample_frame_index.push_back(frame.frame.index);
            result->m_sample_frame_time_ms.push_back(double(frame.frame.duration_ns) * 1e-6);
        }
        result->m_frame_time = calculate_statistics(std::move(frame_durations));

        std::vector<AggregatedNode> nodes;
        std::unordered_map<uint64_t, uint32_t> path_lookup;
        for (const FrameRecord& frame : frame_stats_completed_frames) {
            std::vector<uint32_t> local_to_global(frame.nodes.size());
            for (size_t local_index = 0; local_index < frame.nodes.size(); ++local_index) {
                const FrameNodeData& node = frame.nodes[local_index];
                const int32_t parent = node.parent_index < 0 ? -1 : int32_t(local_to_global[size_t(node.parent_index)]);
                const uint64_t key = (uint64_t(uint32_t(parent + 1)) << 32) | node.site_id;
                uint32_t global_index;
                if (auto it = path_lookup.find(key); it != path_lookup.end()) {
                    global_index = it->second;
                } else {
                    global_index = uint32_t(nodes.size());
                    AggregatedNode aggregated;
                    aggregated.parent_index = parent;
                    aggregated.site_id = node.site_id;
                    nodes.push_back(std::move(aggregated));
                    path_lookup.emplace(key, global_index);
                }
                local_to_global[local_index] = global_index;
                AggregatedNode& aggregated = nodes[global_index];
                aggregated.cpu.merge(node.cpu);
                aggregated.gpu.merge(node.gpu);
            }
        }

        std::vector<std::vector<uint32_t>> children(nodes.size());
        std::vector<uint32_t> roots;
        for (uint32_t index = 0; index < nodes.size(); ++index) {
            const int32_t parent = nodes[index].parent_index;
            if (parent < 0)
                roots.push_back(index);
            else
                children[size_t(parent)].push_back(index);
        }
        const auto node_less = [&](uint32_t left, uint32_t right)
        {
            const ProfilerSite* left_site = find_site(sites, nodes[left].site_id);
            const ProfilerSite* right_site = find_site(sites, nodes[right].site_id);
            const std::string_view left_name = left_site ? left_site->name : std::string_view();
            const std::string_view right_name = right_site ? right_site->name : std::string_view();
            if (left_name != right_name)
                return left_name < right_name;
            return nodes[left].site_id < nodes[right].site_id;
        };
        std::sort(roots.begin(), roots.end(), node_less);
        for (std::vector<uint32_t>& siblings : children)
            std::sort(siblings.begin(), siblings.end(), node_less);

        std::vector<uint32_t> canonical_order;
        canonical_order.reserve(nodes.size());
        const auto append_preorder = [&](auto&& self, uint32_t index) -> void
        {
            canonical_order.push_back(index);
            for (uint32_t child : children[index])
                self(self, child);
        };
        for (uint32_t root : roots)
            append_preorder(append_preorder, root);

        std::vector<uint32_t> old_to_canonical(nodes.size());
        for (uint32_t index = 0; index < canonical_order.size(); ++index)
            old_to_canonical[canonical_order[index]] = index;

        const size_t entry_count = nodes.size();
        const size_t matrix_size = frame_count * entry_count;
        result->m_sample_call_count.resize(matrix_size);
        result->m_sample_cpu_time_ms.resize(matrix_size);
        result->m_sample_gpu_time_ms.resize(matrix_size);
        result->m_sample_gpu_status.resize(matrix_size, ProfilerFrameGpuStatus::absent);
        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            const FrameRecord& source_frame = frame_stats_completed_frames[frame_index];
            std::vector<uint32_t> local_to_global(source_frame.nodes.size());
            for (size_t local_index = 0; local_index < source_frame.nodes.size(); ++local_index) {
                const FrameNodeData& node = source_frame.nodes[local_index];
                const int32_t parent = node.parent_index < 0 ? -1 : int32_t(local_to_global[size_t(node.parent_index)]);
                const uint64_t key = (uint64_t(uint32_t(parent + 1)) << 32) | node.site_id;
                const uint32_t old_index = path_lookup.at(key);
                local_to_global[local_index] = old_index;
                const size_t offset = frame_index * entry_count + old_to_canonical[old_index];
                result->m_sample_call_count[offset] = uint32_t(node.cpu.count);
                result->m_sample_cpu_time_ms[offset] = double(node.cpu.total_ns) * 1e-6;
                result->m_sample_gpu_time_ms[offset] = double(node.gpu.total_ns) * 1e-6;
                if (node.gpu_requested_count == 0) {
                    result->m_sample_gpu_status[offset] = ProfilerFrameGpuStatus::unavailable;
                } else if (node.missing_gpu_call_count != 0) {
                    result->m_sample_gpu_status[offset] = ProfilerFrameGpuStatus::incomplete;
                } else {
                    result->m_sample_gpu_status[offset] = ProfilerFrameGpuStatus::complete;
                }
            }
        }

        result->m_entries.reserve(entry_count);
        std::vector<double> cpu_frames;
        std::vector<double> valid_gpu_frames;
        cpu_frames.reserve(frame_count);
        valid_gpu_frames.reserve(frame_count);
        for (size_t entry_index = 0; entry_index < canonical_order.size(); ++entry_index) {
            const uint32_t old_index = canonical_order[entry_index];
            AggregatedNode& node = nodes[old_index];
            ProfilerFrameStatsEntry entry;
            entry.parent_index = node.parent_index < 0 ? -1 : int32_t(old_to_canonical[size_t(node.parent_index)]);
            entry.site_id = node.site_id;
            if (const ProfilerSite* site = find_site(sites, node.site_id))
                entry.name = site->name;
            bool has_complete_gpu_frame = false;
            cpu_frames.clear();
            valid_gpu_frames.clear();
            for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
                const size_t offset = frame_index * entry_count + entry_index;
                cpu_frames.push_back(result->m_sample_cpu_time_ms[offset]);
                switch (result->m_sample_gpu_status[offset]) {
                case ProfilerFrameGpuStatus::absent:
                    valid_gpu_frames.push_back(0.0);
                    break;
                case ProfilerFrameGpuStatus::unavailable:
                    break;
                case ProfilerFrameGpuStatus::complete:
                    has_complete_gpu_frame = true;
                    valid_gpu_frames.push_back(result->m_sample_gpu_time_ms[offset]);
                    break;
                case ProfilerFrameGpuStatus::incomplete:
                    break;
                }
            }
            entry.cpu_time_per_frame = calculate_statistics_ms(cpu_frames);
            if (has_complete_gpu_frame)
                entry.gpu_time_per_frame = calculate_statistics_ms(valid_gpu_frames);
            entry.cpu_time_per_call = node.cpu.statistics();
            entry.gpu_time_per_call = node.gpu.statistics();
            result->m_entries.push_back(std::move(entry));
        }
        result->m_diagnostics = make_diagnostics();
        return result;
    }

    void publish(bool publish_live, bool publish_frame_stats)
    {
        ref<ProfilerTrace> live;
        ref<ProfilerFrameStats> frame_stats;
        if (publish_live) {
            const uint64_t now = now_ns();
            live = make_trace(
                live_zones,
                live_frames,
                live_zones.empty() ? now : live_zones.front().start_ns,
                now,
                ProfilerCaptureStopReason::user,
                false,
                live_zones.size() * sizeof(StoredZone)
            );
        }
        if (publish_frame_stats)
            frame_stats = make_frame_stats();
        std::lock_guard lock(snapshot_mutex);
        if (publish_live) {
            if (live_published && live_published->self_py())
                retired_live_snapshots.push_back(std::move(live_published));
            live_published = std::move(live);
        }
        if (publish_frame_stats) {
            if (frame_stats_published && frame_stats_published->self_py())
                retired_frame_stats_snapshots.push_back(std::move(frame_stats_published));
            frame_stats_published = std::move(frame_stats);
        }
    }

    bool flush_targets_satisfied() const
    {
        if (!flush_pending)
            return false;
        std::lock_guard lock(thread_mutex);
        if (flush_targets.size() > threads.size())
            return false;
        for (size_t i = 0; i < flush_targets.size(); ++i) {
            if (threads[i]->read_index.value.load(std::memory_order_acquire) < flush_targets[i])
                return false;
        }
        return true;
    }

    void worker_main()
    {
        platform::set_current_thread_name("Profiler");
        auto last_publish = std::chrono::steady_clock::now();
        bool live_dirty = false;
        bool frame_stats_dirty = false;
        for (;;) {
            {
                std::unique_lock lock(control_mutex);
                control_cv.wait_for(
                    lock,
                    std::chrono::milliseconds(1),
                    [&]
                    {
                        return stopping || flush_pending || clear_frame_stats_requested != clear_frame_stats_completed
                            || live_snapshot_refresh_requested.load(std::memory_order_relaxed)
                            || frame_stats_snapshot_refresh_requested.load(std::memory_order_relaxed);
                    }
                );
            }
            if (drain()) {
                live_dirty = true;
                frame_stats_dirty = true;
            }
            const auto now = std::chrono::steady_clock::now();
            const bool interval_elapsed = now - last_publish >= std::chrono::milliseconds(16);
            const bool publish_live = live_snapshot_refresh_requested.exchange(false, std::memory_order_relaxed)
                || (live_dirty && live_snapshot_interest.load(std::memory_order_relaxed) && interval_elapsed);
            const bool publish_frame_stats
                = frame_stats_snapshot_refresh_requested.exchange(false, std::memory_order_relaxed)
                || (frame_stats_dirty && frame_stats_snapshot_interest.load(std::memory_order_relaxed)
                    && interval_elapsed);
            if (publish_live || publish_frame_stats) {
                publish(publish_live, publish_frame_stats);
                live_dirty &= !publish_live;
                frame_stats_dirty &= !publish_frame_stats;
                last_publish = now;
            }
            {
                std::lock_guard lock(control_mutex);
                if (flush_targets_satisfied()) {
                    publish(true, true);
                    live_dirty = false;
                    frame_stats_dirty = false;
                    flush_completed = flush_requested;
                    flush_pending = false;
                    control_cv.notify_all();
                }
                if (clear_frame_stats_requested != clear_frame_stats_completed) {
                    frame_stats_pending_frames.clear();
                    frame_stats_completed_frames.clear();
                    pending_frames.clear();
                    frame_stats_min_index = next_frame_index.load(std::memory_order_relaxed);
                    publish(false, true);
                    frame_stats_dirty = false;
                    clear_frame_stats_completed = clear_frame_stats_requested;
                    control_cv.notify_all();
                }
                if (stopping) {
                    const bool final_live = live_dirty && live_snapshot_interest.load(std::memory_order_relaxed);
                    const bool final_frame_stats
                        = frame_stats_dirty && frame_stats_snapshot_interest.load(std::memory_order_relaxed);
                    if (final_live || final_frame_stats)
                        publish(final_live, final_frame_stats);
                    return;
                }
            }
        }
    }

    void flush()
    {
        std::unique_lock control_lock(control_mutex);
        ++flush_requested;
        const uint64_t request = flush_requested;
        {
            std::lock_guard thread_lock(thread_mutex);
            flush_targets.clear();
            flush_targets.reserve(threads.size());
            for (const auto& thread : threads)
                flush_targets.push_back(thread->write_index.value.load(std::memory_order_acquire));
        }
        flush_pending = true;
        control_cv.notify_all();
        control_cv.wait(
            control_lock,
            [&]
            {
                return flush_completed >= request;
            }
        );
    }

    Profiler* owner;
    ProfilerDesc desc;
    uint64_t instance_id;
    std::shared_ptr<void> lifetime{std::make_shared<uint8_t>(uint8_t{0})};
    mutable std::mutex thread_mutex;
    std::vector<std::unique_ptr<ThreadData>> threads;
    std::vector<ProfilerTimeline> timelines;
    std::vector<CpuEvent> drained_cpu_events;

    std::thread worker;
    mutable std::mutex control_mutex;
    std::condition_variable control_cv;
    bool stopping{false};
    bool flush_pending{false};
    uint64_t flush_requested{0};
    uint64_t flush_completed{0};
    std::vector<uint64_t> flush_targets;
    uint64_t clear_frame_stats_requested{0};
    uint64_t clear_frame_stats_completed{0};

    mutable std::mutex snapshot_mutex;
    ref<ProfilerTrace> live_published;
    ref<ProfilerFrameStats> frame_stats_published;
    // Snapshot objects can transition to Python ownership after publication. The
    // collector must never increment or decrement such references without the GIL,
    // so old snapshots are moved here and released with the profiler itself.
    std::vector<ref<ProfilerTrace>> retired_live_snapshots;
    std::vector<ref<ProfilerFrameStats>> retired_frame_stats_snapshots;
    std::atomic<bool> live_snapshot_interest{false};
    std::atomic<bool> frame_stats_snapshot_interest{false};
    std::atomic<bool> live_snapshot_refresh_requested{false};
    std::atomic<bool> frame_stats_snapshot_refresh_requested{false};

    std::deque<StoredZone> live_zones;
    std::vector<ProfilerFrame> live_frames;
    std::deque<uint32_t> frame_stats_pending_frames;
    std::deque<FrameRecord> frame_stats_completed_frames;
    std::unordered_map<uint32_t, PendingFrameData> pending_frames;
    uint32_t frame_stats_min_index{0};
    uint64_t hierarchy_loss_count{0};

    mutable std::mutex capture_mutex;
    ProfilerCaptureDesc capture_desc;
    CaptureState capture_state{CaptureState::idle};
    ProfilerCaptureStopReason capture_reason{ProfilerCaptureStopReason::user};
    uint64_t capture_start_ns{0};
    uint64_t capture_stop_ns{0};
    std::vector<StoredZone> capture_zones;
    std::vector<ProfilerFrame> capture_frames;

    std::atomic<uint32_t> next_frame_index{0};
    std::mutex global_frame_mutex;
    std::atomic<uint64_t> global_frame{0};
    CpuEvent global_frame_event;
    std::mutex sealed_frame_mutex;
    std::vector<CpuEvent> sealed_frame_events;
    std::atomic<uint64_t> gpu_query_exhaustion_count{0};
    std::atomic<uint64_t> pending_gpu_zone_count{0};
    std::mutex gpu_mutex;
    std::vector<std::unique_ptr<GpuContext>> gpu_contexts;
    std::mutex gpu_result_mutex;
    std::vector<GpuResult> pending_gpu_results;
};

uint32_t Profiler::register_site(std::string_view file, uint32_t line, std::string_view function, std::string_view name)
{
    SiteRegistry& registry = site_registry();
    std::lock_guard lock(registry.mutex);
    const uint32_t file_id = registry.strings.intern(file);
    const uint32_t function_signature_id = registry.strings.intern(function);
    const uint32_t function_id = registry.compact_function_id(function_signature_id);
    const uint32_t name_id = name.empty() || name == function ? function_id : registry.strings.intern(name);
    const SiteKey key{file_id, line, function_signature_id, name_id};
    if (auto it = registry.ids.find(key); it != registry.ids.end())
        return it->second;
    const uint32_t id = uint32_t(registry.sites.size() + 1);
    registry.sites.push_back(
        {id, registry.strings.get(file_id), line, registry.strings.get(function_id), registry.strings.get(name_id)}
    );
    registry.ids.emplace(key, id);
    return id;
}

uint64_t ProfilerTrace::zone_count() const
{
    uint64_t result = 0;
    for (const auto& chunk : m_zone_chunks)
        result += chunk->size();
    return result;
}

ProfilerDurationStatistics ProfilerZoneSelection::statistics() const
{
    std::vector<uint64_t> durations;
    durations.reserve(m_indices.size());
    uint32_t base = 0;
    size_t selection_index = 0;
    for (const auto& chunk : m_trace->m_zone_chunks) {
        const uint32_t end = base + uint32_t(chunk->size());
        while (selection_index < m_indices.size() && m_indices[selection_index] < end) {
            durations.push_back(chunk->duration_ns[m_indices[selection_index] - base]);
            ++selection_index;
        }
        base = end;
    }
    return calculate_statistics(std::move(durations));
}

ProfilerFrameStatsSampleView ProfilerFrameStats::sample(size_t index) const
{
    const uint32_t frame_index = m_sample_frame_index.at(index);
    const size_t offset = index * entry_count();
    return {
        frame_index,
        m_sample_frame_time_ms[index],
        std::span<const uint32_t>(m_sample_call_count).subspan(offset, entry_count()),
        std::span<const double>(m_sample_cpu_time_ms).subspan(offset, entry_count()),
        std::span<const double>(m_sample_gpu_time_ms).subspan(offset, entry_count()),
        std::span<const ProfilerFrameGpuStatus>(m_sample_gpu_status).subspan(offset, entry_count()),
    };
}

std::string ProfilerFrameStats::to_string() const
{
    std::ostringstream stream;
    stream << "ProfilerFrameStats(pending_frame_count=" << pending_frame_count() << ", sample_count=" << sample_count()
           << ", latest=" << latest_frame_ms() << " ms"
           << ", producer_drop_count=" << m_diagnostics.producer_drop_count
           << ", thread_event_queue_high_water_mark=" << m_diagnostics.thread_event_queue_high_water_mark
           << ", hierarchy_loss_count=" << m_diagnostics.hierarchy_loss_count
           << ", gpu_query_exhaustion_count=" << m_diagnostics.gpu_query_exhaustion_count
           << ", pending_gpu_zone_count=" << m_diagnostics.pending_gpu_zone_count << ')';
    for (size_t index = 0; index < entries().size(); ++index) {
        const ProfilerFrameStatsEntry& entry = entries()[index];
        size_t depth = 0;
        int32_t parent = entry.parent_index;
        while (parent >= 0 && size_t(parent) < index && depth <= entries().size()) {
            ++depth;
            parent = entries()[size_t(parent)].parent_index;
        }
        uint64_t total_calls = 0;
        for (size_t sample_index = 0; sample_index < sample_count(); ++sample_index)
            total_calls += sample(sample_index).call_count[index];
        const double mean_calls_per_frame = sample_count() == 0 ? 0.0 : double(total_calls) / double(sample_count());
        stream << '\n'
               << std::string((depth + 1) * 2, ' ') << entry.name << ": calls/frame mean=" << mean_calls_per_frame
               << ", cpu/frame=";
        if (entry.cpu_time_per_frame.count == 0)
            stream << "n/a";
        else
            stream << entry.cpu_time_per_frame.mean_ms << " ms mean, " << entry.cpu_time_per_frame.p95_ms << " ms p95";
        stream << ", gpu/frame=";
        if (entry.gpu_time_per_frame.count == 0)
            stream << "n/a";
        else
            stream << entry.gpu_time_per_frame.mean_ms << " ms mean, " << entry.gpu_time_per_frame.p95_ms << " ms p95";
        stream << ", cpu/call=" << entry.cpu_time_per_call.count << " calls";
        if (entry.gpu_time_per_call.count != 0)
            stream << ", gpu/call=" << entry.gpu_time_per_call.count << " calls";
    }
    return stream.str();
}

ref<ProfilerZoneSelection> ProfilerTrace::query_zones(
    std::optional<std::string> name,
    std::optional<ProfilerTimelineType> timeline_type,
    std::optional<uint32_t> frame_begin,
    std::optional<uint32_t> frame_end,
    std::optional<uint64_t> range_start_ns,
    std::optional<uint64_t> range_end_ns
) const
{
    SGL_CHECK(!frame_begin || !frame_end || *frame_begin <= *frame_end, "frame_begin must not exceed frame_end");
    SGL_CHECK(!range_start_ns || !range_end_ns || *range_start_ns <= *range_end_ns, "start_ns must not exceed end_ns");
    ref<ProfilerZoneSelection> result(new ProfilerZoneSelection());
    result->m_trace = ref<ProfilerTrace>(const_cast<ProfilerTrace*>(this));
    uint32_t global_index = 0;
    for (const auto& chunk : m_zone_chunks) {
        for (size_t i = 0; i < chunk->size(); ++i, ++global_index) {
            const ProfilerSite* site = find_site(m_sites, chunk->site_id[i]);
            if (name && (!site || site->name != *name))
                continue;
            if (timeline_type) {
                const uint32_t id = chunk->timeline_id[i];
                if (id >= m_timelines.size() || m_timelines[id].type != *timeline_type)
                    continue;
            }
            const int32_t frame = chunk->frame_index[i];
            if ((frame_begin || frame_end) && frame < 0)
                continue;
            if (frame_begin && uint32_t(frame) < *frame_begin)
                continue;
            if (frame_end && uint32_t(frame) >= *frame_end)
                continue;
            const uint64_t zone_start = chunk->start_ns[i];
            const uint64_t zone_end = zone_start + chunk->duration_ns[i];
            if (range_start_ns && zone_end <= *range_start_ns)
                continue;
            if (range_end_ns && zone_start >= *range_end_ns)
                continue;
            result->m_indices.push_back(global_index);
        }
    }
    return result;
}

void ProfilerTrace::write_to_json(const std::filesystem::path& path) const
{
    std::ofstream stream(path, std::ios::binary);
    SGL_CHECK(stream, "Failed to open profiler trace file '{}'", path);
    stream << "{\"traceEvents\":[";
    bool first = true;
    auto separator = [&]
    {
        if (!first)
            stream << ',';
        first = false;
    };
    for (size_t timeline_id = 0; timeline_id < m_timelines.size(); ++timeline_id) {
        const ProfilerTimeline& timeline = m_timelines[timeline_id];
        separator();
        stream << "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":" << timeline_id << ",\"args\":{\"name\":";
        write_json_string(stream, timeline.name);
        stream << "}}";
    }
    for (const auto& chunk : m_zone_chunks) {
        for (size_t i = 0; i < chunk->size(); ++i) {
            separator();
            const ProfilerSite* site = find_site(m_sites, chunk->site_id[i]);
            stream << "{\"ph\":\"X\",\"name\":";
            write_json_string(stream, site ? site->name : "unknown");
            stream << ",\"pid\":1,\"tid\":" << chunk->timeline_id[i] << ",\"ts\":" << std::fixed << std::setprecision(3)
                   << double(chunk->start_ns[i]) / 1000.0 << ",\"dur\":" << double(chunk->duration_ns[i]) / 1000.0
                   << ",\"args\":{\"correlation_id\":" << chunk->correlation_id[i] << "}}";
        }
    }
    struct FlowStart {
        uint64_t timestamp_ns;
        uint32_t timeline_id;
    };
    std::unordered_map<uint64_t, FlowStart> cpu_flow_starts;
    for (const auto& chunk : m_zone_chunks) {
        for (size_t i = 0; i < chunk->size(); ++i) {
            const uint32_t timeline_id = chunk->timeline_id[i];
            if (timeline_id < m_timelines.size() && m_timelines[timeline_id].type == ProfilerTimelineType::cpu)
                cpu_flow_starts.emplace(chunk->correlation_id[i], FlowStart{chunk->start_ns[i], timeline_id});
        }
    }
    for (const auto& chunk : m_zone_chunks) {
        for (size_t i = 0; i < chunk->size(); ++i) {
            const uint32_t timeline_id = chunk->timeline_id[i];
            if (timeline_id >= m_timelines.size() || m_timelines[timeline_id].type != ProfilerTimelineType::gpu)
                continue;
            auto cpu = cpu_flow_starts.find(chunk->correlation_id[i]);
            if (cpu == cpu_flow_starts.end())
                continue;
            separator();
            stream << "{\"ph\":\"s\",\"cat\":\"cpu_gpu\",\"name\":\"CPU to GPU\",\"id\":" << chunk->correlation_id[i]
                   << ",\"pid\":1,\"tid\":" << cpu->second.timeline_id
                   << ",\"ts\":" << double(cpu->second.timestamp_ns) / 1000.0 << '}';
            separator();
            stream << "{\"ph\":\"f\",\"bp\":\"e\",\"cat\":\"cpu_gpu\",\"name\":\"CPU to GPU\",\"id\":"
                   << chunk->correlation_id[i] << ",\"pid\":1,\"tid\":" << timeline_id
                   << ",\"ts\":" << double(chunk->start_ns[i]) / 1000.0 << '}';
        }
    }
    for (const ProfilerFrame& frame : m_frames) {
        separator();
        const ProfilerSite* site = find_site(m_sites, frame.site_id);
        stream << "{\"ph\":\"X\",\"cat\":\"frame\",\"name\":";
        write_json_string(stream, site ? site->name : "frame");
        stream << ",\"pid\":1,\"tid\":" << frame.timeline_id << ",\"ts\":" << double(frame.start_ns) / 1000.0
               << ",\"dur\":" << double(frame.duration_ns) / 1000.0 << '}';
    }
    stream << "],\"displayTimeUnit\":\"ns\"}";
    SGL_CHECK(stream, "Failed while writing profiler trace file '{}'", path);
}

Profiler::Profiler(ProfilerDesc desc)
    : m_desc(desc)
    , m_enable_auto_zones(desc.enable_auto_zones)
    , m_enable_debug_groups(desc.enable_debug_groups)
{
    SGL_CHECK(
        desc.thread_event_capacity > 0 && (desc.thread_event_capacity & (desc.thread_event_capacity - 1)) == 0,
        "thread_event_capacity must be a positive power of two"
    );
    SGL_CHECK(desc.live_frame_count > 0, "live_frame_count must be positive");
    SGL_CHECK(desc.live_event_capacity > 0, "live_event_capacity must be positive");
    SGL_CHECK(desc.frame_stats_window_size > 0, "frame_stats_window_size must be positive");
    SGL_CHECK(
        desc.gpu_query_pool_size > 0 && (desc.gpu_query_pool_size & 1) == 0,
        "gpu_query_pool_size must be positive and even"
    );
    m_impl = std::make_unique<ProfilerImpl>(this, desc);
    if (tls_profiler_stack.empty())
        tls_profiler_stack.push_back(this);
}

Profiler::~Profiler()
{
    if (tls_profiler_stack.size() == 1 && tls_profiler_stack.back() == this)
        tls_profiler_stack.clear();
    std::erase_if(
        tls_thread_cache,
        [&](const TlsCacheEntry& entry)
        {
            return entry.profiler == this && entry.instance_id == m_impl->instance_id;
        }
    );
    m_impl.reset();
}

bool Profiler::capture_active() const
{
    std::lock_guard lock(m_impl->capture_mutex);
    return m_impl->capture_state == ProfilerImpl::CaptureState::recording;
}

void Profiler::start_capture(ProfilerCaptureDesc desc)
{
    SGL_CHECK(desc.max_memory_bytes > 0, "max_memory_bytes must be positive");
    flush();
    std::lock_guard lock(m_impl->capture_mutex);
    SGL_CHECK(
        m_impl->capture_state == ProfilerImpl::CaptureState::idle,
        "A capture is already active or awaiting retrieval"
    );
    m_impl->capture_desc = desc;
    m_impl->clear_capture_storage();
    m_impl->capture_reason = ProfilerCaptureStopReason::user;
    m_impl->capture_start_ns = now_ns();
    m_impl->capture_stop_ns = 0;
    m_impl->capture_state = ProfilerImpl::CaptureState::recording;
}

ref<ProfilerTrace> Profiler::stop_capture()
{
    tick();
    flush();
    std::lock_guard lock(m_impl->capture_mutex);
    SGL_CHECK(m_impl->capture_state != ProfilerImpl::CaptureState::idle, "No active or completed capture");
    if (m_impl->capture_state == ProfilerImpl::CaptureState::recording) {
        m_impl->capture_state = ProfilerImpl::CaptureState::ready;
        m_impl->capture_stop_ns = now_ns();
        m_impl->capture_reason = ProfilerCaptureStopReason::user;
    }
    ref<ProfilerTrace> result = m_impl->make_trace(
        m_impl->capture_zones,
        m_impl->capture_frames,
        m_impl->capture_start_ns,
        m_impl->capture_stop_ns,
        m_impl->capture_reason,
        m_impl->capture_reason == ProfilerCaptureStopReason::memory_limit,
        m_impl->capture_storage_bytes()
    );
    m_impl->capture_state = ProfilerImpl::CaptureState::idle;
    m_impl->clear_capture_storage();
    return result;
}

void Profiler::discard_capture()
{
    std::lock_guard lock(m_impl->capture_mutex);
    SGL_CHECK(m_impl->capture_state != ProfilerImpl::CaptureState::idle, "No active or completed capture");
    m_impl->capture_state = ProfilerImpl::CaptureState::idle;
    m_impl->clear_capture_storage();
}

ref<ProfilerTrace> Profiler::live_snapshot() const
{
    if (!m_impl->live_snapshot_interest.exchange(true, std::memory_order_relaxed)) {
        m_impl->live_snapshot_refresh_requested.store(true, std::memory_order_relaxed);
        m_impl->control_cv.notify_all();
    }
    std::lock_guard lock(m_impl->snapshot_mutex);
    return m_impl->live_published;
}

ref<ProfilerFrameStats> Profiler::frame_stats_snapshot() const
{
    if (!m_impl->frame_stats_snapshot_interest.exchange(true, std::memory_order_relaxed)) {
        m_impl->frame_stats_snapshot_refresh_requested.store(true, std::memory_order_relaxed);
        m_impl->control_cv.notify_all();
    }
    std::lock_guard lock(m_impl->snapshot_mutex);
    return m_impl->frame_stats_published;
}

void Profiler::release_retired_snapshots()
{
    std::lock_guard lock(m_impl->snapshot_mutex);
    m_impl->retired_live_snapshots.clear();
    m_impl->retired_frame_stats_snapshots.clear();
}

void Profiler::clear_frame_stats()
{
    flush();
    std::unique_lock lock(m_impl->control_mutex);
    const uint64_t request = ++m_impl->clear_frame_stats_requested;
    m_impl->control_cv.notify_all();
    m_impl->control_cv.wait(
        lock,
        [&]
        {
            return m_impl->clear_frame_stats_completed >= request;
        }
    );
}

void Profiler::tick()
{
    m_impl->tick_gpu();
}

void Profiler::flush()
{
    m_impl->flush();
}

ProfilerZoneToken Profiler::begin_zone(uint32_t site_id, CommandEncoder* command_encoder) noexcept
{
    ProfilerZoneToken token;
    if (!enabled())
        return token;
    ThreadData* data = m_impl->thread_data();
    if (data->zone_depth == MAX_ZONE_DEPTH) {
        data->stack_overflow_count.fetch_add(1, std::memory_order_relaxed);
        return token;
    }
    const uint32_t sequence = ++data->local_sequence;
    const uint64_t correlation = (uint64_t(data->timeline_id + 1) << 32) | sequence;
    token.profiler = this;
    token.thread_data = data;
    token.command_encoder = command_encoder;
    token.start_ns = now_ns();
    token.correlation_id = correlation;
    token.parent_correlation_id = data->zone_depth ? data->zone_stack[data->zone_depth - 1] : 0;
    uint64_t gpu_parent_correlation_id = 0;
    if (command_encoder) {
        const uint64_t recording_id = command_encoder->recording_id();
        for (uint32_t i = data->zone_depth; i > 0; --i) {
            if (data->zone_gpu_recording_stack[i - 1] == recording_id) {
                gpu_parent_correlation_id = data->zone_stack[i - 1];
                break;
            }
        }
    }
    token.site_id = site_id;
    m_impl->attach_zone_to_global_frame(token);
    token.stack_index = data->zone_depth;
    data->zone_stack[data->zone_depth] = correlation;
    data->zone_gpu_recording_stack[data->zone_depth] = command_encoder ? command_encoder->recording_id() : 0;
    ++data->zone_depth;
    if (command_encoder && enable_debug_groups()) {
        const std::string_view name = site_name(site_id);
        if (!name.empty()) {
            command_encoder->push_debug_group(name.data(), {});
            token.debug_group_active = true;
        }
    }
    if (command_encoder)
        m_impl->begin_gpu_zone(token, command_encoder, gpu_parent_correlation_id);
    return token;
}

void Profiler::end_zone(const ProfilerZoneToken& token) noexcept
{
    if (token.profiler != this || !token.thread_data)
        return;
    ThreadData* data = static_cast<ThreadData*>(token.thread_data);
    const uint64_t end_ns = now_ns();
    if (token.gpu_state)
        m_impl->end_gpu_zone(token);
    if (token.debug_group_active && token.command_encoder)
        token.command_encoder->pop_debug_group();
    if (data->zone_depth == 0 || token.stack_index + 1 != data->zone_depth
        || data->zone_stack[token.stack_index] != token.correlation_id) {
        data->stack_overflow_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    --data->zone_depth;
    CpuEvent event;
    event.type = CpuEvent::Type::zone;
    event.start_ns = token.start_ns;
    event.duration_ns = end_ns - token.start_ns;
    event.correlation_id = token.correlation_id;
    event.parent_correlation_id = token.parent_correlation_id;
    event.timeline_id = data->timeline_id;
    event.site_id = token.site_id;
    event.frame_index = token.frame_index;
    if (!token.command_encoder)
        event.gpu_timing_status = GpuTimingStatus::unavailable;
    else if (token.gpu_state)
        event.gpu_timing_status = GpuTimingStatus::pending;
    else
        event.gpu_timing_status = GpuTimingStatus::missing;
    data->push(event);
    if (token.frame_index != INVALID_INDEX)
        m_impl->release_zone_from_global_frame(token.frame_index);
}

ProfilerFrameToken Profiler::begin_frame(uint32_t site_id) noexcept
{
    ProfilerFrameToken token;
    if (!enabled())
        return token;
    ThreadData* data = m_impl->thread_data();
    m_impl->begin_global_frame(token, data, site_id, now_ns());
    return token;
}

void Profiler::end_frame(const ProfilerFrameToken& token) noexcept
{
    if (token.profiler != this)
        return;
    m_impl->end_global_frame(token, now_ns());
}

std::string Profiler::to_string() const
{
    return fmt::format("Profiler(enabled={}, capture_active={})", enabled(), capture_active());
}

Profiler* current_profiler()
{
    SGL_CHECK(!tls_profiler_stack.empty(), "No current profiler on this thread");
    return tls_profiler_stack.back();
}

Profiler* current_profiler_or_null() noexcept
{
    return tls_profiler_stack.empty() ? nullptr : tls_profiler_stack.back();
}

void push_current_profiler(Profiler* profiler)
{
    SGL_CHECK_NOT_NULL(profiler);
    tls_profiler_stack.push_back(profiler);
}

Profiler* pop_current_profiler()
{
    SGL_CHECK(!tls_profiler_stack.empty(), "No current profiler on this thread");
    Profiler* result = tls_profiler_stack.back();
    tls_profiler_stack.pop_back();
    return result;
}

ProfilerScope::ProfilerScope(Profiler* profiler)
    : m_profiler(profiler)
{
    push_current_profiler(profiler);
}

ProfilerScope::~ProfilerScope()
{
    SGL_ASSERT(current_profiler_or_null() == m_profiler);
    pop_current_profiler();
}

detail::ProfilerZoneGuard::ProfilerZoneGuard(uint32_t site_id, CommandEncoder* command_encoder) noexcept
{
    if (Profiler* profiler = current_profiler_or_null())
        m_token = profiler->begin_zone(site_id, command_encoder);
}

detail::ProfilerZoneGuard::~ProfilerZoneGuard() noexcept
{
    if (m_token.profiler)
        m_token.profiler->end_zone(m_token);
}

detail::ProfilerFrameGuard::ProfilerFrameGuard(uint32_t site_id) noexcept
{
    if (Profiler* profiler = current_profiler_or_null())
        m_token = profiler->begin_frame(site_id);
}

detail::ProfilerFrameGuard::~ProfilerFrameGuard() noexcept
{
    if (m_token.profiler)
        m_token.profiler->end_frame(m_token);
}

} // namespace sgl
