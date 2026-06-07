// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler.h"

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/hash.h"
#include "sgl/core/logger.h"
#include "sgl/core/timer.h"

#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/query.h"

#include <concurrentqueue.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sgl {

namespace {

    constexpr uint32_t kInvalidProfilerId = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t kMaxGpuQueryPoolPagesPerContext = 16;

    // ----------------------------------------------------------------------------
    // String/source registries
    // ----------------------------------------------------------------------------

    struct StringRegistry {
        std::mutex mutex;
        std::deque<std::string> entries;
        std::unordered_map<std::string_view, const char*> string_by_value;

        StringRegistry() = default;

        const char* intern(std::string_view value)
        {
            std::lock_guard lock(mutex);

            auto it = string_by_value.find(value);
            if (it != string_by_value.end())
                return it->second;

            const std::string& entry = entries.emplace_back(value);
            auto [inserted_it, inserted] = string_by_value.emplace(entry, entry.c_str());
            SGL_ASSERT(inserted);
            return inserted_it->second;
        }

        static StringRegistry& get()
        {
            static StringRegistry registry;
            return registry;
        }

        SGL_NON_COPYABLE_AND_MOVABLE(StringRegistry);
    };

    struct SourceLocationRegistry {
        struct Key {
            std::string_view file;
            uint32_t line{0};
            std::string_view function;
        };

        struct Entry {
            std::string file;
            uint32_t line{0};
            std::string function;
            ProfilerSourceLocation source_location;

            Entry(std::string file_, uint32_t line_, std::string function_)
                : file(std::move(file_))
                , line(line_)
                , function(std::move(function_))
            {
                source_location.file = file.c_str();
                source_location.line = line;
                source_location.function = function.c_str();
            }

            Key key() const
            {
                return {
                    .file = file,
                    .line = line,
                    .function = function,
                };
            }

            SGL_NON_COPYABLE_AND_MOVABLE(Entry);
        };

        struct KeyHasher {
            size_t operator()(const Key& key) const { return sgl::hash(key.file, key.line, key.function); }
        };

        struct KeyEquals {
            bool operator()(const Key& a, const Key& b) const
            {
                return a.file == b.file && a.line == b.line && a.function == b.function;
            }
        };

        std::mutex mutex;
        std::deque<Entry> entries;
        std::unordered_map<Key, const ProfilerSourceLocation*, KeyHasher, KeyEquals> source_location_by_key;

        SourceLocationRegistry() = default;

        const ProfilerSourceLocation* intern(std::string_view file, uint32_t line, std::string_view function)
        {
            std::lock_guard lock(mutex);

            Key key{
                .file = file,
                .line = line,
                .function = function,
            };
            auto it = source_location_by_key.find(key);
            if (it != source_location_by_key.end())
                return it->second;

            Entry& entry = entries.emplace_back(std::string(file), line, std::string(function));
            auto [inserted_it, inserted] = source_location_by_key.emplace(entry.key(), &entry.source_location);
            SGL_ASSERT(inserted);
            return inserted_it->second;
        }

        static SourceLocationRegistry& get()
        {
            static SourceLocationRegistry registry;
            return registry;
        }

        SGL_NON_COPYABLE_AND_MOVABLE(SourceLocationRegistry);
    };

    const char* timeline_type_name(ProfilerTimelineType type)
    {
        switch (type) {
        case ProfilerTimelineType::cpu:
            return "cpu";
        case ProfilerTimelineType::gpu:
            return "gpu";
        }
        SGL_UNREACHABLE();
    }

    const char* timeline_category(ProfilerTimelineType type)
    {
        switch (type) {
        case ProfilerTimelineType::cpu:
            return "sgl.cpu";
        case ProfilerTimelineType::gpu:
            return "sgl.gpu";
        }
        SGL_UNREACHABLE();
    }

    std::string display_function_name(const ProfilerSourceLocation* source_location)
    {
        if (!source_location || !source_location->function)
            return {};

        std::string_view function(source_location->function);
        size_t paren = function.find('(');
        if (paren != std::string_view::npos) {
            size_t end = paren;
            while (end > 0 && function[end - 1] == ' ')
                --end;
            size_t begin = function.rfind(' ', end == 0 ? 0 : end - 1);
            if (begin == std::string_view::npos)
                begin = 0;
            else
                ++begin;
            if (begin < end)
                return std::string(function.substr(begin, end - begin));
        }

        return std::string(function);
    }

    const char* source_file(const ProfilerSourceLocation* source_location)
    {
        return source_location && source_location->file ? source_location->file : "";
    }

    const char* source_function(const ProfilerSourceLocation* source_location)
    {
        return source_location && source_location->function ? source_location->function : "";
    }

    const char* fallback_zone_name(const ProfilerSourceLocation* source_location)
    {
        return source_location && source_location->function ? source_location->function : "zone";
    }

    // ----------------------------------------------------------------------------
    // JSON helpers
    // ----------------------------------------------------------------------------

    void write_json_string(std::ostream& stream, std::string_view value)
    {
        stream << '"';

        const char* chunk_begin = value.data();
        const char* it = value.data();
        const char* const end = value.data() + value.size();
        while (it != end) {
            const unsigned char c = static_cast<unsigned char>(*it);
            const char* escaped = nullptr;
            switch (c) {
            case '"':
                escaped = "\\\"";
                break;
            case '\\':
                escaped = "\\\\";
                break;
            case '\b':
                escaped = "\\b";
                break;
            case '\f':
                escaped = "\\f";
                break;
            case '\n':
                escaped = "\\n";
                break;
            case '\r':
                escaped = "\\r";
                break;
            case '\t':
                escaped = "\\t";
                break;
            default:
                break;
            }

            if (escaped) {
                stream.write(chunk_begin, it - chunk_begin);
                stream << escaped;
                chunk_begin = it + 1;
            } else if (c < 0x20) {
                static constexpr char hex[] = "0123456789abcdef";
                char escape[] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 0xf]};
                stream.write(chunk_begin, it - chunk_begin);
                stream.write(escape, sizeof(escape));
                chunk_begin = it + 1;
            }

            ++it;
        }

        stream.write(chunk_begin, it - chunk_begin);
        stream << '"';
    }

    void write_trace_time_us(std::ostream& stream, uint64_t timestamp_ns)
    {
        stream << (timestamp_ns / 1000);

        uint64_t fractional_ns = timestamp_ns % 1000;
        if (fractional_ns != 0) {
            char buffer[] = {
                '.',
                char('0' + (fractional_ns / 100)),
                char('0' + ((fractional_ns / 10) % 10)),
                char('0' + (fractional_ns % 10)),
            };
            stream.write(buffer, sizeof(buffer));
        }
    }

    void write_trace_event_separator(std::ostream& stream, bool& first_event)
    {
        if (first_event)
            first_event = false;
        else
            stream << ',';
    }

    const ProfilerNameRecord* find_name_record(const ProfilerTrace& trace, uint32_t id)
    {
        const auto& names = trace.names();
        return id < names.size() ? &names[id] : nullptr;
    }

    const ProfilerSourceRecord* find_source_record(const ProfilerTrace& trace, uint32_t id)
    {
        const auto& sources = trace.sources();
        return id < sources.size() ? &sources[id] : nullptr;
    }

    const ProfilerTimelineRecord* find_timeline_record(const ProfilerTrace& trace, uint32_t id)
    {
        const auto& timelines = trace.timelines();
        return id < timelines.size() ? &timelines[id] : nullptr;
    }

    // ----------------------------------------------------------------------------
    // Event/GPU/stat data
    // ----------------------------------------------------------------------------

    enum class ProfilerEventType : uint8_t {
        begin_zone,
        end_zone,
        gpu_query_result,
        begin_frame,
        end_frame,
    };

    struct GpuQueryTicket {
        uint64_t id{0};

        explicit operator bool() const { return id != 0; }
    };

    struct ProfilerGpuEvent {
        GpuQueryTicket ticket;
        uint64_t command_stream_id{0};
        uint32_t timeline_id{kInvalidProfilerId};
        CommandQueueType queue{CommandQueueType::graphics};
    };

    struct ProfilerGpuQueryResultEvent {
        GpuQueryTicket ticket;
        QueryResultStatus status{QueryResultStatus::invalid};
        uint64_t start_timestamp{0};
        uint64_t end_timestamp{0};
    };

    struct ProfilerEvent {
        ProfilerEventType type{ProfilerEventType::begin_zone};
        uint64_t timestamp{0};
        const ProfilerSourceLocation* source_location{nullptr};
        const char* name{nullptr};
        ProfilerGpuEvent gpu;
        ProfilerGpuQueryResultEvent gpu_result;
    };

    struct OpenGpuZone {
        GpuQueryTicket ticket;
        uint32_t event_id{kInvalidProfilerId};
        uint32_t frame_id{kInvalidProfilerId};
        uint32_t source_id{0};
        uint32_t name_id{0};
        uint32_t stats_node_id{kInvalidProfilerId};
        uint32_t timeline_id{kInvalidProfilerId};
        std::vector<uint32_t> child_indices;
    };

    struct OpenCpuZone {
        uint32_t event_id{kInvalidProfilerId};
        uint32_t trace_zone_id{kInvalidProfilerId};
        uint32_t stats_node_id{kInvalidProfilerId};
        uint32_t frame_id{kInvalidProfilerId};
        uint32_t source_id{0};
        uint32_t name_id{0};
        uint64_t start_timestamp{0};
        std::vector<uint32_t> trace_children;
        bool gpu_started{false};
        ProfilerGpuEvent gpu;
    };

    struct ProfilerGpuZoneRecord {
        GpuQueryTicket ticket;
        uint32_t event_id{kInvalidProfilerId};
        uint32_t frame_id{kInvalidProfilerId};
        uint32_t source_id{0};
        uint32_t name_id{0};
        uint32_t stats_node_id{kInvalidProfilerId};
        uint32_t timeline_id{kInvalidProfilerId};
        int32_t parent_index{-1};
        std::vector<uint32_t> child_indices;
        uint64_t start_timestamp{0};
        uint64_t end_timestamp{0};
        uint32_t trace_zone_id{kInvalidProfilerId};
        bool parent_pending{false};
        bool query_ready{false};
        bool stats_pending_counted{false};
        bool completed{false};
        bool dropped{false};
    };

    struct ThreadData {
        Profiler* profiler{nullptr};
        std::thread::id thread_id;
        ProfilerTimelineInfo timeline_info;
        uint32_t timeline_id{kInvalidProfilerId};
        bool hot_frame_active{false};
        uint32_t active_frame_id{kInvalidProfilerId};

        moodycamel::ConcurrentQueue<ProfilerEvent> queue;
        moodycamel::ProducerToken producer_token{queue};
        moodycamel::ConsumerToken consumer_token{queue};

        std::vector<OpenCpuZone> zone_stack;
        std::unordered_map<uint64_t, std::vector<OpenGpuZone>> gpu_zone_stack_by_command_stream;
        std::vector<ProfilerGpuZoneRecord> gpu_zones;
        std::unordered_map<uint64_t, uint32_t> gpu_zone_by_ticket;
        std::unordered_map<uint64_t, ProfilerGpuQueryResultEvent> orphan_gpu_query_results;

        ThreadData(Profiler* profiler_, std::thread::id thread_id_)
            : profiler(profiler_)
            , thread_id(thread_id_)
        {
            const uint64_t thread_hash = std::hash<std::thread::id>{}(thread_id);
            timeline_info.type = ProfilerTimelineType::cpu;
            timeline_info.name = fmt::format("CPU Thread {}", thread_hash);
            timeline_info.thread_id = thread_hash;
        }

        void queue_event(const ProfilerEvent& event) noexcept
        {
            if (!queue.enqueue(producer_token, event))
                std::terminate();
        }
    };

    struct StatsSample {
        uint32_t frame_id{kInvalidProfilerId};
        double value_ms{0.0};
    };

    struct StatsNodeState {
        uint32_t id{kInvalidProfilerId};
        uint32_t parent_id{kInvalidProfilerId};
        uint32_t source_id{0};
        uint32_t name_id{0};
        uint32_t pending_gpu_sample_count{0};
        std::vector<uint32_t> children;
        std::vector<StatsSample> cpu_samples;
        std::vector<StatsSample> gpu_samples;
    };

    struct StatsNodeKey {
        uint32_t parent_id{kInvalidProfilerId};
        uint32_t source_id{0};
        uint32_t name_id{0};

        bool operator==(const StatsNodeKey& other) const
        {
            return parent_id == other.parent_id && source_id == other.source_id && name_id == other.name_id;
        }
    };

    struct StatsNodeKeyHasher {
        size_t operator()(const StatsNodeKey& key) const
        {
            return sgl::hash(key.parent_id, key.source_id, key.name_id);
        }
    };

    struct GpuContextKey {
        Device* device{nullptr};
        CommandQueueType queue{CommandQueueType::graphics};

        bool operator<(const GpuContextKey& other) const
        {
            if (device != other.device)
                return device < other.device;
            return uint32_t(queue) < uint32_t(other.queue);
        }
    };

    struct GpuTimestampAnchor {
        bool valid{false};
        uint64_t cpu_timestamp_ns{0};
        uint64_t gpu_timestamp{0};
        uint64_t gpu_frequency{0};
    };

    struct GpuQuerySlot {
        GpuQueryTicket ticket;
        ref<QueryPool> query_pool;
        uint32_t begin_query_index{0};
        uint32_t end_query_index{0};
        uint32_t timeline_id{kInvalidProfilerId};
        CommandQueueType queue{CommandQueueType::graphics};
        ThreadData* owner_thread{nullptr};
        uint64_t command_stream_id{0};
        bool closed{false};
        bool pending_emitted{false};
        bool completed{false};
    };

    struct ProfilerGpuQueryPage {
        ref<QueryPool> query_pool;
        uint32_t allocated_query_count{0};
        std::vector<GpuQuerySlot> slots;
    };

    struct ProfilerGpuQueryChunk {
        GpuQuerySlot* next_slot{nullptr};
        GpuQuerySlot* end_slot{nullptr};
    };

    struct ProfilerGpuQueryPair {
        GpuQuerySlot* slot{nullptr};
        ref<QueryPool> query_pool;
        uint32_t begin_query_index{0};
        uint32_t end_query_index{0};
        GpuQueryTicket ticket;
        uint64_t command_stream_id{0};
        uint32_t timeline_id{kInvalidProfilerId};
        CommandQueueType queue{CommandQueueType::graphics};
    };

    struct GpuContext {
        ref<Device> device;
        CommandQueueType queue{CommandQueueType::graphics};
        ProfilerTimelineInfo timeline_info;
        uint32_t timeline_id{kInvalidProfilerId};
        bool disabled{false};
        std::vector<ProfilerGpuQueryPage> query_pages;
    };

    struct GpuQueryThreadCache {
        ProfilerImpl* profiler{nullptr};
        GpuContext* context{nullptr};
        GpuQuerySlot* next_slot{nullptr};
        GpuQuerySlot* end_slot{nullptr};
    };

    thread_local ThreadData* s_thread_data{nullptr};
    thread_local GpuQueryThreadCache s_gpu_query_cache;

} // namespace

// ----------------------------------------------------------------------------
// ProfilerTrace
// ----------------------------------------------------------------------------

void ProfilerTrace::write_to_json(const std::filesystem::path& path) const
{
    std::ofstream stream(path, std::ios::out | std::ios::binary);
    SGL_CHECK(stream.good(), "{}: failed to open profiler trace JSON for writing", path);

    uint64_t base_timestamp = 0;
    bool has_base_timestamp = false;
    for (const ProfilerZoneRecord& zone : m_zones) {
        if (!has_base_timestamp || zone.start_timestamp < base_timestamp) {
            base_timestamp = zone.start_timestamp;
            has_base_timestamp = true;
        }
    }
    for (const ProfilerFrameRecord& frame : m_frames) {
        if (!has_base_timestamp || frame.start_timestamp < base_timestamp) {
            base_timestamp = frame.start_timestamp;
            has_base_timestamp = true;
        }
    }

    stream << "{\"traceEvents\":[";
    bool first_event = true;

    for (const ProfilerTimelineRecord& timeline : m_timelines) {
        const std::string default_name = fmt::format("Timeline {}", timeline.id);
        const std::string& name = timeline.name.empty() ? default_name : timeline.name;
        write_trace_event_separator(stream, first_event);
        stream << "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":0,\"tid\":" << timeline.id << ",\"args\":{\"name\":";
        write_json_string(stream, name);
        stream << "}}";
    }

    for (const ProfilerFrameRecord& frame : m_frames) {
        const ProfilerNameRecord* name_record = find_name_record(*this, frame.name_id);
        const ProfilerSourceRecord* source_record = find_source_record(*this, frame.source_id);
        const std::string_view name = name_record && !name_record->name.empty() ? std::string_view(name_record->name)
                                                                                : std::string_view("frame");

        write_trace_event_separator(stream, first_event);
        stream << "{\"ph\":\"X\",\"cat\":\"sgl.frame\",\"name\":";
        write_json_string(stream, name);
        stream << ",\"pid\":0,\"tid\":0,\"ts\":";
        write_trace_time_us(stream, frame.start_timestamp - base_timestamp);
        stream << ",\"dur\":";
        write_trace_time_us(stream, frame.end_timestamp - frame.start_timestamp);
        stream << ",\"args\":{\"frame_id\":" << frame.id;
        if (source_record) {
            stream << ",\"source_file\":";
            write_json_string(stream, source_record->file);
            stream << ",\"source_line\":" << source_record->line << ",\"source_function\":";
            write_json_string(stream, source_record->original_function);
        }
        stream << "}}";
    }

    for (const ProfilerZoneRecord& zone : m_zones) {
        const ProfilerTimelineRecord* timeline = find_timeline_record(*this, zone.timeline_id);
        const ProfilerNameRecord* name_record = find_name_record(*this, zone.name_id);
        const ProfilerSourceRecord* source_record = find_source_record(*this, zone.source_id);
        const ProfilerTimelineType type = timeline ? timeline->type : ProfilerTimelineType::cpu;
        const std::string_view name = name_record && !name_record->name.empty() ? std::string_view(name_record->name)
                                                                                : std::string_view("zone");

        write_trace_event_separator(stream, first_event);
        stream << "{\"ph\":\"X\",\"cat\":";
        write_json_string(stream, timeline_category(type));
        stream << ",\"name\":";
        write_json_string(stream, name);
        stream << ",\"pid\":0,\"tid\":" << zone.timeline_id << ",\"ts\":";
        write_trace_time_us(stream, zone.start_timestamp - base_timestamp);
        stream << ",\"dur\":";
        write_trace_time_us(stream, zone.end_timestamp - zone.start_timestamp);
        stream << ",\"args\":{\"event_id\":" << zone.event_id << ",\"zone_id\":" << zone.id
               << ",\"frame_id\":" << zone.frame_id << ",\"timeline_type\":";
        write_json_string(stream, timeline_type_name(type));
        stream << ",\"start_timestamp_ns\":" << zone.start_timestamp << ",\"end_timestamp_ns\":" << zone.end_timestamp;

        if (source_record) {
            stream << ",\"source_file\":";
            write_json_string(stream, source_record->file);
            stream << ",\"source_line\":" << source_record->line << ",\"source_function\":";
            write_json_string(stream, source_record->original_function);
        }

        stream << "}}";
    }

    stream << "],\"displayTimeUnit\":\"ns\"}";
    SGL_CHECK(stream.good(), "{}: failed to write profiler trace JSON", path);
}

// ----------------------------------------------------------------------------
// ProfilerImpl
// ----------------------------------------------------------------------------

struct ProfilerImpl {
    Profiler* profiler{nullptr};

    std::mutex data_mutex;
    std::vector<ProfilerTimelineRecord> trace_timelines;
    std::vector<ProfilerSourceRecord> trace_sources;
    std::vector<ProfilerNameRecord> trace_names;
    std::vector<ProfilerFrameRecord> trace_frames;
    std::vector<ProfilerZoneRecord> trace_zones;
    std::vector<uint32_t> trace_child_indices;
    std::vector<uint32_t> trace_root_indices;

    std::unordered_map<const ProfilerSourceLocation*, uint32_t> source_id_by_pointer;
    std::unordered_map<std::string, uint32_t> name_id_by_value;
    std::unordered_map<uint32_t, uint32_t> trace_frame_id_by_frame_id;

    std::vector<StatsNodeState> stats_nodes;
    std::unordered_map<StatsNodeKey, uint32_t, StatsNodeKeyHasher> stats_node_id_by_key;
    uint64_t next_gpu_query_ticket{1};
    uint32_t next_event_id{1};
    uint32_t next_frame_id{0};
    uint32_t completed_frame_count{0};
    uint32_t newest_completed_frame_id{0};

    bool trace_active{false};
    uint64_t trace_start_timestamp{0};
    uint64_t trace_stop_timestamp{0};
    uint64_t trace_epoch_start_timestamp{0};

    std::thread worker_thread;
    std::atomic<bool> worker_stop{false};
    std::atomic<uint32_t> pending_event_count{0};
    std::mutex process_mutex;

    std::vector<ThreadData*> thread_data_storage;
    std::unordered_map<std::thread::id, ThreadData*> thread_data_by_id;
    std::mutex thread_data_mutex;

    std::map<GpuContextKey, GpuContext> gpu_contexts;
    moodycamel::ConcurrentQueue<GpuQuerySlot*> closed_gpu_query_slot_queue;
    std::vector<GpuQuerySlot*> pending_gpu_query_slots;
    std::mutex gpu_mutex;
    std::mutex gpu_query_process_mutex;

    explicit ProfilerImpl(Profiler* profiler_)
        : profiler(profiler_)
    {
        init_dictionaries_locked();
        const uint64_t now = Timer::now();
        trace_active = profiler->desc().trace_enabled_on_start;
        trace_start_timestamp = trace_active ? now : 0;
        trace_epoch_start_timestamp = trace_active ? now : 0;
        start_worker();
    }

    ~ProfilerImpl()
    {
        stop_worker();
        clear_gpu_query_cache();
        for (ThreadData* thread_data : thread_data_storage)
            delete thread_data;
    }

    void init_dictionaries_locked()
    {
        trace_sources.push_back({});
        trace_names.push_back({});
        name_id_by_value.emplace(std::string(), 0);
    }

    ThreadData* get_or_create_thread_data(std::thread::id thread_id)
    {
        std::lock_guard lock(thread_data_mutex);
        auto it = thread_data_by_id.find(thread_id);
        if (it != thread_data_by_id.end())
            return it->second;

        ThreadData* thread_data = new ThreadData(profiler, thread_id);
        {
            std::lock_guard data_lock(data_mutex);
            thread_data->timeline_id = uint32_t(trace_timelines.size());
            trace_timelines.push_back({
                .id = thread_data->timeline_id,
                .type = thread_data->timeline_info.type,
                .name = thread_data->timeline_info.name,
                .thread_id = thread_data->timeline_info.thread_id,
                .device_id = thread_data->timeline_info.device_id,
                .queue = thread_data->timeline_info.queue,
            });
        }

        thread_data_storage.push_back(thread_data);
        thread_data_by_id[thread_id] = thread_data;
        return thread_data;
    }

    ThreadData* get_this_thread_data()
    {
        if (!s_thread_data || s_thread_data->profiler != profiler)
            s_thread_data = get_or_create_thread_data(std::this_thread::get_id());
        return s_thread_data;
    }

    void queue_thread_event(ThreadData* thread_data, const ProfilerEvent& event) noexcept
    {
        pending_event_count.fetch_add(1, std::memory_order_acq_rel);
        thread_data->queue_event(event);
    }

    uint32_t get_source_id_locked(const ProfilerSourceLocation* source_location)
    {
        if (!source_location)
            return 0;

        auto it = source_id_by_pointer.find(source_location);
        if (it != source_id_by_pointer.end())
            return it->second;

        const uint32_t id = uint32_t(trace_sources.size());
        trace_sources.push_back({
            .id = id,
            .file = source_file(source_location),
            .line = source_location->line,
            .original_function = source_function(source_location),
            .display_function = display_function_name(source_location),
        });
        source_id_by_pointer[source_location] = id;
        return id;
    }

    uint32_t get_name_id_locked(const char* name, const ProfilerSourceLocation* source_location)
    {
        std::string display_name = name ? std::string(name) : display_function_name(source_location);
        if (display_name.empty())
            display_name = "zone";

        auto it = name_id_by_value.find(display_name);
        if (it != name_id_by_value.end())
            return it->second;

        const uint32_t id = uint32_t(trace_names.size());
        trace_names.push_back({
            .id = id,
            .name = display_name,
        });
        name_id_by_value.emplace(std::move(display_name), id);
        return id;
    }

    bool zone_in_active_trace_window_locked(uint64_t begin_timestamp) const
    {
        if (!trace_active)
            return false;
        if (begin_timestamp < trace_epoch_start_timestamp || begin_timestamp < trace_start_timestamp)
            return false;
        if (trace_stop_timestamp != 0 && begin_timestamp >= trace_stop_timestamp)
            return false;
        return true;
    }

    uint32_t get_or_create_stats_node_locked(uint32_t parent_id, uint32_t source_id, uint32_t name_id)
    {
        StatsNodeKey key{
            .parent_id = parent_id,
            .source_id = source_id,
            .name_id = name_id,
        };
        auto it = stats_node_id_by_key.find(key);
        if (it != stats_node_id_by_key.end())
            return it->second;

        const uint32_t id = uint32_t(stats_nodes.size());
        stats_nodes.push_back({
            .id = id,
            .parent_id = parent_id,
            .source_id = source_id,
            .name_id = name_id,
        });
        stats_node_id_by_key.emplace(key, id);
        if (parent_id != kInvalidProfilerId && parent_id < stats_nodes.size())
            stats_nodes[parent_id].children.push_back(id);
        return id;
    }

    void append_trace_child_range_locked(ProfilerZoneRecord& zone, const std::vector<uint32_t>& children)
    {
        zone.child_index_begin = uint32_t(trace_child_indices.size());
        zone.child_index_count = uint32_t(children.size());
        trace_child_indices.insert(trace_child_indices.end(), children.begin(), children.end());
    }

    void process_cpu_begin_zone(ThreadData& thread_data, const ProfilerEvent& event)
    {
        std::lock_guard lock(data_mutex);

        const uint32_t event_id = next_event_id++;
        const uint32_t frame_id = thread_data.active_frame_id;
        const uint32_t source_id = get_source_id_locked(event.source_location);
        const uint32_t name_id = get_name_id_locked(event.name, event.source_location);
        const uint32_t parent_stats_node_id
            = thread_data.zone_stack.empty() ? kInvalidProfilerId : thread_data.zone_stack.back().stats_node_id;
        const uint32_t stats_node_id = get_or_create_stats_node_locked(parent_stats_node_id, source_id, name_id);

        uint32_t trace_zone_id = kInvalidProfilerId;
        if (zone_in_active_trace_window_locked(event.timestamp)) {
            trace_zone_id = uint32_t(trace_zones.size());
            trace_zones.push_back({
                .id = trace_zone_id,
                .event_id = event_id,
                .timeline_id = thread_data.timeline_id,
                .frame_id = frame_id,
                .source_id = source_id,
                .name_id = name_id,
                .start_timestamp = event.timestamp,
                .end_timestamp = event.timestamp,
            });
        }

        thread_data.zone_stack.push_back({
            .event_id = event_id,
            .trace_zone_id = trace_zone_id,
            .stats_node_id = stats_node_id,
            .frame_id = frame_id,
            .source_id = source_id,
            .name_id = name_id,
            .start_timestamp = event.timestamp,
            .gpu_started = bool(event.gpu.ticket),
            .gpu = event.gpu,
        });

        if (event.gpu.ticket) {
            thread_data.gpu_zone_stack_by_command_stream[event.gpu.command_stream_id].push_back({
                .ticket = event.gpu.ticket,
                .event_id = event_id,
                .frame_id = frame_id,
                .source_id = source_id,
                .name_id = name_id,
                .stats_node_id = stats_node_id,
                .timeline_id = event.gpu.timeline_id,
            });
        }
    }

    void append_gpu_zone_locked(ThreadData& thread_data, OpenCpuZone& zone)
    {
        if (!zone.gpu_started || !zone.gpu.ticket || zone.gpu.timeline_id == kInvalidProfilerId)
            return;

        auto stack_it = thread_data.gpu_zone_stack_by_command_stream.find(zone.gpu.command_stream_id);
        if (stack_it == thread_data.gpu_zone_stack_by_command_stream.end() || stack_it->second.empty()) {
            log_warn_once("Profiler received a GPU zone end event without an open GPU zone.");
            return;
        }

        std::vector<OpenGpuZone>& gpu_stack = stack_it->second;
        OpenGpuZone open_gpu_zone = std::move(gpu_stack.back());
        gpu_stack.pop_back();

        if (open_gpu_zone.ticket.id != zone.gpu.ticket.id) {
            if (gpu_stack.empty())
                thread_data.gpu_zone_stack_by_command_stream.erase(stack_it);
            log_warn_once("Profiler received mismatched GPU zone begin/end events.");
            return;
        }

        const bool has_parent = !gpu_stack.empty();
        const uint32_t gpu_zone_index = uint32_t(thread_data.gpu_zones.size());
        thread_data.gpu_zones.push_back({
            .ticket = open_gpu_zone.ticket,
            .event_id = open_gpu_zone.event_id,
            .frame_id = open_gpu_zone.frame_id,
            .source_id = open_gpu_zone.source_id,
            .name_id = open_gpu_zone.name_id,
            .stats_node_id = open_gpu_zone.stats_node_id,
            .timeline_id = open_gpu_zone.timeline_id,
            .child_indices = std::move(open_gpu_zone.child_indices),
            .parent_pending = has_parent,
        });

        ProfilerGpuZoneRecord& gpu_zone = thread_data.gpu_zones.back();
        for (uint32_t child_index : gpu_zone.child_indices) {
            if (child_index < thread_data.gpu_zones.size()) {
                thread_data.gpu_zones[child_index].parent_index = int32_t(gpu_zone_index);
                thread_data.gpu_zones[child_index].parent_pending = false;
            }
        }

        if (has_parent)
            gpu_stack.back().child_indices.push_back(gpu_zone_index);
        else
            thread_data.gpu_zone_stack_by_command_stream.erase(stack_it);

        thread_data.gpu_zone_by_ticket[gpu_zone.ticket.id] = gpu_zone_index;

        auto orphan_it = thread_data.orphan_gpu_query_results.find(gpu_zone.ticket.id);
        if (orphan_it != thread_data.orphan_gpu_query_results.end()) {
            ProfilerGpuQueryResultEvent result = orphan_it->second;
            thread_data.orphan_gpu_query_results.erase(orphan_it);
            apply_gpu_query_result_locked(thread_data, result);
        }
    }

    void process_cpu_end_zone(ThreadData& thread_data, const ProfilerEvent& event)
    {
        std::lock_guard lock(data_mutex);

        if (thread_data.zone_stack.empty()) {
            log_warn_once("Profiler received an unmatched CPU zone end event.");
            return;
        }

        OpenCpuZone zone = std::move(thread_data.zone_stack.back());
        thread_data.zone_stack.pop_back();

        const uint64_t end_timestamp = std::max(zone.start_timestamp, event.timestamp);
        if (zone.trace_zone_id != kInvalidProfilerId && zone.trace_zone_id < trace_zones.size()) {
            ProfilerZoneRecord& trace_zone = trace_zones[zone.trace_zone_id];
            trace_zone.end_timestamp = end_timestamp;
            append_trace_child_range_locked(trace_zone, zone.trace_children);

            if (!thread_data.zone_stack.empty() && thread_data.zone_stack.back().trace_zone_id != kInvalidProfilerId)
                thread_data.zone_stack.back().trace_children.push_back(zone.trace_zone_id);
            else
                trace_root_indices.push_back(zone.trace_zone_id);
        }

        if (profiler->frame_stats_enabled() && zone.frame_id != kInvalidProfilerId
            && zone.stats_node_id != kInvalidProfilerId && zone.stats_node_id < stats_nodes.size()) {
            const double duration_ms = double(end_timestamp - zone.start_timestamp) / 1000000.0;
            stats_nodes[zone.stats_node_id].cpu_samples.push_back({
                .frame_id = zone.frame_id,
                .value_ms = duration_ms,
            });
        }

        append_gpu_zone_locked(thread_data, zone);
    }

    void process_begin_frame(ThreadData& thread_data, const ProfilerEvent& event)
    {
        std::lock_guard lock(data_mutex);

        if (thread_data.active_frame_id != kInvalidProfilerId) {
            log_warn_once("Profiler ignored an overlapping begin_frame() call on one thread.");
            return;
        }

        const uint32_t frame_id = next_frame_id++;
        thread_data.active_frame_id = frame_id;
        const uint32_t source_id = get_source_id_locked(event.source_location);
        const uint32_t name_id = get_name_id_locked(event.name, event.source_location);
        if (zone_in_active_trace_window_locked(event.timestamp)) {
            const uint32_t trace_frame_id = uint32_t(trace_frames.size());
            trace_frames.push_back({
                .id = frame_id,
                .name_id = name_id,
                .source_id = source_id,
                .start_timestamp = event.timestamp,
                .end_timestamp = event.timestamp,
            });
            trace_frame_id_by_frame_id[frame_id] = trace_frame_id;
        }
    }

    void process_end_frame(ThreadData& thread_data, const ProfilerEvent& event)
    {
        std::lock_guard lock(data_mutex);

        const uint32_t frame_id = thread_data.active_frame_id;
        if (frame_id == kInvalidProfilerId) {
            log_warn_once("Profiler ignored an end_frame() call without an active frame on this thread.");
            return;
        }
        thread_data.active_frame_id = kInvalidProfilerId;

        auto it = trace_frame_id_by_frame_id.find(frame_id);
        if (it != trace_frame_id_by_frame_id.end() && it->second < trace_frames.size()) {
            ProfilerFrameRecord& frame = trace_frames[it->second];
            frame.end_timestamp = std::max(frame.start_timestamp, event.timestamp);
        }

        newest_completed_frame_id = frame_id;
        ++completed_frame_count;
    }

    void clear_gpu_zone_pending_stats_locked(ProfilerGpuZoneRecord& record)
    {
        if (!record.stats_pending_counted)
            return;

        if (record.stats_node_id != kInvalidProfilerId && record.stats_node_id < stats_nodes.size()
            && stats_nodes[record.stats_node_id].pending_gpu_sample_count > 0) {
            --stats_nodes[record.stats_node_id].pending_gpu_sample_count;
        }
        record.stats_pending_counted = false;
    }

    void mark_gpu_zone_pending_locked(ProfilerGpuZoneRecord& record)
    {
        if (record.completed || record.stats_pending_counted || record.stats_node_id == kInvalidProfilerId
            || record.stats_node_id >= stats_nodes.size())
            return;

        ++stats_nodes[record.stats_node_id].pending_gpu_sample_count;
        record.stats_pending_counted = true;
    }

    void drop_gpu_zone_locked(ThreadData& thread_data, uint32_t zone_index)
    {
        if (zone_index >= thread_data.gpu_zones.size())
            return;

        ProfilerGpuZoneRecord& record = thread_data.gpu_zones[zone_index];
        if (record.completed)
            return;

        for (uint32_t child_index : record.child_indices)
            drop_gpu_zone_locked(thread_data, child_index);

        clear_gpu_zone_pending_stats_locked(record);
        record.completed = true;
        record.dropped = true;
    }

    uint32_t try_complete_gpu_zone_locked(ThreadData& thread_data, uint32_t zone_index)
    {
        if (zone_index >= thread_data.gpu_zones.size())
            return kInvalidProfilerId;

        ProfilerGpuZoneRecord& record = thread_data.gpu_zones[zone_index];
        if (record.completed)
            return record.trace_zone_id;
        if (!record.query_ready)
            return kInvalidProfilerId;

        std::vector<uint32_t> child_zone_ids;
        for (uint32_t child_index : record.child_indices) {
            uint32_t child_trace_zone_id = try_complete_gpu_zone_locked(thread_data, child_index);
            if (child_index < thread_data.gpu_zones.size() && !thread_data.gpu_zones[child_index].completed)
                return kInvalidProfilerId;
            if (child_trace_zone_id != kInvalidProfilerId)
                child_zone_ids.push_back(child_trace_zone_id);
        }

        uint32_t trace_zone_id = kInvalidProfilerId;
        if (zone_in_active_trace_window_locked(record.start_timestamp)) {
            trace_zone_id = uint32_t(trace_zones.size());
            trace_zones.push_back({
                .id = trace_zone_id,
                .event_id = record.event_id,
                .timeline_id = record.timeline_id,
                .frame_id = record.frame_id,
                .source_id = record.source_id,
                .name_id = record.name_id,
                .start_timestamp = record.start_timestamp,
                .end_timestamp = record.end_timestamp,
            });
            append_trace_child_range_locked(trace_zones.back(), child_zone_ids);
            if (record.parent_index < 0 && !record.parent_pending)
                trace_root_indices.push_back(trace_zone_id);
        }

        if (profiler->frame_stats_enabled() && record.frame_id != kInvalidProfilerId
            && record.stats_node_id != kInvalidProfilerId && record.stats_node_id < stats_nodes.size()) {
            stats_nodes[record.stats_node_id].gpu_samples.push_back({
                .frame_id = record.frame_id,
                .value_ms = double(record.end_timestamp - record.start_timestamp) / 1000000.0,
            });
        }

        clear_gpu_zone_pending_stats_locked(record);
        record.trace_zone_id = trace_zone_id;
        record.completed = true;
        return trace_zone_id;
    }

    void try_complete_gpu_zone_and_ancestors_locked(ThreadData& thread_data, uint32_t zone_index)
    {
        while (zone_index < thread_data.gpu_zones.size()) {
            const int32_t parent_index = thread_data.gpu_zones[zone_index].parent_index;
            (void)try_complete_gpu_zone_locked(thread_data, zone_index);
            if (!thread_data.gpu_zones[zone_index].completed || parent_index < 0)
                return;
            zone_index = uint32_t(parent_index);
        }
    }

    void store_orphan_gpu_query_result_locked(ThreadData& thread_data, const ProfilerGpuQueryResultEvent& result)
    {
        ProfilerGpuQueryResultEvent& orphan = thread_data.orphan_gpu_query_results[result.ticket.id];
        if (result.status == QueryResultStatus::ready || result.status == QueryResultStatus::invalid
            || orphan.status != QueryResultStatus::ready) {
            orphan = result;
        }
    }

    void apply_gpu_query_result_locked(ThreadData& thread_data, const ProfilerGpuQueryResultEvent& result)
    {
        if (!result.ticket)
            return;

        auto zone_it = thread_data.gpu_zone_by_ticket.find(result.ticket.id);
        if (zone_it == thread_data.gpu_zone_by_ticket.end()) {
            store_orphan_gpu_query_result_locked(thread_data, result);
            return;
        }

        const uint32_t zone_index = zone_it->second;
        if (zone_index >= thread_data.gpu_zones.size())
            return;

        ProfilerGpuZoneRecord& record = thread_data.gpu_zones[zone_index];
        if (record.completed)
            return;

        switch (result.status) {
        case QueryResultStatus::pending:
            mark_gpu_zone_pending_locked(record);
            break;
        case QueryResultStatus::invalid:
            drop_gpu_zone_locked(thread_data, zone_index);
            break;
        case QueryResultStatus::ready:
            record.start_timestamp = result.start_timestamp;
            record.end_timestamp = std::max(result.start_timestamp, result.end_timestamp);
            record.query_ready = true;
            try_complete_gpu_zone_and_ancestors_locked(thread_data, zone_index);
            break;
        }
    }

    void process_gpu_query_result(ThreadData& thread_data, const ProfilerEvent& event)
    {
        std::lock_guard lock(data_mutex);
        apply_gpu_query_result_locked(thread_data, event.gpu_result);
    }

    bool process_events(ThreadData& thread_data)
    {
        bool processed = false;
        ProfilerEvent event;
        while (thread_data.queue.try_dequeue(thread_data.consumer_token, event)) {
            processed = true;
            try {
                switch (event.type) {
                case ProfilerEventType::begin_zone:
                    process_cpu_begin_zone(thread_data, event);
                    break;
                case ProfilerEventType::end_zone:
                    process_cpu_end_zone(thread_data, event);
                    break;
                case ProfilerEventType::gpu_query_result:
                    process_gpu_query_result(thread_data, event);
                    break;
                case ProfilerEventType::begin_frame:
                    process_begin_frame(thread_data, event);
                    break;
                case ProfilerEventType::end_frame:
                    process_end_frame(thread_data, event);
                    break;
                }
            } catch (...) {
            }

            pending_event_count.fetch_sub(1, std::memory_order_acq_rel);
        }
        return processed;
    }

    bool process_threads()
    {
        std::lock_guard process_lock(process_mutex);
        std::vector<ThreadData*> threads;
        {
            std::lock_guard lock(thread_data_mutex);
            threads = thread_data_storage;
        }

        bool processed = false;
        for (ThreadData* thread_data : threads)
            processed |= process_events(*thread_data);
        return processed;
    }

    void worker_main() noexcept
    {
        while (true) {
            const bool processed_thread_events = process_threads();
            if (worker_stop.load(std::memory_order_acquire)
                && pending_event_count.load(std::memory_order_acquire) == 0) {
                return;
            }

            if (!processed_thread_events)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void start_worker()
    {
        worker_thread = std::thread(
            [this]()
            {
                worker_main();
            }
        );
    }

    void flush_events()
    {
        while (pending_event_count.load(std::memory_order_acquire) > 0)
            std::this_thread::yield();
    }

    void stop_worker() noexcept
    {
        worker_stop.store(true, std::memory_order_release);
        if (worker_thread.joinable())
            worker_thread.join();

        try {
            flush_events();
        } catch (...) {
        }
    }

    void start_trace(bool clear)
    {
        std::lock_guard lock(data_mutex);
        const uint64_t now = Timer::now();
        if (clear)
            clear_trace_locked(now);
        trace_active = true;
        trace_start_timestamp = now;
        trace_stop_timestamp = 0;
        trace_epoch_start_timestamp = std::max(trace_epoch_start_timestamp, now);
    }

    void stop_trace()
    {
        std::lock_guard lock(data_mutex);
        trace_stop_timestamp = Timer::now();
        trace_active = false;
    }

    void clear_trace()
    {
        std::lock_guard lock(data_mutex);
        clear_trace_locked(Timer::now());
    }

    void clear_trace_locked(uint64_t timestamp)
    {
        trace_frames.clear();
        trace_zones.clear();
        trace_child_indices.clear();
        trace_root_indices.clear();
        trace_frame_id_by_frame_id.clear();
        trace_epoch_start_timestamp = timestamp;
        if (trace_active)
            trace_start_timestamp = timestamp;
        trace_stop_timestamp = 0;
    }

    void set_stats_window_size(uint32_t size)
    {
        std::lock_guard lock(data_mutex);
        profiler->m_desc.stats_window_size = size;
        profiler->m_stats_window_size = size;
    }

    ref<ProfilerTrace> trace_snapshot()
    {
        std::lock_guard lock(data_mutex);
        ref<ProfilerTrace> trace = make_ref<ProfilerTrace>();
        trace->m_timelines = trace_timelines;
        trace->m_sources = trace_sources;
        trace->m_names = trace_names;
        trace->m_frames = trace_frames;
        trace->m_zones = trace_zones;
        trace->m_child_indices = trace_child_indices;
        trace->m_root_indices = trace_root_indices;
        return trace;
    }

    ProfilerStatValue summarize_samples_locked(const std::vector<StatsSample>& samples) const
    {
        ProfilerStatValue value;
        if (samples.empty())
            return value;

        const uint32_t window_size = profiler->stats_window_size();
        uint32_t min_frame_id = 0;
        if (completed_frame_count > window_size)
            min_frame_id = newest_completed_frame_id - window_size + 1;

        double sum = 0.0;
        double sum_sq = 0.0;
        for (const StatsSample& sample : samples) {
            if (sample.frame_id == kInvalidProfilerId || sample.frame_id < min_frame_id)
                continue;
            if (!value.valid) {
                value.min_ms = sample.value_ms;
                value.max_ms = sample.value_ms;
                value.valid = true;
            } else {
                value.min_ms = std::min(value.min_ms, sample.value_ms);
                value.max_ms = std::max(value.max_ms, sample.value_ms);
            }
            value.last_ms = sample.value_ms;
            sum += sample.value_ms;
            sum_sq += sample.value_ms * sample.value_ms;
            ++value.sample_count;
        }

        if (value.sample_count == 0) {
            value = {};
            return value;
        }

        value.average_ms = sum / double(value.sample_count);
        const double variance
            = std::max(0.0, sum_sq / double(value.sample_count) - value.average_ms * value.average_ms);
        value.stddev_ms = std::sqrt(variance);
        return value;
    }

    ref<ProfilerStats> stats_snapshot()
    {
        std::lock_guard lock(data_mutex);
        ref<ProfilerStats> stats = make_ref<ProfilerStats>();
        stats->m_sources = trace_sources;
        stats->m_names = trace_names;
        stats->m_completed_frame_count = completed_frame_count;
        stats->m_window_size = profiler->stats_window_size();

        stats->m_nodes.reserve(stats_nodes.size());
        for (const StatsNodeState& node : stats_nodes) {
            const uint32_t child_begin = uint32_t(stats->m_child_indices.size());
            stats->m_child_indices.insert(stats->m_child_indices.end(), node.children.begin(), node.children.end());

            stats->m_nodes.push_back({
                .id = node.id,
                .parent_id = node.parent_id,
                .child_index_begin = child_begin,
                .child_index_count = uint32_t(node.children.size()),
                .source_id = node.source_id,
                .name_id = node.name_id,
                .pending_gpu_sample_count = node.pending_gpu_sample_count,
                .cpu = summarize_samples_locked(node.cpu_samples),
                .gpu = summarize_samples_locked(node.gpu_samples),
            });
        }

        return stats;
    }

    // ------------------------------------------------------------------------
    // GPU queries
    // ------------------------------------------------------------------------

    GpuContext& get_or_create_gpu_context_locked(Device* device, CommandQueueType queue)
    {
        GpuContextKey key{
            .device = device,
            .queue = queue,
        };
        auto it = gpu_contexts.find(key);
        if (it != gpu_contexts.end())
            return it->second;

        GpuContext context;
        context.device = ref<Device>(device);
        context.queue = queue;
        context.timeline_info.type = ProfilerTimelineType::gpu;
        context.timeline_info.name = fmt::format("GPU {} {}", device->info().adapter_name, enum_to_string(queue));
        context.timeline_info.device_id = uint64_t(reinterpret_cast<uintptr_t>(device));
        context.timeline_info.queue = queue;
        {
            std::lock_guard data_lock(data_mutex);
            context.timeline_id = uint32_t(trace_timelines.size());
            trace_timelines.push_back({
                .id = context.timeline_id,
                .type = context.timeline_info.type,
                .name = context.timeline_info.name,
                .thread_id = context.timeline_info.thread_id,
                .device_id = context.timeline_info.device_id,
                .queue = context.timeline_info.queue,
            });
        }

        auto [inserted_it, inserted] = gpu_contexts.emplace(key, std::move(context));
        SGL_ASSERT(inserted);
        return inserted_it->second;
    }

    void clear_gpu_query_cache(GpuContext* context = nullptr) noexcept
    {
        if (s_gpu_query_cache.profiler != this)
            return;
        if (context && s_gpu_query_cache.context != context)
            return;
        s_gpu_query_cache = {};
    }

    ProfilerGpuQueryPair allocate_gpu_query_pair_fast(
        Device* device,
        CommandQueueType queue,
        ThreadData* owner_thread,
        uint64_t command_stream_id
    ) noexcept
    {
        GpuQueryThreadCache& cache = s_gpu_query_cache;
        if (cache.profiler != this || !cache.context || !cache.next_slot)
            return {};
        if (cache.context->device.get() != device || cache.context->queue != queue)
            return {};
        if (cache.context->disabled || device->is_closed())
            return {};
        if (cache.next_slot >= cache.end_slot)
            return {};

        GpuQuerySlot* slot = cache.next_slot++;
        slot->owner_thread = owner_thread;
        slot->command_stream_id = command_stream_id;
        slot->closed = false;
        slot->pending_emitted = false;
        slot->completed = false;

        ProfilerGpuQueryPair pair{
            .slot = slot,
            .query_pool = slot->query_pool,
            .begin_query_index = slot->begin_query_index,
            .end_query_index = slot->end_query_index,
            .ticket = slot->ticket,
            .command_stream_id = command_stream_id,
            .timeline_id = cache.context->timeline_id,
            .queue = queue,
        };
        return pair;
    }

    ProfilerGpuQueryChunk acquire_gpu_query_chunk_locked(GpuContext& context)
    {
        if (context.disabled || context.device->is_closed())
            return {};

        const uint32_t chunk_size = profiler->desc().gpu_query_block_size;
        const uint32_t pool_size = profiler->desc().gpu_query_pool_size;
        SGL_ASSERT(chunk_size >= 2 && chunk_size % 2 == 0);
        SGL_ASSERT(pool_size >= chunk_size && pool_size % 2 == 0);

        while (true) {
            if (context.query_pages.empty() || context.query_pages.back().allocated_query_count + 2 > pool_size) {
                if (context.query_pages.size() >= kMaxGpuQueryPoolPagesPerContext) {
                    context.disabled = true;
                    log_warn_once(
                        "Profiler GPU timestamp query page capacity exhausted for device/queue; GPU profiling is "
                        "disabled for that context."
                    );
                    return {};
                }

                ProfilerGpuQueryPage page{
                    .query_pool = context.device->create_query_pool({
                        .type = QueryType::timestamp,
                        .count = pool_size,
                    }),
                };
                page.slots.reserve(pool_size / 2);
                context.query_pages.push_back(std::move(page));
            }

            ProfilerGpuQueryPage& page = context.query_pages.back();
            const uint32_t first_query_index = page.allocated_query_count;
            uint32_t query_count = std::min(chunk_size, pool_size - first_query_index);
            query_count -= query_count % 2;
            if (query_count >= 2) {
                const size_t first_slot_index = page.slots.size();
                const uint32_t slot_count = query_count / 2;
                for (uint32_t i = 0; i < slot_count; ++i) {
                    const uint32_t begin_query_index = first_query_index + i * 2;
                    page.slots.push_back({
                        .ticket = {.id = next_gpu_query_ticket++},
                        .query_pool = page.query_pool,
                        .begin_query_index = begin_query_index,
                        .end_query_index = begin_query_index + 1,
                        .timeline_id = context.timeline_id,
                        .queue = context.queue,
                    });
                }
                page.allocated_query_count += query_count;
                return {
                    .next_slot = page.slots.data() + first_slot_index,
                    .end_slot = page.slots.data() + first_slot_index + slot_count,
                };
            }

            page.allocated_query_count = pool_size;
        }
    }

    ProfilerGpuQueryPair allocate_gpu_query_pair_slow(CommandEncoder* encoder, ThreadData* owner_thread)
    {
        SGL_CHECK_NOT_NULL(encoder);
        Device* device = encoder->device();
        SGL_CHECK_NOT_NULL(device);
        if (device->is_closed())
            return {};
        if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
            return {};

        const CommandQueueType queue = encoder->queue();
        const uint64_t command_stream_id = encoder->recording_id();
        std::lock_guard lock(gpu_mutex);
        GpuContext& context = get_or_create_gpu_context_locked(device, queue);
        if (ProfilerGpuQueryPair pair = allocate_gpu_query_pair_fast(device, queue, owner_thread, command_stream_id);
            pair.query_pool)
            return pair;

        ProfilerGpuQueryChunk chunk = acquire_gpu_query_chunk_locked(context);
        if (!chunk.next_slot)
            return {};

        s_gpu_query_cache = {
            .profiler = this,
            .context = &context,
            .next_slot = chunk.next_slot,
            .end_slot = chunk.end_slot,
        };
        return allocate_gpu_query_pair_fast(device, queue, owner_thread, command_stream_id);
    }

    ProfilerGpuQueryPair allocate_gpu_query_pair(CommandEncoder* encoder, ThreadData* owner_thread)
    {
        SGL_CHECK_NOT_NULL(encoder);
        Device* device = encoder->device();
        const CommandQueueType queue = encoder->queue();
        const uint64_t command_stream_id = encoder->recording_id();
        if (ProfilerGpuQueryPair pair = allocate_gpu_query_pair_fast(device, queue, owner_thread, command_stream_id);
            pair.query_pool)
            return pair;
        return allocate_gpu_query_pair_slow(encoder, owner_thread);
    }

    static GpuTimestampAnchor capture_gpu_timestamp_anchor(Device* device, CommandQueueType queue)
    {
        GpuTimestampAnchor anchor;
        const uint64_t before_ns = Timer::now();
        TimestampCalibration calibration = device->get_timestamp_calibration(queue);
        const uint64_t after_ns = Timer::now();

        anchor.valid = calibration.gpu_frequency > 0;
        anchor.cpu_timestamp_ns = before_ns + (after_ns - before_ns) / 2;
        anchor.gpu_timestamp = calibration.gpu_timestamp;
        anchor.gpu_frequency = calibration.gpu_frequency;
        return anchor;
    }

    static uint64_t gpu_timestamp_to_cpu_ns(uint64_t timestamp, const GpuTimestampAnchor& anchor)
    {
        SGL_ASSERT(anchor.valid);

        const long double delta_ticks
            = static_cast<long double>(timestamp) - static_cast<long double>(anchor.gpu_timestamp);
        const long double delta_ns = delta_ticks * 1000000000.0L / static_cast<long double>(anchor.gpu_frequency);
        const long double timestamp_ns = static_cast<long double>(anchor.cpu_timestamp_ns) + delta_ns;

        if (timestamp_ns <= 0.0L)
            return 0;
        if (timestamp_ns >= static_cast<long double>(std::numeric_limits<uint64_t>::max()))
            return std::numeric_limits<uint64_t>::max();
        return static_cast<uint64_t>(timestamp_ns + 0.5L);
    }

    void close_gpu_query_slot(GpuQuerySlot* slot) noexcept
    {
        if (!slot || slot->closed)
            return;

        slot->closed = true;
        if (!closed_gpu_query_slot_queue.enqueue(slot))
            std::terminate();
    }

    void queue_gpu_query_result(
        GpuQuerySlot* slot,
        QueryResultStatus status,
        uint64_t start_timestamp = 0,
        uint64_t end_timestamp = 0
    ) noexcept
    {
        if (!slot || !slot->owner_thread)
            return;

        queue_thread_event(
            slot->owner_thread,
            {
                .type = ProfilerEventType::gpu_query_result,
                .gpu_result = {
                    .ticket = slot->ticket,
                    .status = status,
                    .start_timestamp = start_timestamp,
                    .end_timestamp = end_timestamp,
                },
            }
        );
    }

    bool process_gpu_query_slot(GpuQuerySlot* slot, std::map<GpuContextKey, GpuTimestampAnchor>& anchors)
    {
        if (!slot || slot->completed)
            return true;

        if (!slot->query_pool || !slot->owner_thread || slot->query_pool->device()->is_closed()) {
            queue_gpu_query_result(slot, QueryResultStatus::invalid);
            slot->completed = true;
            return true;
        }

        QueryResultStatus status = QueryResultStatus::invalid;
        try {
            status = slot->query_pool->result_status(slot->begin_query_index, 2);
        } catch (...) {
            status = QueryResultStatus::invalid;
        }

        switch (status) {
        case QueryResultStatus::invalid:
            queue_gpu_query_result(slot, QueryResultStatus::invalid);
            slot->completed = true;
            return true;
        case QueryResultStatus::pending:
            if (!slot->pending_emitted) {
                queue_gpu_query_result(slot, QueryResultStatus::pending);
                slot->pending_emitted = true;
            }
            return false;
        case QueryResultStatus::ready:
            break;
        }

        uint64_t timestamps[2] = {};
        try {
            slot->query_pool->get_results(slot->begin_query_index, 2, timestamps);
        } catch (...) {
            queue_gpu_query_result(slot, QueryResultStatus::invalid);
            slot->completed = true;
            return true;
        }

        Device* device = slot->query_pool->device();
        GpuContextKey key{
            .device = device,
            .queue = slot->queue,
        };
        auto anchor_it = anchors.find(key);
        if (anchor_it == anchors.end()) {
            GpuTimestampAnchor anchor;
            try {
                anchor = capture_gpu_timestamp_anchor(device, slot->queue);
            } catch (...) {
                anchor = {};
            }
            anchor_it = anchors.emplace(key, anchor).first;
        }

        if (!anchor_it->second.valid) {
            queue_gpu_query_result(slot, QueryResultStatus::invalid);
            slot->completed = true;
            return true;
        }

        const uint64_t start_timestamp = gpu_timestamp_to_cpu_ns(timestamps[0], anchor_it->second);
        const uint64_t end_timestamp
            = std::max(start_timestamp, gpu_timestamp_to_cpu_ns(timestamps[1], anchor_it->second));
        queue_gpu_query_result(slot, QueryResultStatus::ready, start_timestamp, end_timestamp);
        slot->completed = true;
        return true;
    }

    void process_gpu_query_slots()
    {
        std::lock_guard lock(gpu_query_process_mutex);

        GpuQuerySlot* closed_slot = nullptr;
        while (closed_gpu_query_slot_queue.try_dequeue(closed_slot)) {
            if (closed_slot && !closed_slot->completed)
                pending_gpu_query_slots.push_back(closed_slot);
        }

        std::map<GpuContextKey, GpuTimestampAnchor> anchors;
        pending_gpu_query_slots.erase(
            std::remove_if(
                pending_gpu_query_slots.begin(),
                pending_gpu_query_slots.end(),
                [this, &anchors](GpuQuerySlot* slot)
                {
                    return process_gpu_query_slot(slot, anchors);
                }
            ),
            pending_gpu_query_slots.end()
        );
    }

    SGL_NON_COPYABLE_AND_MOVABLE(ProfilerImpl);
};

// ----------------------------------------------------------------------------
// Current profiler stack
// ----------------------------------------------------------------------------

namespace {

    std::mutex s_current_profiler_stack_mutex;
    std::vector<Profiler*> s_current_profiler_stack;
    std::atomic<Profiler*> s_current_profiler{nullptr};

    void publish_current_profiler_locked() noexcept
    {
        Profiler* profiler = s_current_profiler_stack.empty() ? nullptr : s_current_profiler_stack.back();
        s_current_profiler.store(profiler, std::memory_order_release);
    }

    void remove_current_profiler_entries(Profiler* profiler) noexcept
    {
        std::lock_guard lock(s_current_profiler_stack_mutex);
        auto new_end = std::remove(s_current_profiler_stack.begin(), s_current_profiler_stack.end(), profiler);
        s_current_profiler_stack.erase(new_end, s_current_profiler_stack.end());
        publish_current_profiler_locked();
    }

    void validate_profiler_desc(const ProfilerDesc& desc)
    {
        SGL_CHECK(desc.stats_window_size >= 1, "ProfilerDesc.stats_window_size must be at least 1.");
        SGL_CHECK(desc.gpu_query_pool_size > 0, "ProfilerDesc.gpu_query_pool_size must be positive.");
        SGL_CHECK(desc.gpu_query_block_size > 0, "ProfilerDesc.gpu_query_block_size must be positive.");
        SGL_CHECK(
            desc.gpu_query_pool_size % 2 == 0,
            "ProfilerDesc.gpu_query_pool_size must be an even number of timestamp queries."
        );
        SGL_CHECK(
            desc.gpu_query_block_size % 2 == 0,
            "ProfilerDesc.gpu_query_block_size must be an even number of timestamp queries."
        );
        SGL_CHECK(
            desc.gpu_query_block_size <= desc.gpu_query_pool_size,
            "ProfilerDesc.gpu_query_block_size must be less than or equal to gpu_query_pool_size."
        );
    }

} // namespace

// ----------------------------------------------------------------------------
// Profiler
// ----------------------------------------------------------------------------

Profiler::Profiler(ProfilerDesc desc)
    : m_desc(std::move(desc))
{
    validate_profiler_desc(m_desc);
    m_auto_zones_enabled = m_desc.auto_zones_enabled;
    m_debug_groups_enabled = m_desc.debug_groups_enabled;
    m_frame_stats_enabled = m_desc.frame_stats_enabled;
    m_stats_window_size = m_desc.stats_window_size;
    m_impl = new ProfilerImpl(this);
    push_current_profiler(this);
}

Profiler::~Profiler()
{
    remove_current_profiler_entries(this);
    flush();
    delete m_impl;
}

const ProfilerSourceLocation*
Profiler::intern_source_location(std::string_view file, uint32_t line, std::string_view function)
{
    return SourceLocationRegistry::get().intern(file, line, function);
}

const char* Profiler::intern_name(std::string_view name)
{
    return StringRegistry::get().intern(name);
}

void Profiler::set_stats_window_size(uint32_t size)
{
    SGL_CHECK(size >= 1, "stats_window_size must be at least 1.");
    m_impl->set_stats_window_size(size);
}

void Profiler::start_trace(bool clear)
{
    m_impl->start_trace(clear);
}

void Profiler::stop_trace()
{
    m_impl->stop_trace();
}

void Profiler::clear_trace()
{
    m_impl->clear_trace();
}

ref<ProfilerTrace> Profiler::trace_snapshot()
{
    tick();
    flush();
    return m_impl->trace_snapshot();
}

ref<ProfilerStats> Profiler::stats_snapshot()
{
    tick();
    return m_impl->stats_snapshot();
}

ProfilerZoneToken Profiler::begin_zone(
    const ProfilerSourceLocation* source_location,
    const char* name,
    CommandEncoder* encoder,
    ProfilerZoneFlags flags
) noexcept
{
    ProfilerZoneToken token;

    if (!m_enabled)
        return token;

    if (!source_location) {
        SGL_ASSERT(source_location);
        return token;
    }

    if (name && is_set(flags, ProfilerZoneFlags::copy_name))
        name = Profiler::intern_name(name);

    const uint64_t timestamp = Timer::now();
    ThreadData* thread_data = m_impl->get_this_thread_data();
    token.profiler = this;
    token.thread_data = thread_data;
    token.encoder = encoder;
    token.gpu_query_slot = nullptr;
    token.debug_group_active = false;

    ProfilerGpuEvent gpu_event;

    if (encoder) {
        ProfilerGpuQueryPair query_pair = m_impl->allocate_gpu_query_pair(encoder, thread_data);
        if (query_pair.query_pool) {
            encoder->write_timestamp(query_pair.query_pool.get(), query_pair.begin_query_index);

            gpu_event.ticket = query_pair.ticket;
            gpu_event.command_stream_id = query_pair.command_stream_id;
            gpu_event.timeline_id = query_pair.timeline_id;
            gpu_event.queue = query_pair.queue;

            token.gpu_query_slot = query_pair.slot;
        }
    }

    m_impl->queue_thread_event(
        thread_data,
        {
            .type = ProfilerEventType::begin_zone,
            .timestamp = timestamp,
            .source_location = source_location,
            .name = name,
            .gpu = std::move(gpu_event),
        }
    );

    if (encoder) {
        const char* debug_name = name ? name : fallback_zone_name(source_location);
        if (is_set(flags, ProfilerZoneFlags::debug_group)) {
            encoder->push_debug_group(debug_name, float3(0.5f));
            token.debug_group_active = true;
        }
    }

    return token;
}

void Profiler::end_zone(ProfilerZoneToken token) noexcept
{
    if (!token.profiler)
        return;

    SGL_ASSERT(token.profiler == this);
    ThreadData* thread_data = static_cast<ThreadData*>(token.thread_data);
    SGL_ASSERT(thread_data != nullptr);

    const uint64_t timestamp = Timer::now();

    if (token.gpu_query_slot) {
        GpuQuerySlot* slot = static_cast<GpuQuerySlot*>(token.gpu_query_slot);
        token.encoder->write_timestamp(slot->query_pool.get(), slot->end_query_index);
        m_impl->close_gpu_query_slot(slot);
    }

    if (token.debug_group_active) {
        token.encoder->pop_debug_group();
    }

    m_impl->queue_thread_event(
        thread_data,
        {
            .type = ProfilerEventType::end_zone,
            .timestamp = timestamp,
        }
    );
}

ProfilerFrameToken Profiler::begin_frame(const ProfilerSourceLocation* source_location, const char* name) noexcept
{
    ProfilerFrameToken token;

    if (!m_enabled)
        return token;
    if (!source_location) {
        SGL_ASSERT(source_location);
        return token;
    }

    const uint64_t timestamp = Timer::now();
    ThreadData* thread_data = m_impl->get_this_thread_data();
    if (thread_data->hot_frame_active) {
        log_warn_once("Profiler ignored an overlapping begin_frame() call on one thread.");
        return token;
    }

    token.profiler = this;
    token.thread_data = thread_data;

    m_impl->queue_thread_event(
        thread_data,
        {
            .type = ProfilerEventType::begin_frame,
            .timestamp = timestamp,
            .source_location = source_location,
            .name = name,
        }
    );

    thread_data->hot_frame_active = true;
    return token;
}

void Profiler::end_frame(ProfilerFrameToken token) noexcept
{
    if (!token.profiler)
        return;

    SGL_ASSERT(token.profiler == this);
    ThreadData* thread_data = static_cast<ThreadData*>(token.thread_data);
    SGL_ASSERT(thread_data != nullptr);

    if (!thread_data->hot_frame_active) {
        log_warn_once("Profiler ignored an end_frame() call without an active frame on this thread.");
        return;
    }
    thread_data->hot_frame_active = false;

    const uint64_t timestamp = Timer::now();
    m_impl->queue_thread_event(
        thread_data,
        {
            .type = ProfilerEventType::end_frame,
            .timestamp = timestamp,
        }
    );
}

void Profiler::tick()
{
    m_impl->process_gpu_query_slots();
}

void Profiler::flush()
{
    tick();
    m_impl->flush_events();
    m_impl->process_gpu_query_slots();
    m_impl->flush_events();
}

std::string Profiler::to_string() const
{
    return fmt::format(
        "Profiler(enabled = {}, frame_stats_enabled = {}, auto_zones_enabled = {}, debug_groups_enabled = {}, "
        "stats_window_size = {})",
        m_enabled.load(),
        m_frame_stats_enabled.load(),
        m_auto_zones_enabled.load(),
        m_debug_groups_enabled.load(),
        m_stats_window_size.load()
    );
}

Profiler* current_profiler_or_null()
{
    return s_current_profiler.load(std::memory_order_acquire);
}

Profiler* current_profiler()
{
    Profiler* profiler = current_profiler_or_null();
    SGL_CHECK(profiler, "No current profiler. Create a Profiler or use push_current_profiler() to set one.");
    return profiler;
}

void push_current_profiler(Profiler* profiler)
{
    SGL_CHECK(profiler != nullptr, "Cannot push a null profiler.");

    std::lock_guard lock(s_current_profiler_stack_mutex);
    s_current_profiler_stack.push_back(profiler);
    publish_current_profiler_locked();
}

Profiler* pop_current_profiler()
{
    std::lock_guard lock(s_current_profiler_stack_mutex);

    SGL_CHECK(
        !s_current_profiler_stack.empty(),
        "No profiler to pop. push_current_profiler()/pop_current_profiler() mismatch."
    );

    Profiler* profiler = s_current_profiler_stack.back();
    s_current_profiler_stack.pop_back();
    publish_current_profiler_locked();
    return profiler;
}

} // namespace sgl
