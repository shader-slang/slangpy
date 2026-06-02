// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler.h"

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/hash.h"
#include "sgl/core/short_vector.h"
#include "sgl/core/timer.h"

#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/query.h"

#include <concurrentqueue.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sgl {

// ----------------------------------------------------------------------------
// StringRegistry
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

// ----------------------------------------------------------------------------
// SourceLocationRegistry
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

enum class ProfilerEventType : uint16_t {
    begin_zone,
    end_zone,
    begin_gpu_zone,
    end_gpu_zone,
    begin_frame,
    end_frame,
};

struct ProfilerEventBeginZone {
    uint64_t timestamp;
    const ProfilerSourceLocation* source_location;
    const char* name;
};

struct ProfilerEventEndZone {
    uint64_t timestamp;
};

struct GpuContextData;

struct ProfilerEventBeginGpuZone {
    uint64_t timestamp;
    const ProfilerSourceLocation* source_location;
    const char* name;
    GpuContextData* gpu_context_data;
    uint32_t query_index;
};

struct ProfilerEventEndGpuZone {
    uint64_t timestamp;
    GpuContextData* gpu_context_data;
    uint32_t query_index;
};

struct ProfilerEventBeginFrame {
    uint64_t timestamp;
    const ProfilerSourceLocation* source_location;
    const char* name;
};

struct ProfilerEventEndFrame {
    uint64_t timestamp;
};

struct ProfilerEvent {
    ProfilerEventType type;
    union {
        ProfilerEventBeginZone begin_zone;
        ProfilerEventEndZone end_zone;
        ProfilerEventBeginGpuZone begin_gpu_zone;
        ProfilerEventEndGpuZone end_gpu_zone;
        ProfilerEventBeginFrame begin_frame;
        ProfilerEventEndFrame end_frame;
    };
};

class ProfilerTraceStorage {
public:
    static constexpr size_t CHUNK_SIZE = 64 * 1024 * 1024;

    ~ProfilerTraceStorage()
    {
        for (uint8_t* chunk : m_chunks)
            delete[] chunk;
    }

    template<typename T>
    T* allocate(size_t count = 1)
    {
        static_assert(
            std::is_default_constructible_v<T>,
            "ProfilerTraceStorage can only allocate default constructible types."
        );
        static_assert(
            std::is_trivially_destructible_v<T>,
            "ProfilerTraceStorage can only allocate trivially destructible types."
        );
        size_t size = sizeof(T) * count;
        if (size > m_chunk_remaining) {
            size_t chunk_size = std::max(size, CHUNK_SIZE);
            uint8_t* chunk = new uint8_t[chunk_size];
            m_chunks.push_back(chunk);
            m_chunk_pos = chunk;
            m_chunk_remaining = chunk_size;
        }
        T* ptr = reinterpret_cast<T*>(m_chunk_pos);
        new (ptr) T[count]();
        m_chunk_pos += size;
        m_chunk_remaining -= size;
        return ptr;
    }

private:
    std::vector<uint8_t*> m_chunks;
    uint8_t* m_chunk_pos{nullptr};
    size_t m_chunk_remaining{0};
};

struct ThreadData {
    Profiler* profiler{nullptr};
    std::thread::id thread_id;
    ProfilerTraceStorage* trace_storage{nullptr};

    ProfilerTimelineInfo timeline_info;

    moodycamel::ConcurrentQueue<ProfilerEvent> queue;
    moodycamel::ProducerToken producer_token{queue};
    moodycamel::ConsumerToken consumer_token{queue};

    short_vector<ProfilerZone*> zone_stack;
    short_vector<short_vector<ProfilerZone*>> zone_children_stack;

    std::vector<const ProfilerZone*> zones;

    uint32_t frame_id{0};
    ProfilerFrame* current_frame{nullptr};
    std::vector<const ProfilerFrame*> frames;

    ThreadData(Profiler* profiler_, std::thread::id thread_id_, ProfilerTraceStorage* trace_storage_)
        : profiler(profiler_)
        , thread_id(thread_id_)
        , trace_storage(trace_storage_)
    {
        timeline_info.type = ProfilerTimelineType::cpu;
        timeline_info.name = fmt::format("CPU Thread {}", std::hash<std::thread::id>{}(thread_id));
        timeline_info.thread_id = std::hash<std::thread::id>{}(thread_id);
    }

    void queue_event(ProfilerEvent&& event) { queue.enqueue(producer_token, event); }
};

/// Context data for a single GPU device/queue combination.
/// Manages a pool of timestamp queries for GPU profiling.
struct GpuContextData {
    Profiler* profiler{nullptr};
    Device* device{nullptr};
    CommandQueueType queue{CommandQueueType::graphics};

    ProfilerTimelineInfo timeline_info;

    ref<QueryPool> query_pool;
    std::atomic<uint32_t> next_query_index{0};
    std::atomic<uint32_t> query_head{0};
    std::atomic<uint32_t> query_tail{0};

    struct QueryBlock {
        uint32_t offset{0};
        uint32_t count{0};

        CommandRecordingID recording_id{0};
    };

    GpuContextData(Profiler* profiler_, Device* device_, CommandQueueType queue_)
        : profiler(profiler_)
        , device(device_)
        , queue(queue_)
    {
        timeline_info.type = ProfilerTimelineType::gpu;
        timeline_info.name = fmt::format("GPU {} {}", device->info().adapter_name, enum_to_string(queue));
        timeline_info.device_id = uint64_t(reinterpret_cast<uintptr_t>(device));
        timeline_info.queue = queue;

        query_pool = device->create_query_pool({
            .type = QueryType::timestamp,
            .count = 16 * 1024,
        });
        next_query_index = 0;
    }

    static constexpr uint32_t INVALID_QUERY_INDEX = std::numeric_limits<uint32_t>::max();

    uint32_t allocate_timestamp_query(CommandEncoder* encoder)
    {
        SGL_UNUSED(encoder);
        return INVALID_QUERY_INDEX;
    }
};

struct GpuRecordingData {
    CommandRecordingID recording_id{0};
};

struct ProfilerImpl;

namespace {

    constexpr uint32_t kGpuTimestampQueryPoolSize = 64 * 1024;
    constexpr uint32_t kGpuTimestampQueriesPerBlock = 256;
    static_assert(kGpuTimestampQueryPoolSize % kGpuTimestampQueriesPerBlock == 0);

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

    struct ProfilerGpuQueryBlock {
        ref<QueryPool> query_pool;
        uint32_t first_query_index{0};
        uint32_t query_count{0};
    };

    struct GpuContext {
        ProfilerTimelineInfo timeline_info;
        std::vector<const ProfilerZone*> zones;
        std::vector<ref<QueryPool>> query_pools;
        std::vector<ProfilerGpuQueryBlock> free_query_blocks;
        uint32_t next_query_index{0};
    };

    struct ProfilerGpuQueryPair {
        ref<QueryPool> query_pool;
        uint32_t begin_query_index{0};
        uint32_t end_query_index{0};
    };

    struct ProfilerGpuZoneRecord {
        const ProfilerSourceLocation* source_location{nullptr};
        const char* name{nullptr};
        ref<QueryPool> query_pool;
        uint32_t begin_query_index{0};
        uint32_t end_query_index{0};
        int32_t parent_index{-1};
        std::vector<uint32_t> child_indices;
        bool completed{false};
    };

    struct ProfilerGpuRecordingBatch {
        ref<Device> device;
        CommandQueueType queue{CommandQueueType::graphics};
        std::vector<ProfilerGpuZoneRecord> zones;
        std::vector<ProfilerGpuQueryBlock> query_blocks;
    };

    struct ActiveGpuRecording {
        ref<Device> device;
        CommandQueueType queue{CommandQueueType::graphics};
        std::vector<ProfilerGpuZoneRecord> zones;
        std::vector<ProfilerGpuQueryBlock> query_blocks;
        ref<QueryPool> current_query_pool;
        uint32_t next_query_index{0};
        uint32_t current_query_end_index{0};
        std::vector<uint32_t> zone_stack;
    };

    struct PendingGpuSubmit {
        ref<Device> device;
        CommandQueueType queue{CommandQueueType::graphics};
        uint64_t submit_id{0};
        GpuTimestampAnchor anchor;
        std::vector<ProfilerGpuRecordingBatch> batches;
    };

    struct GpuDeviceCallbackRegistration {
        ref<Device> device;
        CommandRecordingCallbackID submitted_callback_id{0};
        CommandRecordingCallbackID discarded_callback_id{0};
    };

    struct GpuRecordingThreadCache {
        ProfilerImpl* profiler{nullptr};
        CommandRecordingID recording_id{0};
        ActiveGpuRecording* recording{nullptr};
    };

    thread_local GpuRecordingThreadCache s_gpu_recording_cache;

} // namespace

// ----------------------------------------------------------------------------
// ProfilerTrace
// ----------------------------------------------------------------------------

namespace {

    void write_json_string(std::ostream& stream, const char* value)
    {
        stream << '"';

        const char* chunk_begin = value;
        const char* it = value;
        while (*it) {
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

    void update_base_timestamp(uint64_t& base_timestamp, bool& has_base_timestamp, const ProfilerZone* zone)
    {
        if (!has_base_timestamp || zone->start_timestamp < base_timestamp) {
            base_timestamp = zone->start_timestamp;
            has_base_timestamp = true;
        }

        for (const ProfilerZone* child : zone->children)
            update_base_timestamp(base_timestamp, has_base_timestamp, child);
    }

    void write_trace_event_separator(std::ostream& stream, bool& first_event)
    {
        if (first_event)
            first_event = false;
        else
            stream << ',';
    }

    const char* timeline_type_name(ProfilerTimelineType type)
    {
        switch (type) {
        case ProfilerTimelineType::cpu:
            return "cpu";
        case ProfilerTimelineType::gpu:
            return "gpu";
        default:
            return "unknown";
        }
    }

    const char* timeline_category(ProfilerTimelineType type)
    {
        switch (type) {
        case ProfilerTimelineType::cpu:
            return "sgl.cpu";
        case ProfilerTimelineType::gpu:
            return "sgl.gpu";
        default:
            return "sgl";
        }
    }

    void write_trace_metadata_event(
        std::ostream& stream,
        size_t timeline_index,
        const ProfilerTimelineInfo& metadata,
        bool& first_event
    )
    {
        const std::string default_name = fmt::format("Timeline {}", timeline_index);
        const std::string& name = metadata.name.empty() ? default_name : metadata.name;

        write_trace_event_separator(stream, first_event);
        stream << "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":0,\"tid\":" << timeline_index
               << ",\"args\":{\"name\":";
        write_json_string(stream, name.c_str());
        stream << "}}";
    }

    void write_trace_zone(
        std::ostream& stream,
        const ProfilerZone* zone,
        const ProfilerTimelineInfo& metadata,
        size_t timeline_index,
        uint64_t base_timestamp,
        bool& first_event
    )
    {
        write_trace_event_separator(stream, first_event);

        const ProfilerSourceLocation* source_location = zone->source_location;
        const char* name = zone->name
            ? zone->name
            : ((source_location && source_location->function) ? source_location->function : "zone");

        stream << "{\"ph\":\"X\",\"cat\":";
        write_json_string(stream, timeline_category(metadata.type));
        stream << ",\"name\":";
        write_json_string(stream, name);
        stream << ",\"pid\":0,\"tid\":" << timeline_index << ",\"ts\":";
        write_trace_time_us(stream, zone->start_timestamp - base_timestamp);
        stream << ",\"dur\":";
        write_trace_time_us(stream, zone->end_timestamp - zone->start_timestamp);
        stream << ",\"args\":{\"start_timestamp_ns\":" << zone->start_timestamp
               << ",\"end_timestamp_ns\":" << zone->end_timestamp << ",\"timeline_type\":";
        write_json_string(stream, timeline_type_name(metadata.type));

        if (source_location) {
            stream << ",\"source_file\":";
            write_json_string(stream, source_location->file ? source_location->file : "");
            stream << ",\"source_line\":" << source_location->line << ",\"source_function\":";
            write_json_string(stream, source_location->function ? source_location->function : "");
        }

        stream << "}}";

        for (const ProfilerZone* child : zone->children)
            write_trace_zone(stream, child, metadata, timeline_index, base_timestamp, first_event);
    }

} // namespace

void ProfilerTrace::write_to_json(const std::filesystem::path& path) const
{
    std::ofstream stream(path, std::ios::out | std::ios::binary);
    SGL_CHECK(stream.good(), "{}: failed to open profiler trace JSON for writing", path);

    uint64_t base_timestamp = 0;
    bool has_base_timestamp = false;
    for (const Timeline& timeline : m_timelines) {
        for (const ProfilerZone* zone : timeline.zones)
            update_base_timestamp(base_timestamp, has_base_timestamp, zone);
    }

    stream << "{\"traceEvents\":[";
    bool first_event = true;
    for (size_t timeline_index = 0; timeline_index < m_timelines.size(); ++timeline_index) {
        const Timeline& timeline = m_timelines[timeline_index];
        write_trace_metadata_event(stream, timeline_index, timeline.info, first_event);
        for (const ProfilerZone* zone : timeline.zones) {
            write_trace_zone(stream, zone, timeline.info, timeline_index, base_timestamp, first_event);
            stream << "\n";
        }
    }
    stream << "],\"displayTimeUnit\":\"ns\"}";

    SGL_CHECK(stream.good(), "{}: failed to write profiler trace JSON", path);
}

// ----------------------------------------------------------------------------
// ProfilerImpl
// ----------------------------------------------------------------------------

thread_local ThreadData* s_thread_data{nullptr};
thread_local GpuContextData* s_gpu_context_data{nullptr};

struct ProfilerImpl {

    Profiler* profiler{nullptr};

    std::shared_ptr<ProfilerTraceStorage> trace_storage;

    // Thread data.
    short_vector<ThreadData*> thread_data_storage;
    std::unordered_map<std::thread::id, ThreadData*> thread_data_by_id;
    std::mutex thread_data_mutex;

    // GPU context data.
    short_vector<GpuContextData*> gpu_context_data_storage;
    std::map<std::pair<Device*, CommandQueueType>, GpuContextData*> gpu_context_data_by_key;
    std::mutex gpu_context_data_mutex;

    // GPU data.
    std::map<GpuContextKey, GpuContext> gpu_contexts;
    std::map<CommandRecordingID, ActiveGpuRecording> active_gpu_recordings;
    std::vector<PendingGpuSubmit> pending_gpu_submits;
    std::vector<GpuDeviceCallbackRegistration> gpu_device_callback_registrations;
    std::mutex gpu_mutex;

    ProfilerImpl(Profiler* profiler_)
        : profiler(profiler_)
    {
        trace_storage = std::make_shared<ProfilerTraceStorage>();
    }

    ~ProfilerImpl()
    {
        unregister_gpu_device_callbacks();
        for (ThreadData* thread_data : thread_data_storage)
            delete thread_data;
        for (GpuContextData* context_data : gpu_context_data_storage)
            delete context_data;
    }

    ThreadData* get_or_create_thread_data(std::thread::id thread_id)
    {
        std::lock_guard lock(thread_data_mutex);
        auto it = thread_data_by_id.find(thread_id);
        if (it != thread_data_by_id.end())
            return it->second;
        ThreadData* thread_data = new ThreadData(profiler, thread_id, trace_storage.get());
        thread_data_storage.push_back(thread_data);
        thread_data_by_id[thread_id] = thread_data;
        return thread_data;
    }

    ThreadData* get_this_thread_data()
    {
        if (!s_thread_data || s_thread_data->profiler != profiler) {
            s_thread_data = get_or_create_thread_data(std::this_thread::get_id());
        }
        return s_thread_data;
    }

    GpuContextData* get_or_create_gpu_context_data(Device* device, CommandQueueType queue)
    {
        std::lock_guard lock(gpu_context_data_mutex);
        auto key = std::make_pair(device, queue);
        auto it = gpu_context_data_by_key.find(key);
        if (it != gpu_context_data_by_key.end())
            return it->second;
        GpuContextData* context_data = new GpuContextData(profiler, device, queue);
        gpu_context_data_storage.push_back(context_data);
        gpu_context_data_by_key[key] = context_data;
        return context_data;
    }

    GpuContextData* get_gpu_context_data(Device* device, CommandQueueType queue)
    {
        if (!s_gpu_context_data || s_gpu_context_data->profiler != profiler || s_gpu_context_data->device != device
            || s_gpu_context_data->queue != queue) {
            s_gpu_context_data = get_or_create_gpu_context_data(device, queue);
        }
        return s_gpu_context_data;
    }

    GpuContextData* get_gpu_context_data(CommandEncoder* encoder)
    {
        SGL_ASSERT(encoder);
        return get_gpu_context_data(encoder->device(), encoder->queue());
    }

    void process_events(ThreadData* thread_data)
    {
        ProfilerEvent event;
        while (thread_data->queue.try_dequeue(thread_data->consumer_token, event)) {
            switch (event.type) {
            case ProfilerEventType::begin_zone: {
                const ProfilerEventBeginZone& event_data = event.begin_zone;
                ProfilerZone* zone = thread_data->trace_storage->allocate<ProfilerZone>();
                zone->start_timestamp = event_data.timestamp;
                zone->source_location = event_data.source_location;
                zone->name = event_data.name;
                zone->parent = thread_data->zone_stack.empty() ? nullptr : thread_data->zone_stack.back();
                zone->children = {};
                thread_data->zone_stack.push_back(zone);
                thread_data->zone_children_stack.resize(thread_data->zone_stack.size());
                break;
            }
            case ProfilerEventType::end_zone: {
                const ProfilerEventEndZone& event_data = event.end_zone;
                SGL_ASSERT(!thread_data->zone_stack.empty());
                ProfilerZone* zone = thread_data->zone_stack.back();
                zone->end_timestamp = event_data.timestamp;
                if (zone->parent) {
                    thread_data->zone_children_stack[thread_data->zone_stack.size() - 2].push_back(zone);
                }
                auto& children = thread_data->zone_children_stack.back();
                if (children.size() > 0) {
                    const ProfilerZone** children_data
                        = thread_data->trace_storage->allocate<const ProfilerZone*>(children.size());
                    std::copy(children.begin(), children.end(), children_data);
                    zone->children = {children_data, children.size()};
                    children.clear();
                }
                thread_data->zone_stack.pop_back();
                thread_data->zone_children_stack.pop_back();

                // If this is a root zone, add it to the thread's zones list.
                if (!zone->parent)
                    thread_data->zones.push_back(zone);

                break;
            }
            case ProfilerEventType::begin_gpu_zone: {
                break;
            }
            case ProfilerEventType::end_gpu_zone: {
                break;
            }
            case ProfilerEventType::begin_frame: {
                const ProfilerEventBeginFrame& event_data = event.begin_frame;
                SGL_ASSERT(!thread_data->current_frame);
                ProfilerFrame* frame = thread_data->trace_storage->allocate<ProfilerFrame>();
                frame->start_timestamp = event_data.timestamp;
                frame->source_location = event_data.source_location;
                frame->name = event_data.name;
                frame->frame_id = thread_data->frame_id++;
                thread_data->current_frame = frame;
                break;
            }
            case ProfilerEventType::end_frame: {
                const ProfilerEventEndFrame& event_data = event.end_frame;
                SGL_ASSERT(thread_data->current_frame);
                ProfilerFrame* frame = thread_data->current_frame;
                frame->end_timestamp = event_data.timestamp;
                thread_data->current_frame = nullptr;
                thread_data->frames.push_back(frame);
                break;
            }
            default:
                SGL_THROW("Unknown queue event type: {}", static_cast<uint16_t>(event.type));
            }
        }
    }

    void process_threads()
    {
        std::lock_guard lock(thread_data_mutex);
        for (ThreadData* thread_data : thread_data_storage) {
            process_events(thread_data);
        }
    }

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
        context.timeline_info.type = ProfilerTimelineType::gpu;
        context.timeline_info.name = fmt::format("GPU {} {}", device->info().adapter_name, enum_to_string(queue));
        context.timeline_info.queue = queue;

        auto [inserted_it, inserted] = gpu_contexts.emplace(key, std::move(context));
        SGL_ASSERT(inserted);
        return inserted_it->second;
    }

    ProfilerGpuQueryBlock acquire_gpu_query_block_locked(Device* device, CommandQueueType queue)
    {
        SGL_CHECK_NOT_NULL(device);

        GpuContext& context = get_or_create_gpu_context_locked(device, queue);
        if (!context.free_query_blocks.empty()) {
            ProfilerGpuQueryBlock block = std::move(context.free_query_blocks.back());
            context.free_query_blocks.pop_back();
            block.query_pool->reset(block.first_query_index, block.query_count);
            return block;
        }

        if (context.query_pools.empty()
            || context.next_query_index + kGpuTimestampQueriesPerBlock > context.query_pools.back()->desc().count) {
            context.query_pools.push_back(device->create_query_pool({
                .type = QueryType::timestamp,
                .count = kGpuTimestampQueryPoolSize,
            }));
            context.next_query_index = 0;
        }

        ProfilerGpuQueryBlock block{
            .query_pool = context.query_pools.back(),
            .first_query_index = context.next_query_index,
            .query_count = kGpuTimestampQueriesPerBlock,
        };
        context.next_query_index += kGpuTimestampQueriesPerBlock;
        block.query_pool->reset(block.first_query_index, block.query_count);
        return block;
    }

    void release_gpu_query_blocks_locked(
        Device* device,
        CommandQueueType queue,
        std::vector<ProfilerGpuQueryBlock>&& query_blocks
    )
    {
        if (query_blocks.empty())
            return;

        SGL_CHECK_NOT_NULL(device);
        GpuContext& context = get_or_create_gpu_context_locked(device, queue);
        for (ProfilerGpuQueryBlock& block : query_blocks) {
            if (block.query_pool && block.query_count > 0)
                context.free_query_blocks.push_back(std::move(block));
        }
    }

    void
    release_gpu_query_blocks(Device* device, CommandQueueType queue, std::vector<ProfilerGpuQueryBlock>&& query_blocks)
    {
        std::lock_guard lock(gpu_mutex);
        release_gpu_query_blocks_locked(device, queue, std::move(query_blocks));
    }

    bool gpu_recording_supported(Device* device, CommandQueueType queue) const noexcept
    {
        if (!profiler->enabled() || !device)
            return false;
        if (queue != CommandQueueType::graphics)
            return false;
        if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
            return false;
        return device->info().timestamp_frequency > 0;
    }

    void ensure_gpu_device_callbacks_registered_locked(Device* device)
    {
        SGL_CHECK_NOT_NULL(device);

        for (const GpuDeviceCallbackRegistration& registration : gpu_device_callback_registrations) {
            if (registration.device.get() == device)
                return;
        }

        GpuDeviceCallbackRegistration registration;
        registration.device = ref<Device>(device);
        registration.submitted_callback_id = device->_register_command_recording_submitted_callback(
            [this, device](const CommandRecordingSubmittedEvent& event)
            {
                on_command_recording_submitted(device, event);
            }
        );
        registration.discarded_callback_id = device->_register_command_recording_discarded_callback(
            [this, device](const CommandRecordingDiscardedEvent& event)
            {
                on_command_recording_discarded(device, event);
            }
        );
        gpu_device_callback_registrations.push_back(std::move(registration));
    }

    void unregister_gpu_device_callbacks() noexcept
    {
        std::vector<GpuDeviceCallbackRegistration> registrations;
        {
            std::lock_guard lock(gpu_mutex);
            registrations = std::move(gpu_device_callback_registrations);
        }

        for (GpuDeviceCallbackRegistration& registration : registrations) {
            try {
                if (registration.device) {
                    registration.device->_unregister_command_recording_submitted_callback(
                        registration.submitted_callback_id
                    );
                    registration.device->_unregister_command_recording_discarded_callback(
                        registration.discarded_callback_id
                    );
                }
            } catch (...) {
            }
        }
    }

    void clear_gpu_recording_cache(CommandRecordingID recording_id = 0) noexcept
    {
        if (s_gpu_recording_cache.profiler != this)
            return;
        if (recording_id != 0 && s_gpu_recording_cache.recording_id != recording_id)
            return;

        s_gpu_recording_cache = {};
    }

    ActiveGpuRecording* get_cached_active_gpu_recording(CommandRecordingID recording_id) noexcept
    {
        if (s_gpu_recording_cache.profiler == this && s_gpu_recording_cache.recording_id == recording_id)
            return s_gpu_recording_cache.recording;
        return nullptr;
    }

    ActiveGpuRecording* get_or_create_active_gpu_recording_slow(CommandEncoder* encoder)
    {
        SGL_CHECK_NOT_NULL(encoder);
        Device* device = encoder->device();
        SGL_CHECK_NOT_NULL(device);

        const CommandRecordingID recording_id = encoder->recording_id();
        SGL_CHECK_NE(recording_id, 0);

        std::lock_guard lock(gpu_mutex);
        ensure_gpu_device_callbacks_registered_locked(device);

        auto it = active_gpu_recordings.find(recording_id);
        if (it != active_gpu_recordings.end()) {
            s_gpu_recording_cache = {
                .profiler = this,
                .recording_id = recording_id,
                .recording = &it->second,
            };
            return &it->second;
        }

        ActiveGpuRecording recording;
        recording.device = ref<Device>(device);
        recording.queue = encoder->queue();
        auto [inserted_it, inserted] = active_gpu_recordings.emplace(recording_id, std::move(recording));
        SGL_ASSERT(inserted);
        s_gpu_recording_cache = {
            .profiler = this,
            .recording_id = recording_id,
            .recording = &inserted_it->second,
        };
        return &inserted_it->second;
    }

    ActiveGpuRecording* get_or_create_active_gpu_recording(CommandEncoder* encoder)
    {
        const CommandRecordingID recording_id = encoder->recording_id();
        if (ActiveGpuRecording* recording = get_cached_active_gpu_recording(recording_id))
            return recording;
        return get_or_create_active_gpu_recording_slow(encoder);
    }

    ActiveGpuRecording* find_active_gpu_recording_slow(CommandEncoder* encoder) noexcept
    {
        try {
            const CommandRecordingID recording_id = encoder->recording_id();
            if (recording_id == 0)
                return nullptr;

            std::lock_guard lock(gpu_mutex);
            auto it = active_gpu_recordings.find(recording_id);
            if (it == active_gpu_recordings.end())
                return nullptr;

            s_gpu_recording_cache = {
                .profiler = this,
                .recording_id = recording_id,
                .recording = &it->second,
            };
            return &it->second;
        } catch (...) {
            return nullptr;
        }
    }

    ActiveGpuRecording* find_active_gpu_recording(CommandEncoder* encoder) noexcept
    {
        const CommandRecordingID recording_id = encoder->recording_id();
        if (ActiveGpuRecording* recording = get_cached_active_gpu_recording(recording_id))
            return recording;
        return find_active_gpu_recording_slow(encoder);
    }

    ProfilerGpuQueryPair allocate_gpu_query_pair_fast(ActiveGpuRecording& recording) noexcept
    {
        if (!recording.current_query_pool || recording.next_query_index + 2 > recording.current_query_end_index)
            return {};

        ProfilerGpuQueryPair pair{
            .query_pool = recording.current_query_pool,
            .begin_query_index = recording.next_query_index,
            .end_query_index = recording.next_query_index + 1,
        };
        recording.next_query_index += 2;
        return pair;
    }

    ProfilerGpuQueryPair allocate_gpu_query_pair_slow(ActiveGpuRecording& recording)
    {
        std::lock_guard lock(gpu_mutex);

        if (ProfilerGpuQueryPair pair = allocate_gpu_query_pair_fast(recording); pair.query_pool)
            return pair;

        ProfilerGpuQueryBlock block = acquire_gpu_query_block_locked(recording.device.get(), recording.queue);
        recording.current_query_pool = block.query_pool;
        recording.next_query_index = block.first_query_index;
        recording.current_query_end_index = block.first_query_index + block.query_count;
        recording.query_blocks.push_back(std::move(block));
        return allocate_gpu_query_pair_fast(recording);
    }

    ProfilerGpuQueryPair allocate_gpu_query_pair(ActiveGpuRecording& recording)
    {
        if (ProfilerGpuQueryPair pair = allocate_gpu_query_pair_fast(recording); pair.query_pool)
            return pair;
        return allocate_gpu_query_pair_slow(recording);
    }

    void begin_gpu_zone(CommandEncoder* encoder, const ProfilerSourceLocation* source_location, const char* name)
    {
        SGL_CHECK_NOT_NULL(encoder);
        SGL_CHECK_NOT_NULL(source_location);

        if (!gpu_recording_supported(encoder->device(), encoder->queue()))
            return;

        ActiveGpuRecording* recording = get_or_create_active_gpu_recording(encoder);
        if (!recording)
            return;

        ProfilerGpuQueryPair query_pair = allocate_gpu_query_pair(*recording);
        if (!query_pair.query_pool)
            return;

        encoder->write_timestamp(query_pair.query_pool.get(), query_pair.begin_query_index);

        const uint32_t zone_index = uint32_t(recording->zones.size());
        const int32_t parent_index = recording->zone_stack.empty() ? -1 : int32_t(recording->zone_stack.back());
        if (parent_index >= 0)
            recording->zones[size_t(parent_index)].child_indices.push_back(zone_index);

        recording->zones.push_back({
            .source_location = source_location,
            .name = name,
            .query_pool = std::move(query_pair.query_pool),
            .begin_query_index = query_pair.begin_query_index,
            .end_query_index = query_pair.end_query_index,
            .parent_index = parent_index,
            .child_indices = {},
            .completed = false,
        });
        recording->zone_stack.push_back(zone_index);
    }

    void end_gpu_zone(CommandEncoder* encoder) noexcept
    {
        if (!encoder)
            return;

        try {
            ActiveGpuRecording* recording = find_active_gpu_recording(encoder);
            if (!recording || recording->zone_stack.empty())
                return;

            const uint32_t zone_index = recording->zone_stack.back();
            recording->zone_stack.pop_back();

            ProfilerGpuZoneRecord& zone = recording->zones[zone_index];
            try {
                encoder->write_timestamp(zone.query_pool.get(), zone.end_query_index);
                zone.completed = true;
            } catch (...) {
                zone.completed = false;
            }
        } catch (...) {
        }
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

    ProfilerZone* resolve_gpu_zone(
        const ProfilerGpuRecordingBatch& batch,
        const GpuTimestampAnchor& anchor,
        size_t zone_index,
        std::vector<ProfilerZone*>& resolved_zones
    )
    {
        if (zone_index >= batch.zones.size())
            return nullptr;

        ProfilerZone*& cached_zone = resolved_zones[zone_index];
        if (cached_zone)
            return cached_zone;

        const ProfilerGpuZoneRecord& record = batch.zones[zone_index];
        if (!record.completed || !record.query_pool)
            return nullptr;

        const uint32_t begin_index = record.begin_query_index;
        const uint32_t end_index = record.end_query_index;
        SGL_ASSERT(begin_index < record.query_pool->desc().count);
        SGL_ASSERT(end_index < record.query_pool->desc().count);

        const uint64_t start_timestamp = gpu_timestamp_to_cpu_ns(record.query_pool->get_result(begin_index), anchor);
        const uint64_t end_timestamp = gpu_timestamp_to_cpu_ns(record.query_pool->get_result(end_index), anchor);

        ProfilerZone* zone = trace_storage->allocate<ProfilerZone>();
        zone->start_timestamp = start_timestamp;
        zone->end_timestamp = std::max(start_timestamp, end_timestamp);
        zone->source_location = record.source_location;
        zone->name = record.name;
        zone->parent = record.parent_index >= 0
            ? resolve_gpu_zone(batch, anchor, size_t(record.parent_index), resolved_zones)
            : nullptr;
        zone->children = {};
        cached_zone = zone;

        short_vector<const ProfilerZone*> children;
        for (uint32_t child_index : record.child_indices) {
            ProfilerZone* child = resolve_gpu_zone(batch, anchor, child_index, resolved_zones);
            if (child)
                children.push_back(child);
        }

        if (!children.empty()) {
            const ProfilerZone** children_data = trace_storage->allocate<const ProfilerZone*>(children.size());
            std::copy(children.begin(), children.end(), children_data);
            zone->children = {children_data, children.size()};
        }

        return zone;
    }

    void resolve_gpu_batch_locked(const ProfilerGpuRecordingBatch& batch, const GpuTimestampAnchor& anchor)
    {
        if (batch.zones.empty())
            return;

        GpuContext& context = get_or_create_gpu_context_locked(batch.device.get(), batch.queue);

        std::vector<ProfilerZone*> resolved_zones(batch.zones.size(), nullptr);
        for (size_t i = 0; i < batch.zones.size(); ++i) {
            const ProfilerGpuZoneRecord& record = batch.zones[i];
            if (record.parent_index >= 0)
                continue;

            ProfilerZone* zone = resolve_gpu_zone(batch, anchor, i, resolved_zones);
            if (zone)
                context.zones.push_back(zone);
        }
    }

    void release_gpu_recording_batches(std::vector<ProfilerGpuRecordingBatch>&& batches) noexcept
    {
        for (ProfilerGpuRecordingBatch& batch : batches) {
            try {
                release_gpu_query_blocks(batch.device.get(), batch.queue, std::move(batch.query_blocks));
            } catch (...) {
            }
        }
    }

    void submit_gpu_recordings(uint64_t submit_id, std::vector<ProfilerGpuRecordingBatch>&& batches) noexcept
    {
        if (batches.empty())
            return;

        PendingGpuSubmit pending;
        try {
            pending.submit_id = submit_id;
            pending.device = batches.front().device;
            pending.queue = batches.front().queue;
            pending.batches = std::move(batches);

            try {
                pending.anchor = capture_gpu_timestamp_anchor(pending.device.get(), pending.queue);
            } catch (...) {
                pending.anchor = {};
            }

            std::lock_guard lock(gpu_mutex);
            pending_gpu_submits.push_back(std::move(pending));
        } catch (...) {
            release_gpu_recording_batches(std::move(pending.batches));
            release_gpu_recording_batches(std::move(batches));
        }
    }

    void submit_gpu_recording(uint64_t submit_id, ProfilerGpuRecordingBatch&& batch) noexcept
    {
        std::vector<ProfilerGpuRecordingBatch> batches;
        try {
            batches.reserve(1);
            batches.push_back(std::move(batch));
        } catch (...) {
            try {
                release_gpu_query_blocks(batch.device.get(), batch.queue, std::move(batch.query_blocks));
            } catch (...) {
            }
            return;
        }
        submit_gpu_recordings(submit_id, std::move(batches));
    }

    void on_command_recording_submitted(Device* device, const CommandRecordingSubmittedEvent& event) noexcept
    {
        if (!device || event.id == 0)
            return;

        clear_gpu_recording_cache(event.id);

        ActiveGpuRecording recording;
        {
            std::lock_guard lock(gpu_mutex);
            auto it = active_gpu_recordings.find(event.id);
            if (it == active_gpu_recordings.end())
                return;

            recording = std::move(it->second);
            active_gpu_recordings.erase(it);
        }

        if (recording.zones.empty() && recording.query_blocks.empty())
            return;

        ref<Device> batch_device = recording.device;
        CommandQueueType queue = recording.queue;
        if (event.command_buffer) {
            batch_device = ref<Device>(event.command_buffer->device());
            queue = event.command_buffer->queue();
        }

        ProfilerGpuRecordingBatch batch{
            .device = std::move(batch_device),
            .queue = queue,
            .zones = std::move(recording.zones),
            .query_blocks = std::move(recording.query_blocks),
        };
        submit_gpu_recording(event.submit_id, std::move(batch));
    }

    void on_command_recording_discarded(Device* device, const CommandRecordingDiscardedEvent& event) noexcept
    {
        if (!device || event.id == 0)
            return;

        clear_gpu_recording_cache(event.id);

        {
            std::lock_guard lock(gpu_mutex);
            auto it = active_gpu_recordings.find(event.id);
            if (it == active_gpu_recordings.end())
                return;

            // A discarded recording may contain timestamp writes that never become submitted GPU work.
            // Keep those query ranges abandoned instead of trying to prove they are safe to recycle.
            active_gpu_recordings.erase(it);
        }
    }

    void process_gpu_submits()
    {
        std::lock_guard lock(gpu_mutex);

        size_t index = 0;
        while (index < pending_gpu_submits.size()) {
            PendingGpuSubmit& pending = pending_gpu_submits[index];
            if (!pending.device->is_submit_finished(pending.submit_id)) {
                ++index;
                continue;
            }

            if (pending.anchor.valid) {
                for (const ProfilerGpuRecordingBatch& batch : pending.batches) {
                    try {
                        resolve_gpu_batch_locked(batch, pending.anchor);
                    } catch (...) {
                    }
                }
            }

            for (ProfilerGpuRecordingBatch& batch : pending.batches)
                release_gpu_query_blocks_locked(batch.device.get(), batch.queue, std::move(batch.query_blocks));

            pending_gpu_submits.erase(pending_gpu_submits.begin() + index);
        }
    }

    SGL_NON_COPYABLE_AND_MOVABLE(ProfilerImpl);
};

// ----------------------------------------------------------------------------
// Profiler
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

} // namespace

Profiler::Profiler(ProfilerDesc desc)
    : m_desc(std::move(desc))
{
    m_impl = new ProfilerImpl(this);
    push_current_profiler(this);
}

Profiler::~Profiler()
{
    remove_current_profiler_entries(this);
    // TODO process remaining events and clean up zones
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

ref<ProfilerTrace> Profiler::trace_snapshot()
{
    ref<ProfilerTrace> trace = make_ref<ProfilerTrace>();
    trace->m_trace_storage = m_impl->trace_storage;

    short_vector<ThreadData*> thread_data_storage;
    {
        std::lock_guard lock(m_impl->thread_data_mutex);
        thread_data_storage = m_impl->thread_data_storage;
    }
    for (ThreadData* thread_data : thread_data_storage) {
        ProfilerTrace::Timeline timeline;
        timeline.info = thread_data->timeline_info;
        timeline.zones = thread_data->zones;
        trace->m_timelines.push_back(std::move(timeline));
    }
    {
        std::lock_guard lock(m_impl->gpu_mutex);
        for (const auto& [key, context] : m_impl->gpu_contexts) {
            SGL_UNUSED(key);
            ProfilerTrace::Timeline timeline;
            timeline.info = context.timeline_info;
            timeline.zones = context.zones;
            trace->m_timelines.push_back(std::move(timeline));
        }
    }
    return trace;
}

bool Profiler::begin_zone(
    const ProfilerSourceLocation* source_location,
    const char* name,
    CommandEncoder* encoder,
    ProfilerZoneFlags flags
) noexcept
{
    if (!m_enabled)
        return false;

    uint64_t timestamp = Timer::now();

    if (!source_location) {
        SGL_ASSERT(source_location);
        return false;
    }

    if (name && is_set(flags, ProfilerZoneFlags::copy_name))
        name = Profiler::intern_name(name);

    ThreadData* thread_data = m_impl->get_this_thread_data();
    thread_data->queue_event({
        .type = ProfilerEventType::begin_zone,
        .begin_zone{
            .timestamp = timestamp,
            .source_location = source_location,
            .name = name,
        },
    });

    if (encoder) {
        const char* debug_name = name ? name : (source_location->function ? source_location->function : "zone");
        if (is_set(flags, ProfilerZoneFlags::debug_group))
            encoder->push_debug_group(debug_name, float3(0.5f));

        GpuContextData* gpu_context_data = m_impl->get_gpu_context_data(encoder);
        uint32_t query_index = gpu_context_data->allocate_timestamp_query(encoder);
        if (query_index != GpuContextData::INVALID_QUERY_INDEX)
            encoder->write_timestamp(gpu_context_data->query_pool, query_index);

        thread_data->queue_event({
            .type = ProfilerEventType::begin_gpu_zone,
            .begin_gpu_zone{
                .timestamp = timestamp,
                .source_location = source_location,
                .name = name,
                .gpu_context_data = gpu_context_data,
                .query_index = query_index,
            },
        });

        // try {
        //     m_impl->begin_gpu_zone(encoder, source_location, name);
        // } catch (...) {
        // }
    }

    return true;
}

void Profiler::end_zone(CommandEncoder* encoder, ProfilerZoneFlags flags) noexcept
{
    uint64_t timestamp = Timer::now();

    ThreadData* thread_data = m_impl->get_this_thread_data();
    thread_data->queue_event({
        .type = ProfilerEventType::end_zone,
        .end_zone{
            .timestamp = timestamp,
        },
    });

    if (encoder) {
        if (is_set(flags, ProfilerZoneFlags::debug_group))
            encoder->pop_debug_group();

        GpuContextData* gpu_context_data = m_impl->get_gpu_context_data(encoder);
        uint32_t query_index = gpu_context_data->allocate_timestamp_query(encoder);
        if (query_index != GpuContextData::INVALID_QUERY_INDEX)
            encoder->write_timestamp(gpu_context_data->query_pool, query_index);

        thread_data->queue_event({
            .type = ProfilerEventType::begin_gpu_zone,
            .begin_gpu_zone{
                .timestamp = timestamp,
                .gpu_context_data = gpu_context_data,
                .query_index = query_index,
            },
        });

        // m_impl->end_gpu_zone(encoder);
        // try {
        //     if (is_set(flags, ProfilerZoneFlags::debug_group))
        //         encoder->pop_debug_group();
        // } catch (...) {
        // }
    }
}

bool Profiler::begin_frame(const ProfilerSourceLocation* source_location, const char* name) noexcept
{
    if (!m_enabled)
        return false;

    uint64_t timestamp = Timer::now();

    if (!source_location) {
        SGL_ASSERT(source_location);
        return false;
    }

    ThreadData* thread_data = m_impl->get_this_thread_data();
    thread_data->queue_event({
        .type = ProfilerEventType::begin_frame,
        .begin_frame{
            .timestamp = timestamp,
            .source_location = source_location,
            .name = name,
        },
    });

    return true;
}

void Profiler::end_frame() noexcept
{
    uint64_t timestamp = Timer::now();

    ThreadData* thread_data = m_impl->get_this_thread_data();
    thread_data->queue_event({
        .type = ProfilerEventType::end_frame,
        .end_frame{
            .timestamp = timestamp,
        },
    });
}

void Profiler::tick()
{
    m_impl->process_threads();
    m_impl->process_gpu_submits();
}

std::string Profiler::to_string() const
{
    return fmt::format(
        "Profiler(enabled = {}, auto_zones_enabled = {}, debug_groups_enabled = {})",
        m_enabled.load(),
        m_auto_zones_enabled.load(),
        m_debug_groups_enabled.load()
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
