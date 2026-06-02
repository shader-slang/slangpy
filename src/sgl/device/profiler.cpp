// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler.h"

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/hash.h"
#include "sgl/core/short_vector.h"
#include "sgl/core/timer.h"

#include <deque>
#include <mutex>
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

    ProfilerTraceStorage* trace_storage{nullptr};

    std::deque<ProfilerEvent> queue;

    short_vector<ProfilerZone*> zone_stack;
    short_vector<short_vector<ProfilerZone*>> zone_children_stack;

    std::vector<ProfilerZone*> zones;
};

// ----------------------------------------------------------------------------
// ProfilerImpl
// ----------------------------------------------------------------------------

thread_local ThreadData* s_thread_data{nullptr};

struct ProfilerImpl {

    Profiler* profiler{nullptr};

    std::shared_ptr<ProfilerTraceStorage> trace_storage;

    // Thread data.
    short_vector<ThreadData*> thread_data_storage;
    std::unordered_map<std::thread::id, ThreadData*> thread_data_by_id;
    std::mutex thread_data_mutex;

    ProfilerImpl(Profiler* profiler_)
        : profiler(profiler_)
    {
        trace_storage = std::make_shared<ProfilerTraceStorage>();
    }

    ~ProfilerImpl()
    {
        for (ThreadData* thread_data : thread_data_storage) {
            delete thread_data;
        }
    }

    ThreadData* get_or_create_thread_data(std::thread::id thread_id)
    {
        std::lock_guard lock(thread_data_mutex);
        auto it = thread_data_by_id.find(thread_id);
        if (it != thread_data_by_id.end())
            return it->second;
        ThreadData* thread_data = new ThreadData();
        thread_data->profiler = profiler;
        thread_data->trace_storage = trace_storage.get();
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

    void queue_event(ThreadData* thread_data, ProfilerEvent&& event) { thread_data->queue.push_back(event); }

    void process_events(ThreadData* thread_data)
    {
        for (const ProfilerEvent& event : thread_data->queue) {
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
                    ProfilerZone** children_data = thread_data->trace_storage->allocate<ProfilerZone*>(children.size());
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
            case ProfilerEventType::begin_frame: // No-op for now.
            case ProfilerEventType::end_frame:   // No-op for now.
                break;
            default:
                SGL_THROW("Unknown queue event type: {}", static_cast<uint16_t>(event.type));
            }
        }
        thread_data->queue.clear();
    }

    void process_threads()
    {
        std::lock_guard lock(thread_data_mutex);
        for (ThreadData* thread_data : thread_data_storage) {
            process_events(thread_data);
        }
    }

    SGL_NON_COPYABLE_AND_MOVABLE(ProfilerImpl);
};

// ----------------------------------------------------------------------------
// Profiler
// ----------------------------------------------------------------------------

thread_local std::vector<ref<Profiler>> s_current_profiler_stack;

Profiler::Profiler(ProfilerDesc desc)
    : m_desc(std::move(desc))
{
    m_impl = new ProfilerImpl(this);
}

Profiler::~Profiler()
{
    // TODO process remaining events and clean up zones
    delete m_impl;
}

Profiler* current_profiler_or_null()
{
    if (s_current_profiler_stack.empty())
        return nullptr;
    return s_current_profiler_stack.back().get();
}

Profiler* current_profiler()
{
    Profiler* profiler = current_profiler_or_null();
    SGL_CHECK(profiler, "No current profiler. Use push_current_profiler() or ProfilerScope to set one.");
    return profiler;
}

void push_current_profiler(Profiler* profiler)
{
    SGL_CHECK(profiler != nullptr, "Cannot push a null profiler.");
    s_current_profiler_stack.push_back(ref<Profiler>(profiler));
}

Profiler* pop_current_profiler()
{
    SGL_CHECK(
        !s_current_profiler_stack.empty(),
        "No profiler to pop. push_current_profiler()/pop_current_profiler() mismatch."
    );

    Profiler* profiler = s_current_profiler_stack.back().get();
    s_current_profiler_stack.pop_back();
    return profiler;
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

    SGL_UNUSED(encoder);
    SGL_UNUSED(flags);

    ThreadData* thread_data = m_impl->get_this_thread_data();
    m_impl->queue_event(
        thread_data,
        {
            .type = ProfilerEventType::begin_zone,
            .begin_zone{
                .timestamp = timestamp,
                .source_location = source_location,
                .name = name,
            },
        }
    );

    return true;
}

void Profiler::end_zone(CommandEncoder* encoder) noexcept
{
    SGL_UNUSED(encoder);

    uint64_t timestamp = Timer::now();

    ThreadData* thread_data = m_impl->get_this_thread_data();
    m_impl->queue_event(
        thread_data,
        {
            .type = ProfilerEventType::end_zone,
            .end_zone{
                .timestamp = timestamp,
            },
        }
    );
}

bool Profiler::begin_frame(const ProfilerSourceLocation* source_location, const char* name) noexcept
{
    if (!source_location) {
        SGL_ASSERT(source_location);
        return false;
    }
    SGL_UNUSED(name);
    return m_enabled.load();
}

void Profiler::end_frame() noexcept { }

void Profiler::tick()
{
    m_impl->process_threads();
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

} // namespace sgl
