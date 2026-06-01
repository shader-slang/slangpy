// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler.h"

#include "sgl/core/error.h"
#include "sgl/core/format.h"
#include "sgl/core/hash.h"

#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sgl {
namespace {

    struct ProfilerInternedStringRegistry {
        std::mutex mutex;
        std::deque<std::string> names;
        std::unordered_map<std::string_view, const char*> names_by_value;

        ProfilerInternedStringRegistry() = default;

        const char* intern(std::string_view value)
        {
            std::lock_guard lock(mutex);

            auto it = names_by_value.find(value);
            if (it != names_by_value.end())
                return it->second;

            const std::string& name = names.emplace_back(value);
            auto [inserted_it, inserted] = names_by_value.emplace(name, name.c_str());
            SGL_ASSERT(inserted);

            return inserted_it->second;
        }

        SGL_NON_COPYABLE_AND_MOVABLE(ProfilerInternedStringRegistry);
    };

    ProfilerInternedStringRegistry& name_registry()
    {
        static ProfilerInternedStringRegistry registry;
        return registry;
    }

    struct ProfilerSourceLocationKeyView {
        std::string_view file;
        uint32_t line{0};
        std::string_view function;
    };

    struct ProfilerSourceLocationKeyHash {
        size_t operator()(const ProfilerSourceLocationKeyView& key) const
        {
            return sgl::hash(key.file, key.line, key.function);
        }
    };

    struct ProfilerSourceLocationKeyEqual {
        bool operator()(const ProfilerSourceLocationKeyView& a, const ProfilerSourceLocationKeyView& b) const
        {
            return a.file == b.file && a.line == b.line && a.function == b.function;
        }
    };

    struct InternedProfilerSourceLocation {
        std::string file;
        uint32_t line{0};
        std::string function;
        ProfilerSourceLocation source_location;

        InternedProfilerSourceLocation(std::string file_, uint32_t line_, std::string function_)
            : file(std::move(file_))
            , line(line_)
            , function(std::move(function_))
        {
            source_location.file = file.c_str();
            source_location.line = line;
            source_location.function = function.c_str();
        }

        ProfilerSourceLocationKeyView key() const
        {
            return {
                .file = file,
                .line = line,
                .function = function,
            };
        }

        SGL_NON_COPYABLE_AND_MOVABLE(InternedProfilerSourceLocation);
    };

    struct ProfilerSourceLocationRegistry {
        std::mutex mutex;
        std::deque<InternedProfilerSourceLocation> source_locations;
        std::unordered_map<
            ProfilerSourceLocationKeyView,
            const ProfilerSourceLocation*,
            ProfilerSourceLocationKeyHash,
            ProfilerSourceLocationKeyEqual>
            source_locations_by_key;

        ProfilerSourceLocationRegistry() = default;

        const ProfilerSourceLocation* intern(std::string_view file, uint32_t line, std::string_view function)
        {
            std::lock_guard lock(mutex);

            ProfilerSourceLocationKeyView key_view{
                .file = file,
                .line = line,
                .function = function,
            };
            auto it = source_locations_by_key.find(key_view);
            if (it != source_locations_by_key.end())
                return it->second;

            InternedProfilerSourceLocation& record
                = source_locations.emplace_back(std::string(file), line, std::string(function));
            auto [inserted_it, inserted] = source_locations_by_key.emplace(record.key(), &record.source_location);
            SGL_ASSERT(inserted);

            return inserted_it->second;
        }

        SGL_NON_COPYABLE_AND_MOVABLE(ProfilerSourceLocationRegistry);
    };

    ProfilerSourceLocationRegistry& source_location_registry()
    {
        static ProfilerSourceLocationRegistry registry;
        return registry;
    }

} // namespace

thread_local std::vector<ref<Profiler>> s_current_profiler_stack;

Profiler::Profiler(ProfilerDesc desc)
    : m_desc(std::move(desc))
{
}

Profiler::~Profiler() = default;

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
    return source_location_registry().intern(file, line, function);
}

const char* Profiler::intern_name(std::string_view name)
{
    return name_registry().intern(name);
}

bool Profiler::begin_zone(
    const ProfilerSourceLocation* source_location,
    const char* name,
    CommandEncoder* encoder,
    ProfilerZoneFlags flags
) noexcept
{
    if (!source_location) {
        SGL_ASSERT(source_location);
        return false;
    }
    SGL_UNUSED(name);
    SGL_UNUSED(encoder);
    SGL_UNUSED(flags);
    return m_enabled.load();
}

void Profiler::end_zone(CommandEncoder* encoder) noexcept
{
    SGL_UNUSED(encoder);
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
