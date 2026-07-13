// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/command.h"
#include "sgl/utils/profiler.h"

#include <array>
#include <memory>

namespace {

sgl::ref<sgl::Profiler> current_profiler_ref()
{
    if (sgl::Profiler* current = sgl::current_profiler_or_null())
        return sgl::ref<sgl::Profiler>(current);
    return {};
}

int python_frame_instruction_offset(PyFrameObject* frame)
{
#if PY_VERSION_HEX >= 0x030b0000
    return PyFrame_GetLasti(frame);
#else
    return frame->f_lasti;
#endif
}

std::string python_function_name(nb::handle code, PyFrameObject* frame)
{
    nb::object qualname = nb::getattr(code, "co_qualname", nb::none());
    if (!qualname.is_none())
        return nb::cast<std::string>(qualname);

    std::string function = nb::cast<std::string>(nb::getattr(code, "co_name"));
    nb::object frame_object = nb::borrow<nb::object>(reinterpret_cast<PyObject*>(frame));
    nb::dict locals = nb::cast<nb::dict>(nb::getattr(frame_object, "f_locals"));
    if (locals.contains("self")) {
        nb::handle self = locals["self"];
        function = nb::cast<std::string>(nb::getattr(self.type(), "__qualname__")) + "." + function;
    } else if (locals.contains("cls")) {
        nb::handle cls = locals["cls"];
        nb::object class_qualname = nb::getattr(cls, "__qualname__", nb::none());
        if (!class_qualname.is_none())
            function = nb::cast<std::string>(class_qualname) + "." + function;
    }
    return function;
}

class PyProfilerSiteCache {
public:
    uint32_t resolve(PyFrameObject* frame, nb::object code, int instruction_offset, nb::handle name = {})
    {
        const Py_hash_t name_hash = hash_name(name);
        const uint64_t hash = hash_key(code, instruction_offset, name_hash);
        std::array<Entry, way_count>& set = m_entries[hash % set_count];
        for (Entry& entry : set) {
            if (entry.site_id != 0 && entry.code.ptr() == code.ptr() && entry.instruction_offset == instruction_offset
                && entry.name_hash == name_hash && names_equal(entry.name, name)) {
                entry.last_use = ++m_clock;
                return entry.site_id;
            }
        }

        const std::string file = nb::cast<std::string>(nb::getattr(code, "co_filename"));
        const int line_number = PyFrame_GetLineNumber(frame);
        const uint32_t line = line_number > 0 ? uint32_t(line_number) : 0;
        const std::string function = python_function_name(code, frame);
        const std::string site_name = name.is_valid() ? nb::cast<std::string>(name) : function;
        const uint32_t site_id = sgl::Profiler::register_site(file, line, function, site_name);

        Entry* victim = &set.front();
        for (Entry& entry : set) {
            if (entry.site_id == 0) {
                victim = &entry;
                break;
            }
            if (entry.last_use < victim->last_use)
                victim = &entry;
        }
        victim->code = std::move(code);
        victim->name = name.is_valid() ? nb::borrow<nb::object>(name) : nb::object();
        victim->instruction_offset = instruction_offset;
        victim->name_hash = name_hash;
        victim->site_id = site_id;
        victim->last_use = ++m_clock;
        return site_id;
    }

private:
    static constexpr size_t set_count = 64;
    static constexpr size_t way_count = 4;

    struct Entry {
        nb::object code;
        nb::object name;
        int instruction_offset{0};
        Py_hash_t name_hash{0};
        uint32_t site_id{0};
        uint64_t last_use{0};
    };

    static Py_hash_t hash_name(nb::handle name)
    {
        if (!name.is_valid())
            return 0;
        const Py_hash_t result = PyObject_Hash(name.ptr());
        if (result == -1 && PyErr_Occurred())
            nb::raise_python_error();
        return result;
    }

    static uint64_t hash_key(nb::handle code, int instruction_offset, Py_hash_t name_hash)
    {
        uint64_t value = uint64_t(reinterpret_cast<uintptr_t>(code.ptr()));
        auto combine = [&](uint64_t next)
        {
            value ^= next + 0x9e3779b97f4a7c15ull + (value << 6) + (value >> 2);
        };
        combine(uint64_t(uint32_t(instruction_offset)));
        combine(uint64_t(name_hash));
        return value;
    }

    static bool names_equal(const nb::object& cached, nb::handle requested)
    {
        if (!cached.is_valid() || !requested.is_valid())
            return cached.is_valid() == requested.is_valid();
        if (cached.ptr() == requested.ptr())
            return true;
        const int result = PyObject_RichCompareBool(cached.ptr(), requested.ptr(), Py_EQ);
        if (result < 0)
            nb::raise_python_error();
        return result != 0;
    }

    std::array<std::array<Entry, way_count>, set_count> m_entries;
    uint64_t m_clock{0};
};

struct PyProfilerCallSite {
    PyFrameObject* frame{nullptr};
    nb::object code;
    int instruction_offset{0};
};

PyProfilerCallSite python_caller_site()
{
    PyFrameObject* frame = PyEval_GetFrame();
    SGL_CHECK(frame, "Could not determine the Python profiler callsite");
    nb::object code = nb::steal<nb::object>(reinterpret_cast<PyObject*>(PyFrame_GetCode(frame)));
    return {frame, std::move(code), python_frame_instruction_offset(frame)};
}

class PyProfilerZoneScope {
public:
    PyProfilerZoneScope() = default;

    PyProfilerZoneScope(sgl::ref<sgl::Profiler> profiler, uint32_t site_id, sgl::CommandEncoder* command_encoder)
        : m_profiler(std::move(profiler))
        , m_command_encoder(m_profiler ? command_encoder : nullptr)
        , m_site_id(site_id)
    {
    }

    PyProfilerZoneScope* enter()
    {
        SGL_CHECK(!m_entered, "Profiler zone context cannot be entered more than once");
        if (m_profiler)
            m_token = m_profiler->begin_zone(m_site_id, m_command_encoder);
        m_entered = true;
        return this;
    }

    void exit(nb::object, nb::object, nb::object)
    {
        if (!m_entered)
            return;
        if (m_token.profiler)
            m_token.profiler->end_zone(m_token);
        m_token = {};
        m_entered = false;
    }

private:
    sgl::ref<sgl::Profiler> m_profiler;
    sgl::ref<sgl::CommandEncoder> m_command_encoder;
    uint32_t m_site_id{0};
    sgl::ProfilerZoneToken m_token;
    bool m_entered{false};
};

class PyProfilerFrameScope {
public:
    PyProfilerFrameScope() = default;

    PyProfilerFrameScope(sgl::ref<sgl::Profiler> profiler, uint32_t site_id)
        : m_profiler(std::move(profiler))
        , m_site_id(site_id)
    {
    }

    PyProfilerFrameScope* enter()
    {
        SGL_CHECK(!m_entered, "Profiler frame context cannot be entered more than once");
        if (m_profiler) {
            m_token = m_profiler->begin_frame(m_site_id);
            if (m_profiler->enabled() && !m_token.profiler)
                SGL_THROW("A profiler frame is already active or closing");
        }
        m_entered = true;
        return this;
    }

    void exit(nb::object, nb::object, nb::object)
    {
        if (!m_entered)
            return;
        if (m_token.profiler)
            m_token.profiler->end_frame(m_token);
        m_token = {};
        m_entered = false;
    }

private:
    sgl::ref<sgl::Profiler> m_profiler;
    uint32_t m_site_id{0};
    sgl::ProfilerFrameToken m_token;
    bool m_entered{false};
};

template<typename T>
nb::ndarray<nb::numpy, const T> readonly_array(const std::vector<T>& values, nb::handle owner)
{
    size_t shape[1] = {values.size()};
    return nb::ndarray<nb::numpy, const T>(values.data(), 1, shape, owner);
}

template<typename T>
nb::ndarray<nb::numpy, const T>
readonly_matrix(const std::vector<T>& values, size_t row_count, size_t column_count, nb::handle owner)
{
    SGL_ASSERT(values.size() == row_count * column_count);
    size_t shape[2] = {row_count, column_count};
    return nb::ndarray<nb::numpy, const T>(values.data(), 2, shape, owner);
}

nb::ndarray<nb::numpy, const uint8_t> readonly_gpu_status_matrix(
    const std::vector<sgl::ProfilerFrameGpuStatus>& values,
    size_t row_count,
    size_t column_count,
    nb::handle owner
)
{
    static_assert(sizeof(sgl::ProfilerFrameGpuStatus) == sizeof(uint8_t));
    SGL_ASSERT(values.size() == row_count * column_count);
    size_t shape[2] = {row_count, column_count};
    return nb::ndarray<nb::numpy, const uint8_t>(reinterpret_cast<const uint8_t*>(values.data()), 2, shape, owner);
}

} // namespace

SGL_PY_EXPORT(utils_profiler)
{
    using namespace sgl;

    nb::sgl_enum<ProfilerTimelineType>(m, "ProfilerTimelineType", D(ProfilerTimelineType));
    nb::sgl_enum<ProfilerCaptureStopReason>(m, "ProfilerCaptureStopReason", D(ProfilerCaptureStopReason));
    nb::sgl_enum<ProfilerFrameGpuStatus>(m, "ProfilerFrameGpuStatus", D(ProfilerFrameGpuStatus));

    nb::class_<ProfilerDesc>(m, "ProfilerDesc", D(ProfilerDesc))
        .def(
            nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, bool>(),
            "thread_event_capacity"_a = 8192,
            "live_frame_count"_a = 120,
            "live_event_capacity"_a = 100000,
            "frame_stats_window_size"_a = 120,
            "gpu_query_pool_size"_a = 16384,
            "enable_auto_zones"_a = true,
            "enable_debug_groups"_a = false
        )
        .def_rw("thread_event_capacity", &ProfilerDesc::thread_event_capacity, D(ProfilerDesc, thread_event_capacity))
        .def_rw("live_frame_count", &ProfilerDesc::live_frame_count, D(ProfilerDesc, live_frame_count))
        .def_rw("live_event_capacity", &ProfilerDesc::live_event_capacity, D(ProfilerDesc, live_event_capacity))
        .def_rw(
            "frame_stats_window_size",
            &ProfilerDesc::frame_stats_window_size,
            D(ProfilerDesc, frame_stats_window_size)
        )
        .def_rw("gpu_query_pool_size", &ProfilerDesc::gpu_query_pool_size, D(ProfilerDesc, gpu_query_pool_size))
        .def_rw("enable_auto_zones", &ProfilerDesc::enable_auto_zones, D(ProfilerDesc, enable_auto_zones))
        .def_rw("enable_debug_groups", &ProfilerDesc::enable_debug_groups, D(ProfilerDesc, enable_debug_groups));

    nb::class_<ProfilerCaptureDesc>(m, "ProfilerCaptureDesc", D(ProfilerCaptureDesc))
        .def(nb::init<uint64_t>(), "max_memory_bytes"_a = 256ull * 1024ull * 1024ull)
        .def_rw("max_memory_bytes", &ProfilerCaptureDesc::max_memory_bytes, D(ProfilerCaptureDesc, max_memory_bytes));

    nb::class_<ProfilerSite>(m, "ProfilerSite", D(ProfilerSite))
        .def_ro("id", &ProfilerSite::id, D(ProfilerSite, id))
        .def_ro("file", &ProfilerSite::file, D(ProfilerSite, file))
        .def_ro("line", &ProfilerSite::line, D(ProfilerSite, line))
        .def_ro("function", &ProfilerSite::function, D(ProfilerSite, function))
        .def_ro("name", &ProfilerSite::name, D(ProfilerSite, name));

    nb::class_<ProfilerTimeline>(m, "ProfilerTimeline", D(ProfilerTimeline))
        .def_ro("type", &ProfilerTimeline::type, D(ProfilerTimeline, type))
        .def_ro("name", &ProfilerTimeline::name, D(ProfilerTimeline, name))
        .def_ro("thread_id", &ProfilerTimeline::thread_id, D(ProfilerTimeline, thread_id))
        .def_ro("device_id", &ProfilerTimeline::device_id, D(ProfilerTimeline, device_id))
        .def_ro("queue", &ProfilerTimeline::queue, D(ProfilerTimeline, queue));

    nb::class_<ProfilerFrame>(m, "ProfilerFrame", D(ProfilerFrame))
        .def_ro("index", &ProfilerFrame::index, D(ProfilerFrame, index))
        .def_ro("site_id", &ProfilerFrame::site_id, D(ProfilerFrame, site_id))
        .def_ro("timeline_id", &ProfilerFrame::timeline_id, D(ProfilerFrame, timeline_id))
        .def_ro("start_ns", &ProfilerFrame::start_ns, D(ProfilerFrame, start_ns))
        .def_ro("duration_ns", &ProfilerFrame::duration_ns, D(ProfilerFrame, duration_ns));

    nb::class_<ProfilerDurationStatistics>(m, "ProfilerDurationStatistics", D(ProfilerDurationStatistics))
        .def_ro("count", &ProfilerDurationStatistics::count, D(ProfilerDurationStatistics, count))
        .def_ro("total_ms", &ProfilerDurationStatistics::total_ms, D(ProfilerDurationStatistics, total_ms))
        .def_ro("minimum_ms", &ProfilerDurationStatistics::minimum_ms, D(ProfilerDurationStatistics, minimum_ms))
        .def_ro("maximum_ms", &ProfilerDurationStatistics::maximum_ms, D(ProfilerDurationStatistics, maximum_ms))
        .def_ro("mean_ms", &ProfilerDurationStatistics::mean_ms, D(ProfilerDurationStatistics, mean_ms))
        .def_ro(
            "standard_deviation_ms",
            &ProfilerDurationStatistics::standard_deviation_ms,
            D(ProfilerDurationStatistics, standard_deviation_ms)
        )
        .def_ro("p50_ms", &ProfilerDurationStatistics::p50_ms, D(ProfilerDurationStatistics, p50_ms))
        .def_ro("p90_ms", &ProfilerDurationStatistics::p90_ms, D(ProfilerDurationStatistics, p90_ms))
        .def_ro("p95_ms", &ProfilerDurationStatistics::p95_ms, D(ProfilerDurationStatistics, p95_ms))
        .def_ro("p99_ms", &ProfilerDurationStatistics::p99_ms, D(ProfilerDurationStatistics, p99_ms));

    nb::class_<ProfilerCallStatistics>(m, "ProfilerCallStatistics", D(ProfilerCallStatistics))
        .def_ro("count", &ProfilerCallStatistics::count, D(ProfilerCallStatistics, count))
        .def_ro("total_ms", &ProfilerCallStatistics::total_ms, D(ProfilerCallStatistics, total_ms))
        .def_ro("minimum_ms", &ProfilerCallStatistics::minimum_ms, D(ProfilerCallStatistics, minimum_ms))
        .def_ro("maximum_ms", &ProfilerCallStatistics::maximum_ms, D(ProfilerCallStatistics, maximum_ms))
        .def_ro("mean_ms", &ProfilerCallStatistics::mean_ms, D(ProfilerCallStatistics, mean_ms))
        .def_ro(
            "standard_deviation_ms",
            &ProfilerCallStatistics::standard_deviation_ms,
            D(ProfilerCallStatistics, standard_deviation_ms)
        );

    nb::class_<ProfilerDiagnostics>(m, "ProfilerDiagnostics", D(ProfilerDiagnostics))
        .def_ro(
            "producer_drop_count",
            &ProfilerDiagnostics::producer_drop_count,
            D(ProfilerDiagnostics, producer_drop_count)
        )
        .def_ro(
            "thread_event_queue_high_water_mark",
            &ProfilerDiagnostics::thread_event_queue_high_water_mark,
            D(ProfilerDiagnostics, thread_event_queue_high_water_mark)
        )
        .def_ro(
            "hierarchy_loss_count",
            &ProfilerDiagnostics::hierarchy_loss_count,
            D(ProfilerDiagnostics, hierarchy_loss_count)
        )
        .def_ro(
            "gpu_query_exhaustion_count",
            &ProfilerDiagnostics::gpu_query_exhaustion_count,
            D(ProfilerDiagnostics, gpu_query_exhaustion_count)
        )
        .def_ro(
            "pending_gpu_zone_count",
            &ProfilerDiagnostics::pending_gpu_zone_count,
            D(ProfilerDiagnostics, pending_gpu_zone_count)
        );

    nb::class_<ProfilerFrameStatsEntry>(m, "ProfilerFrameStatsEntry", D(ProfilerFrameStatsEntry))
        .def_ro("parent_index", &ProfilerFrameStatsEntry::parent_index, D(ProfilerFrameStatsEntry, parent_index))
        .def_ro("site_id", &ProfilerFrameStatsEntry::site_id, D(ProfilerFrameStatsEntry, site_id))
        .def_ro("name", &ProfilerFrameStatsEntry::name, D(ProfilerFrameStatsEntry, name))
        .def_ro(
            "cpu_time_per_frame",
            &ProfilerFrameStatsEntry::cpu_time_per_frame,
            D(ProfilerFrameStatsEntry, cpu_time_per_frame)
        )
        .def_ro(
            "gpu_time_per_frame",
            &ProfilerFrameStatsEntry::gpu_time_per_frame,
            D(ProfilerFrameStatsEntry, gpu_time_per_frame)
        )
        .def_ro(
            "cpu_time_per_call",
            &ProfilerFrameStatsEntry::cpu_time_per_call,
            D(ProfilerFrameStatsEntry, cpu_time_per_call)
        )
        .def_ro(
            "gpu_time_per_call",
            &ProfilerFrameStatsEntry::gpu_time_per_call,
            D(ProfilerFrameStatsEntry, gpu_time_per_call)
        );

    nb::class_<ProfilerFrameStats, Object>(m, "ProfilerFrameStats", D(ProfilerFrameStats))
        .def_prop_ro("sample_count", &ProfilerFrameStats::sample_count, D(ProfilerFrameStats, sample_count))
        .def_prop_ro("entry_count", &ProfilerFrameStats::entry_count, D(ProfilerFrameStats, entry_count))
        .def_prop_ro(
            "pending_frame_count",
            &ProfilerFrameStats::pending_frame_count,
            D(ProfilerFrameStats, pending_frame_count)
        )
        .def_prop_ro("latest_frame_ms", &ProfilerFrameStats::latest_frame_ms, D(ProfilerFrameStats, latest_frame_ms))
        .def_prop_ro(
            "frame_time",
            &ProfilerFrameStats::frame_time,
            nb::rv_policy::reference_internal,
            D(ProfilerFrameStats, frame_time)
        )
        .def_prop_ro(
            "entries",
            &ProfilerFrameStats::entries,
            nb::rv_policy::reference_internal,
            D(ProfilerFrameStats, entries)
        )
        .def_prop_ro(
            "sample_frame_index",
            [](ProfilerFrameStats& self)
            {
                return readonly_array(self.sample_frame_index(), nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerFrameStats, sample_frame_index)
        )
        .def_prop_ro(
            "sample_frame_time_ms",
            [](ProfilerFrameStats& self)
            {
                return readonly_array(self.sample_frame_time_ms(), nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerFrameStats, sample_frame_time_ms)
        )
        .def_prop_ro(
            "sample_call_count",
            [](ProfilerFrameStats& self)
            {
                return readonly_matrix(
                    self.sample_call_count(),
                    self.sample_count(),
                    self.entry_count(),
                    nb::cast(&self, nb::rv_policy::reference)
                );
            },
            D(ProfilerFrameStats, sample_call_count)
        )
        .def_prop_ro(
            "sample_cpu_time_ms",
            [](ProfilerFrameStats& self)
            {
                return readonly_matrix(
                    self.sample_cpu_time_ms(),
                    self.sample_count(),
                    self.entry_count(),
                    nb::cast(&self, nb::rv_policy::reference)
                );
            },
            D(ProfilerFrameStats, sample_cpu_time_ms)
        )
        .def_prop_ro(
            "sample_gpu_time_ms",
            [](ProfilerFrameStats& self)
            {
                return readonly_matrix(
                    self.sample_gpu_time_ms(),
                    self.sample_count(),
                    self.entry_count(),
                    nb::cast(&self, nb::rv_policy::reference)
                );
            },
            D(ProfilerFrameStats, sample_gpu_time_ms)
        )
        .def_prop_ro(
            "sample_gpu_status",
            [](ProfilerFrameStats& self)
            {
                return readonly_gpu_status_matrix(
                    self.sample_gpu_status(),
                    self.sample_count(),
                    self.entry_count(),
                    nb::cast(&self, nb::rv_policy::reference)
                );
            },
            D(ProfilerFrameStats, sample_gpu_status)
        )
        .def_prop_ro(
            "diagnostics",
            &ProfilerFrameStats::diagnostics,
            nb::rv_policy::reference_internal,
            D(ProfilerFrameStats, diagnostics)
        )
        .def("__str__", &ProfilerFrameStats::to_string, D(ProfilerFrameStats, to_string));

    nb::class_<ProfilerZoneChunk, Object>(m, "ProfilerZoneChunk", D(ProfilerZoneChunk))
        .def_prop_ro("count", &ProfilerZoneChunk::size, D(ProfilerZoneChunk, size))
        .def_prop_ro(
            "start_ns",
            [](ProfilerZoneChunk& self)
            {
                return readonly_array(self.start_ns, nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneChunk, start_ns)
        )
        .def_prop_ro(
            "duration_ns",
            [](ProfilerZoneChunk& self)
            {
                return readonly_array(self.duration_ns, nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneChunk, duration_ns)
        )
        .def_prop_ro(
            "correlation_id",
            [](ProfilerZoneChunk& self)
            {
                return readonly_array(self.correlation_id, nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneChunk, correlation_id)
        )
        .def_prop_ro(
            "timeline_id",
            [](ProfilerZoneChunk& self)
            {
                return readonly_array(self.timeline_id, nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneChunk, timeline_id)
        )
        .def_prop_ro(
            "site_id",
            [](ProfilerZoneChunk& self)
            {
                return readonly_array(self.site_id, nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneChunk, site_id)
        )
        .def_prop_ro(
            "parent_index",
            [](ProfilerZoneChunk& self)
            {
                return readonly_array(self.parent_index, nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneChunk, parent_index)
        )
        .def_prop_ro(
            "frame_index",
            [](ProfilerZoneChunk& self)
            {
                return readonly_array(self.frame_index, nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneChunk, frame_index)
        );

    nb::class_<ProfilerZoneSelection, Object>(m, "ProfilerZoneSelection", D(ProfilerZoneSelection))
        .def_prop_ro("count", &ProfilerZoneSelection::count, D(ProfilerZoneSelection, count))
        .def_prop_ro(
            "indices",
            [](ProfilerZoneSelection& self)
            {
                return readonly_array(self.indices(), nb::cast(&self, nb::rv_policy::reference));
            },
            D(ProfilerZoneSelection, indices)
        )
        .def("statistics", &ProfilerZoneSelection::statistics, D(ProfilerZoneSelection, statistics));

    nb::class_<ProfilerTrace, Object>(m, "ProfilerTrace", D(ProfilerTrace))
        .def_prop_ro("start_ns", &ProfilerTrace::start_ns, D(ProfilerTrace, start_ns))
        .def_prop_ro("stop_ns", &ProfilerTrace::stop_ns, D(ProfilerTrace, stop_ns))
        .def_prop_ro("stop_reason", &ProfilerTrace::stop_reason, D(ProfilerTrace, stop_reason))
        .def_prop_ro("truncated", &ProfilerTrace::truncated, D(ProfilerTrace, truncated))
        .def_prop_ro("memory_bytes", &ProfilerTrace::memory_bytes, D(ProfilerTrace, memory_bytes))
        .def_prop_ro(
            "diagnostics",
            &ProfilerTrace::diagnostics,
            nb::rv_policy::reference_internal,
            D(ProfilerTrace, diagnostics)
        )
        .def_prop_ro("sites", &ProfilerTrace::sites, nb::rv_policy::reference_internal, D(ProfilerTrace, sites))
        .def_prop_ro(
            "timelines",
            &ProfilerTrace::timelines,
            nb::rv_policy::reference_internal,
            D(ProfilerTrace, timelines)
        )
        .def_prop_ro("frames", &ProfilerTrace::frames, nb::rv_policy::reference_internal, D(ProfilerTrace, frames))
        .def_prop_ro(
            "zone_chunks",
            &ProfilerTrace::zone_chunks,
            nb::rv_policy::reference_internal,
            D(ProfilerTrace, zone_chunks)
        )
        .def_prop_ro("zone_count", &ProfilerTrace::zone_count, D(ProfilerTrace, zone_count))
        .def(
            "query_zones",
            &ProfilerTrace::query_zones,
            "name"_a = nb::none(),
            "timeline_type"_a = nb::none(),
            "frame_begin"_a = nb::none(),
            "frame_end"_a = nb::none(),
            "start_ns"_a = nb::none(),
            "end_ns"_a = nb::none(),
            D(ProfilerTrace, query_zones)
        )
        .def("write_to_json", &ProfilerTrace::write_to_json, "path"_a, D(ProfilerTrace, write_to_json));

    nb::class_<PyProfilerZoneScope>(m, "ProfilerZoneScope", D_NA(ProfilerZoneScope))
        .def("__enter__", &PyProfilerZoneScope::enter, nb::rv_policy::reference, D_NA(ProfilerZoneScope, enter))
        .def(
            "__exit__",
            &PyProfilerZoneScope::exit,
            "exc_type"_a.none(),
            "exc_val"_a.none(),
            "exc_tb"_a.none(),
            D_NA(ProfilerZoneScope, exit)
        );
    nb::class_<PyProfilerFrameScope>(m, "ProfilerFrameScope", D_NA(ProfilerFrameScope))
        .def("__enter__", &PyProfilerFrameScope::enter, nb::rv_policy::reference, D_NA(ProfilerFrameScope, enter))
        .def(
            "__exit__",
            &PyProfilerFrameScope::exit,
            "exc_type"_a.none(),
            "exc_val"_a.none(),
            "exc_tb"_a.none(),
            D_NA(ProfilerFrameScope, exit)
        );

    nb::class_<Profiler, Object>(m, "Profiler", D(Profiler))
        .def(nb::init<ProfilerDesc>(), "desc"_a = ProfilerDesc())
        .def(
            "__enter__",
            [](Profiler* self)
            {
                push_current_profiler(self);
                return self;
            },
            nb::rv_policy::reference,
            D_NA(Profiler, enter)
        )
        .def(
            "__exit__",
            [](Profiler* self, nb::object, nb::object, nb::object)
            {
                SGL_CHECK(current_profiler_or_null() == self, "Profiler activation stack changed before context exit");
                pop_current_profiler();
            },
            "exc_type"_a.none(),
            "exc_val"_a.none(),
            "exc_tb"_a.none(),
            D_NA(Profiler, exit)
        )
        .def_prop_rw("enabled", &Profiler::enabled, &Profiler::set_enabled, D(Profiler, enabled))
        .def_prop_rw(
            "enable_auto_zones",
            &Profiler::enable_auto_zones,
            &Profiler::set_enable_auto_zones,
            D(Profiler, enable_auto_zones)
        )
        .def_prop_rw(
            "enable_debug_groups",
            &Profiler::enable_debug_groups,
            &Profiler::set_enable_debug_groups,
            D(Profiler, enable_debug_groups)
        )
        .def_prop_ro("capture_active", &Profiler::capture_active, D(Profiler, capture_active))
        .def_prop_ro(
            "desc",
            [](const Profiler& self) -> ProfilerDesc
            {
                return self.desc();
            },
            D(Profiler, desc)
        )
        .def("start_capture", &Profiler::start_capture, "desc"_a = ProfilerCaptureDesc(), D(Profiler, start_capture))
        .def("stop_capture", &Profiler::stop_capture, D(Profiler, stop_capture))
        .def("discard_capture", &Profiler::discard_capture, D(Profiler, discard_capture))
        .def(
            "live_snapshot",
            [](Profiler* self)
            {
                self->release_retired_snapshots();
                return self->live_snapshot();
            },
            D(Profiler, live_snapshot)
        )
        .def(
            "frame_stats_snapshot",
            [](Profiler* self)
            {
                self->release_retired_snapshots();
                return self->frame_stats_snapshot();
            },
            D(Profiler, frame_stats_snapshot)
        )
        .def("clear_frame_stats", &Profiler::clear_frame_stats, D(Profiler, clear_frame_stats))
        .def("tick", &Profiler::tick, D(Profiler, tick))
        .def("flush", &Profiler::flush, D(Profiler, flush));

    m.def("current_profiler", &current_profiler, nb::rv_policy::reference, D(current_profiler));
    m.def(
        "current_profiler_or_null",
        &current_profiler_or_null,
        nb::rv_policy::reference,
        nb::sig("def current_profiler_or_null() -> Profiler | None"),
        D(current_profiler_or_null)
    );

    auto profiler_site_cache = std::make_shared<PyProfilerSiteCache>();
    m.def(
        "profile_zone",
        [profiler_site_cache](nb::str name, CommandEncoder* command_encoder)
        {
            ref<Profiler> profiler = current_profiler_ref();
            if (!profiler)
                return PyProfilerZoneScope();
            PyProfilerCallSite site = python_caller_site();
            const uint32_t site_id
                = profiler_site_cache->resolve(site.frame, std::move(site.code), site.instruction_offset, name);
            return PyProfilerZoneScope(std::move(profiler), site_id, command_encoder);
        },
        "name"_a,
        "command_encoder"_a = nb::none(),
        "Profile a named CPU zone and optional command-encoder GPU zone using the current profiler."
    );
    m.def(
        "profile_function",
        [profiler_site_cache](CommandEncoder* command_encoder)
        {
            ref<Profiler> profiler = current_profiler_ref();
            if (!profiler)
                return PyProfilerZoneScope();
            PyProfilerCallSite site = python_caller_site();
            const uint32_t site_id
                = profiler_site_cache->resolve(site.frame, std::move(site.code), site.instruction_offset);
            return PyProfilerZoneScope(std::move(profiler), site_id, command_encoder);
        },
        "command_encoder"_a = nb::none(),
        "Profile the calling Python function and optional command-encoder GPU work using the current profiler."
    );
    m.def(
        "profile_frame",
        [profiler_site_cache](nb::str name)
        {
            ref<Profiler> profiler = current_profiler_ref();
            if (!profiler)
                return PyProfilerFrameScope();
            PyProfilerCallSite site = python_caller_site();
            const uint32_t site_id
                = profiler_site_cache->resolve(site.frame, std::move(site.code), site.instruction_offset, name);
            return PyProfilerFrameScope(std::move(profiler), site_id);
        },
        "name"_a = "frame",
        "Record a named frame boundary using the current profiler."
    );
}
