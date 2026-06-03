// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/profiler.h"

#include "sgl/device/command.h"

#include <string>
#include <utility>

namespace {

class PyProfilerZoneContext {
public:
    PyProfilerZoneContext(
        sgl::Profiler* profiler,
        std::string name,
        sgl::CommandEncoder* encoder,
        sgl::ProfilerZoneFlags flags
    )
        : m_profiler(profiler)
        , m_name(std::move(name))
        , m_encoder(encoder)
        , m_flags(flags)
    {
    }

    PyProfilerZoneContext* enter()
    {
        if (!m_profiler)
            return this;

        static const sgl::ProfilerSourceLocation* source_location
            = sgl::Profiler::intern_source_location("<python>", 0, "Profiler.zone");
        const char* name = m_name.empty() ? nullptr : sgl::Profiler::intern_name(m_name);
        m_active = m_profiler->begin_zone(source_location, name, m_encoder, m_flags);
        return this;
    }

    void exit(nb::object, nb::object, nb::object)
    {
        if (m_active && m_profiler)
            m_profiler->end_zone(m_encoder, m_flags);
        m_active = false;
    }

private:
    sgl::Profiler* m_profiler{nullptr};
    std::string m_name;
    sgl::CommandEncoder* m_encoder{nullptr};
    sgl::ProfilerZoneFlags m_flags{sgl::ProfilerZoneFlags::none};
    bool m_active{false};
};

class PyProfilerFrameContext {
public:
    PyProfilerFrameContext(sgl::Profiler* profiler, std::string name)
        : m_profiler(profiler)
        , m_name(std::move(name))
    {
    }

    PyProfilerFrameContext* enter()
    {
        if (!m_profiler)
            return this;

        static const sgl::ProfilerSourceLocation* source_location
            = sgl::Profiler::intern_source_location("<python>", 0, "Profiler.frame");
        const char* name = m_name.empty() ? nullptr : sgl::Profiler::intern_name(m_name);
        m_active = m_profiler->begin_frame(source_location, name);
        return this;
    }

    void exit(nb::object, nb::object, nb::object)
    {
        if (m_active && m_profiler)
            m_profiler->end_frame();
        m_active = false;
    }

private:
    sgl::Profiler* m_profiler{nullptr};
    std::string m_name;
    bool m_active{false};
};

} // namespace

SGL_PY_EXPORT(device_profiler)
{
    using namespace sgl;

    nb::sgl_enum_flags<ProfilerZoneFlags>(m, "ProfilerZoneFlags");
    nb::sgl_enum<ProfilerTimelineType>(m, "ProfilerTimelineType");

    nb::class_<ProfilerDesc>(m, "ProfilerDesc", "Descriptor for creating a Profiler.")
        .def(nb::init<>())
        .def_rw("frame_stats_enabled", &ProfilerDesc::frame_stats_enabled)
        .def_rw("trace_enabled_on_start", &ProfilerDesc::trace_enabled_on_start)
        .def_rw("stats_window_size", &ProfilerDesc::stats_window_size)
        .def_rw("gpu_query_pool_size", &ProfilerDesc::gpu_query_pool_size)
        .def_rw("gpu_query_block_size", &ProfilerDesc::gpu_query_block_size)
        .def_rw("auto_zones_enabled", &ProfilerDesc::auto_zones_enabled)
        .def_rw("debug_groups_enabled", &ProfilerDesc::debug_groups_enabled);

    nb::class_<ProfilerSourceLocation>(m, "ProfilerSourceLocation", "Stable metadata for a profiler source callsite.")
        .def(nb::init<>())
        .def_prop_ro(
            "file",
            [](const ProfilerSourceLocation& self)
            {
                return self.file ? self.file : "";
            }
        )
        .def_prop_ro(
            "line",
            [](const ProfilerSourceLocation& self)
            {
                return self.line;
            }
        )
        .def_prop_ro(
            "function",
            [](const ProfilerSourceLocation& self)
            {
                return self.function ? self.function : "";
            }
        );

    nb::class_<ProfilerTimelineInfo>(m, "ProfilerTimelineInfo", "Metadata for a profiler timeline/lane.")
        .def(nb::init<>())
        .def_rw("type", &ProfilerTimelineInfo::type)
        .def_rw("name", &ProfilerTimelineInfo::name)
        .def_rw("thread_id", &ProfilerTimelineInfo::thread_id)
        .def_rw("device_id", &ProfilerTimelineInfo::device_id)
        .def_rw("queue", &ProfilerTimelineInfo::queue);

    nb::class_<ProfilerTimelineRecord>(m, "ProfilerTimelineRecord")
        .def_ro("id", &ProfilerTimelineRecord::id)
        .def_ro("type", &ProfilerTimelineRecord::type)
        .def_ro("name", &ProfilerTimelineRecord::name)
        .def_ro("thread_id", &ProfilerTimelineRecord::thread_id)
        .def_ro("device_id", &ProfilerTimelineRecord::device_id)
        .def_ro("queue", &ProfilerTimelineRecord::queue);

    nb::class_<ProfilerSourceRecord>(m, "ProfilerSourceRecord")
        .def_ro("id", &ProfilerSourceRecord::id)
        .def_ro("file", &ProfilerSourceRecord::file)
        .def_ro("line", &ProfilerSourceRecord::line)
        .def_ro("original_function", &ProfilerSourceRecord::original_function)
        .def_ro("display_function", &ProfilerSourceRecord::display_function);

    nb::class_<ProfilerNameRecord>(m, "ProfilerNameRecord")
        .def_ro("id", &ProfilerNameRecord::id)
        .def_ro("name", &ProfilerNameRecord::name);

    nb::class_<ProfilerFrameRecord>(m, "ProfilerFrameRecord")
        .def_ro("id", &ProfilerFrameRecord::id)
        .def_ro("name_id", &ProfilerFrameRecord::name_id)
        .def_ro("source_id", &ProfilerFrameRecord::source_id)
        .def_ro("start_timestamp", &ProfilerFrameRecord::start_timestamp)
        .def_ro("end_timestamp", &ProfilerFrameRecord::end_timestamp);

    nb::class_<ProfilerZoneRecord>(m, "ProfilerZoneRecord")
        .def_ro("id", &ProfilerZoneRecord::id)
        .def_ro("event_id", &ProfilerZoneRecord::event_id)
        .def_ro("child_index_begin", &ProfilerZoneRecord::child_index_begin)
        .def_ro("child_index_count", &ProfilerZoneRecord::child_index_count)
        .def_ro("timeline_id", &ProfilerZoneRecord::timeline_id)
        .def_ro("frame_id", &ProfilerZoneRecord::frame_id)
        .def_ro("source_id", &ProfilerZoneRecord::source_id)
        .def_ro("name_id", &ProfilerZoneRecord::name_id)
        .def_ro("start_timestamp", &ProfilerZoneRecord::start_timestamp)
        .def_ro("end_timestamp", &ProfilerZoneRecord::end_timestamp);

    nb::class_<ProfilerStatValue>(m, "ProfilerStatValue")
        .def_ro("valid", &ProfilerStatValue::valid)
        .def_ro("last_ms", &ProfilerStatValue::last_ms)
        .def_ro("min_ms", &ProfilerStatValue::min_ms)
        .def_ro("max_ms", &ProfilerStatValue::max_ms)
        .def_ro("average_ms", &ProfilerStatValue::average_ms)
        .def_ro("stddev_ms", &ProfilerStatValue::stddev_ms)
        .def_ro("sample_count", &ProfilerStatValue::sample_count);

    nb::class_<ProfilerStatsNode>(m, "ProfilerStatsNode")
        .def_ro("id", &ProfilerStatsNode::id)
        .def_ro("parent_id", &ProfilerStatsNode::parent_id)
        .def_ro("child_index_begin", &ProfilerStatsNode::child_index_begin)
        .def_ro("child_index_count", &ProfilerStatsNode::child_index_count)
        .def_ro("source_id", &ProfilerStatsNode::source_id)
        .def_ro("name_id", &ProfilerStatsNode::name_id)
        .def_ro("pending_gpu_sample_count", &ProfilerStatsNode::pending_gpu_sample_count)
        .def_ro("cpu", &ProfilerStatsNode::cpu)
        .def_ro("gpu", &ProfilerStatsNode::gpu);

    nb::class_<ProfilerTrace, Object>(m, "ProfilerTrace")
        .def_prop_ro("timelines", &ProfilerTrace::timelines, nb::rv_policy::reference_internal)
        .def_prop_ro("sources", &ProfilerTrace::sources, nb::rv_policy::reference_internal)
        .def_prop_ro("names", &ProfilerTrace::names, nb::rv_policy::reference_internal)
        .def_prop_ro("frames", &ProfilerTrace::frames, nb::rv_policy::reference_internal)
        .def_prop_ro("zones", &ProfilerTrace::zones, nb::rv_policy::reference_internal)
        .def_prop_ro("child_indices", &ProfilerTrace::child_indices, nb::rv_policy::reference_internal)
        .def_prop_ro("root_indices", &ProfilerTrace::root_indices, nb::rv_policy::reference_internal)
        .def("write_to_json", &ProfilerTrace::write_to_json, "path"_a);

    nb::class_<ProfilerStats, Object>(m, "ProfilerStats")
        .def_prop_ro("sources", &ProfilerStats::sources, nb::rv_policy::reference_internal)
        .def_prop_ro("names", &ProfilerStats::names, nb::rv_policy::reference_internal)
        .def_prop_ro("nodes", &ProfilerStats::nodes, nb::rv_policy::reference_internal)
        .def_prop_ro("child_indices", &ProfilerStats::child_indices, nb::rv_policy::reference_internal)
        .def_prop_ro("completed_frame_count", &ProfilerStats::completed_frame_count)
        .def_prop_ro("window_size", &ProfilerStats::window_size);

    nb::class_<PyProfilerZoneContext>(m, "_ProfilerZoneContext")
        .def("__enter__", &PyProfilerZoneContext::enter, nb::rv_policy::reference_internal)
        .def("__exit__", &PyProfilerZoneContext::exit, "exc_type"_a.none(), "exc_val"_a.none(), "exc_tb"_a.none());

    nb::class_<PyProfilerFrameContext>(m, "_ProfilerFrameContext")
        .def("__enter__", &PyProfilerFrameContext::enter, nb::rv_policy::reference_internal)
        .def("__exit__", &PyProfilerFrameContext::exit, "exc_type"_a.none(), "exc_val"_a.none(), "exc_tb"_a.none());

    nb::class_<Profiler, Object>(m, "Profiler", D(Profiler))
        .def(nb::init<ProfilerDesc>(), "desc"_a = ProfilerDesc())
        .def(
            "__enter__",
            [](Profiler* self) -> Profiler*
            {
                push_current_profiler(self);
                return self;
            },
            nb::rv_policy::reference
        )
        .def(
            "__exit__",
            [](Profiler* self, nb::object /*exc_type*/, nb::object /*exc_val*/, nb::object /*exc_tb*/)
            {
                if (current_profiler_or_null() == self)
                    pop_current_profiler();
            },
            "exc_type"_a.none(),
            "exc_val"_a.none(),
            "exc_tb"_a.none()
        )
        .def_prop_rw("enabled", &Profiler::enabled, &Profiler::set_enabled)
        .def_prop_rw("auto_zones_enabled", &Profiler::auto_zones_enabled, &Profiler::set_auto_zones_enabled)
        .def_prop_rw("debug_groups_enabled", &Profiler::debug_groups_enabled, &Profiler::set_debug_groups_enabled)
        .def_prop_rw("frame_stats_enabled", &Profiler::frame_stats_enabled, &Profiler::set_frame_stats_enabled)
        .def_prop_rw("stats_window_size", &Profiler::stats_window_size, &Profiler::set_stats_window_size)
        .def_prop_ro("desc", &Profiler::desc)
        .def(
            "zone",
            [](Profiler* self, nb::object name, CommandEncoder* command_encoder, ProfilerZoneFlags flags)
            {
                std::string zone_name;
                if (!name.is_none())
                    zone_name = nb::cast<std::string>(name);
                return PyProfilerZoneContext(self, std::move(zone_name), command_encoder, flags);
            },
            "name"_a = nb::none(),
            "command_encoder"_a = nullptr,
            "flags"_a = ProfilerZoneFlags::none
        )
        .def(
            "frame",
            [](Profiler* self, nb::object name)
            {
                std::string frame_name;
                if (!name.is_none())
                    frame_name = nb::cast<std::string>(name);
                return PyProfilerFrameContext(self, std::move(frame_name));
            },
            "name"_a = nb::none()
        )
        .def("start_trace", &Profiler::start_trace, "clear"_a = true)
        .def("stop_trace", &Profiler::stop_trace)
        .def("clear_trace", &Profiler::clear_trace)
        .def("tick", &Profiler::tick)
        .def("flush", &Profiler::flush)
        .def("trace_snapshot", &Profiler::trace_snapshot)
        .def("stats_snapshot", &Profiler::stats_snapshot);

    m.def("push_current_profiler", &push_current_profiler, "profiler"_a);
    m.def("pop_current_profiler", &pop_current_profiler, nb::rv_policy::reference);
    m.def("current_profiler", &current_profiler, nb::rv_policy::reference);
    m.def("current_profiler_or_null", &current_profiler_or_null, nb::rv_policy::reference);
}
