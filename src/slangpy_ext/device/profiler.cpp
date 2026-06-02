// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/profiler.h"

#include "sgl/device/command.h"

SGL_PY_EXPORT(device_profiler)
{
    using namespace sgl;

    nb::sgl_enum_flags<ProfilerZoneFlags>(m, "ProfilerZoneFlags");
    nb::sgl_enum<ProfilerTimelineType>(m, "ProfilerTimelineType");

    nb::class_<ProfilerDesc>(m, "ProfilerDesc", "Descriptor for creating a Profiler.").def(nb::init<>());

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

    nb::class_<ProfilerTrace, Object>(m, "ProfilerTrace").def("write_to_json", &ProfilerTrace::write_to_json, "path"_a);

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
        .def_prop_ro("desc", &Profiler::desc)
        .def("tick", &Profiler::tick)
        .def("trace_snapshot", &Profiler::trace_snapshot);

    m.def("push_current_profiler", &push_current_profiler, "profiler"_a);
    m.def("pop_current_profiler", &pop_current_profiler, nb::rv_policy::reference);
    m.def("current_profiler", &current_profiler, nb::rv_policy::reference);
    m.def("current_profiler_or_null", &current_profiler_or_null, nb::rv_policy::reference);
}
