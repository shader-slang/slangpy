// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/platform.h"

SGL_PY_EXPORT(core_platform)
{
    using namespace sgl::platform;

    nb::class_<sgl::WindowHandle>(m, "WindowHandle", D(WindowHandle))
#if SGL_WINDOWS
        .def(
            "__init__",
            [](sgl::WindowHandle* self, uintptr_t hwnd) { new (self) sgl::WindowHandle{reinterpret_cast<HWND>(hwnd)}; },
            "hwnd"_a
        )
#elif SGL_LINUX
        .def(
            "__init__",
            [](sgl::WindowHandle* self, uintptr_t xdisplay, uint32_t xwindow)
            { new (self) sgl::WindowHandle{reinterpret_cast<void*>(xdisplay), xwindow}; },
            "xdisplay"_a,
            "xwindow"_a
        )
#elif SGL_MACOS
        .def(
            "__init__",
            [](sgl::WindowHandle* self, uintptr_t nswindow)
            { new (self) sgl::WindowHandle{reinterpret_cast<void*>(nswindow)}; },
            "nswindow"_a
        )
#endif
        ;

    nb::module_ platform = nb::module_::import_("slangpy.platform");

    nb::class_<FileDialogFilter>(platform, "FileDialogFilter", D(platform, FileDialogFilter))
        .def(nb::init<>())
        .def(nb::init<std::string_view, std::string_view>(), "name"_a, "pattern"_a)
        .def(nb::init_implicit<std::pair<std::string_view, std::string_view>>())
        .def_rw("name", &FileDialogFilter::name, D(platform, FileDialogFilter, name))
        .def_rw("pattern", &FileDialogFilter::pattern, D(platform, FileDialogFilter, pattern));

    platform.def(
        "open_file_dialog",
        [](FileDialogFilterList filters) { return open_file_dialog(filters); },
        "filters"_a = FileDialogFilterList(),
        D(platform, open_file_dialog)
    );
    platform.def(
        "save_file_dialog",
        [](FileDialogFilterList filters) { return save_file_dialog(filters); },
        "filters"_a = FileDialogFilterList(),
        D(platform, save_file_dialog)
    );
    platform.def("choose_folder_dialog", &choose_folder_dialog, D(platform, choose_folder_dialog));

    platform.def("display_scale_factor", &display_scale_factor, D(platform, display_scale_factor));

    platform.def("executable_path", &executable_path, D(platform, executable_path));
    platform.def("executable_directory", &executable_directory, D(platform, executable_directory));
    platform.def("executable_name", &executable_name, D(platform, executable_name));
    platform.def("app_data_directory", &app_data_directory, D(platform, app_data_directory));
    platform.def("home_directory", &home_directory, D(platform, home_directory));
    platform.def("project_directory", &project_directory, D(platform, project_directory));
    platform.def("runtime_directory", &runtime_directory, D(platform, runtime_directory));

    platform.attr("page_size") = page_size();

    nb::class_<MemoryStats>(platform, "MemoryStats", D(platform, MemoryStats))
        .def_ro("rss", &MemoryStats::rss, D(platform, MemoryStats, rss))
        .def_ro("peak_rss", &MemoryStats::peak_rss, D(platform, MemoryStats, peak_rss));

    platform.def("memory_stats", &memory_stats, D(platform, memory_stats));
}
