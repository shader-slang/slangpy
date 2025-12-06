// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/app/app.h"
#include "sgl/device/command.h"
#include "sgl/device/resource.h"
#include "sgl/ui/widgets.h"

namespace sgl {

SGL_DICT_TO_DESC_BEGIN(AppDesc)
SGL_DICT_TO_DESC_FIELD(device, ref<Device>)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AppWindowDesc)
SGL_DICT_TO_DESC_FIELD(width, uint32_t)
SGL_DICT_TO_DESC_FIELD(height, uint32_t)
SGL_DICT_TO_DESC_FIELD(title, std::string)
SGL_DICT_TO_DESC_FIELD(mode, WindowMode)
SGL_DICT_TO_DESC_FIELD(resizable, bool)
SGL_DICT_TO_DESC_FIELD(surface_format, Format)
SGL_DICT_TO_DESC_FIELD(enable_vsync, bool)
SGL_DICT_TO_DESC_END()

class PyApp : App {
public:
    NB_TRAMPOLINE(App, 10);

    PyApp(AppDesc desc)
        : App(desc)
    {
    }
};

class PyAppWindow : AppWindow {
public:
    NB_TRAMPOLINE(AppWindow, 10);

    PyAppWindow(AppWindowDesc desc)
        : AppWindow(desc)
    {
    }

    void render(RenderContext render_context) override { NB_OVERRIDE(render, render_context); }

    void on_resize(uint32_t width, uint32_t height) override { NB_OVERRIDE(on_resize, width, height); }
    void on_keyboard_event(const KeyboardEvent& event) override { NB_OVERRIDE(on_keyboard_event, event); }
    void on_mouse_event(const MouseEvent& event) override { NB_OVERRIDE(on_mouse_event, event); }
    void on_gamepad_event(const GamepadEvent& event) override { NB_OVERRIDE(on_gamepad_event, event); }
    void on_drop_files(std::span<const char*> files) override { NB_OVERRIDE(on_drop_files, files); }
};

} // namespace sgl

SGL_PY_EXPORT(app_app)
{
    using namespace sgl;

    nb::class_<AppDesc>(m, "AppDesc", D(AppDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](AppDesc* self, nb::dict dict)
            {
                new (self) AppDesc(dict_to_AppDesc(dict));
            }
        )
        .def_rw("device", &AppDesc::device, D(AppDesc, device));
    nb::implicitly_convertible<nb::dict, AppDesc>();

    nb::class_<App, PyApp, Object>(m, "App", D(App))
        .def(nb::init<AppDesc>())
        .def(
            "__init__",
            [](App* self, ref<Device> device)
            {
                new (self) PyApp({
                    .device = device,
                });
            },
            "device"_a.none() = nb::none(),
            D(App, App)
        )
        .def_prop_ro("device", &App::device, D(App, device))
        .def("run", &App::run, D(App, run))
        .def("run_frame", &App::run_frame, D(App, run_frame))
        .def("terminate", &App::terminate, D(App, terminate));

    nb::class_<AppWindowDesc>(m, "AppWindowDesc", D(AppWindowDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](AppWindowDesc* self, nb::dict dict)
            {
                new (self) AppWindowDesc(dict_to_AppWindowDesc(dict));
            }
        )
        .def_rw("width", &AppWindowDesc::width, D(AppWindowDesc, width))
        .def_rw("height", &AppWindowDesc::height, D(AppWindowDesc, height))
        .def_rw("title", &AppWindowDesc::title, D(AppWindowDesc, title))
        .def_rw("mode", &AppWindowDesc::mode, D(AppWindowDesc, mode))
        .def_rw("resizable", &AppWindowDesc::resizable, D(AppWindowDesc, resizable))
        .def_rw("surface_format", &AppWindowDesc::surface_format, D(AppWindowDesc, surface_format))
        .def_rw("enable_vsync", &AppWindowDesc::enable_vsync, D(AppWindowDesc, enable_vsync));
    nb::implicitly_convertible<nb::dict, AppWindowDesc>();

    nb::class_<AppWindow, PyAppWindow, Object> app_window(m, "AppWindow", D(AppWindow));

    nb::class_<AppWindow::RenderContext>(app_window, "RenderContext", D(AppWindow, RenderContext))
        .def_ro(
            "surface_texture",
            &AppWindow::RenderContext::surface_texture,
            D(AppWindow, RenderContext, surface_texture)
        )
        .def_ro(
            "command_encoder",
            &AppWindow::RenderContext::command_encoder,
            D(AppWindow, RenderContext, command_encoder)
        );

    app_window //
        .def(
            "__init__",
            [](AppWindow* self,
               App* app,
               uint32_t width,
               uint32_t height,
               std::string title,
               WindowMode mode,
               bool resizable,
               Format surface_format,
               bool enable_vsync)
            {
                new (self) PyAppWindow({
                    .app = app,
                    .width = width,
                    .height = height,
                    .title = title,
                    .mode = mode,
                    .resizable = resizable,
                    .surface_format = surface_format,
                    .enable_vsync = enable_vsync,
                });
            },
            "app"_a,
            "width"_a = AppWindowDesc().width,
            "height"_a = AppWindowDesc().height,
            "title"_a = AppWindowDesc().title,
            "mode"_a = AppWindowDesc().mode,
            "resizable"_a = AppWindowDesc().resizable,
            "surface_format"_a = AppWindowDesc().surface_format,
            "enable_vsync"_a = AppWindowDesc().enable_vsync,
            D(App, App)
        )
        .def_prop_ro("device", &AppWindow::device, D(AppWindow, device))
        .def_prop_ro("screen", &AppWindow::screen, D(AppWindow, screen))
        .def("render", &AppWindow::render, "render_context"_a, D(AppWindow, render))
        .def("on_resize", &AppWindow::on_resize, "width"_a, "height"_a, D(AppWindow, on_resize))
        .def("on_keyboard_event", &AppWindow::on_keyboard_event, "event"_a, D(AppWindow, on_keyboard_event))
        .def("on_mouse_event", &AppWindow::on_mouse_event, "event"_a, D(AppWindow, on_mouse_event))
        .def("on_gamepad_event", &AppWindow::on_gamepad_event, "event"_a, D(AppWindow, on_gamepad_event))
        .def("on_drop_files", &AppWindow::on_drop_files, "files"_a, D(AppWindow, on_drop_files));
}
