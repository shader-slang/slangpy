// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/ui/ui.h"
#include "sgl/ui/widgets.h"

#include "sgl/core/input.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"

#undef D
#define D(...) DOC(sgl, ui, __VA_ARGS__)

namespace sgl {

template<>
struct GcHelper<ui::Context> {
    void traverse(ui::Context*, GcVisitor& visitor) { visitor("screen"); }
    void clear(ui::Context*) { }
};

} // namespace sgl

namespace {

struct MarshaledDrawData {
    std::vector<std::vector<sgl::ui::DrawCommand>> commands;
    std::vector<sgl::ui::DrawList> draw_lists;
    sgl::ui::DrawData draw_data;
};

MarshaledDrawData marshal_draw_data(nb::handle draw_data_obj)
{
    using namespace sgl;

    MarshaledDrawData result;

    nb::object cmd_lists = draw_data_obj.attr("cmd_lists");
    const size_t cmd_list_count = nb::len(cmd_lists);
    result.commands.reserve(cmd_list_count);
    result.draw_lists.reserve(cmd_list_count);

    for (nb::handle cmd_list_handle : cmd_lists) {
        nb::object cmd_buffer = cmd_list_handle.attr("cmd_buffer");
        auto& list_commands = result.commands.emplace_back();
        list_commands.reserve(nb::len(cmd_buffer));

        for (nb::handle cmd_handle : cmd_buffer) {
            nb::object clip_rect = cmd_handle.attr("clip_rect");
            list_commands.push_back(
                ui::DrawCommand{
                    .clip_rect = float4(
                        nb::cast<float>(clip_rect.attr("x")),
                        nb::cast<float>(clip_rect.attr("y")),
                        nb::cast<float>(clip_rect.attr("z")),
                        nb::cast<float>(clip_rect.attr("w"))
                    ),
                    .elem_count = nb::cast<uint32_t>(cmd_handle.attr("elem_count")),
                    .idx_offset = nb::cast<uint32_t>(cmd_handle.attr("idx_offset")),
                    .vtx_offset = nb::cast<uint32_t>(cmd_handle.attr("vtx_offset")),
                    .texture_id = nb::cast<uintptr_t>(cmd_handle.attr("get_tex_id")()),
                }
            );
        }

        nb::object vtx_buffer = cmd_list_handle.attr("vtx_buffer");
        nb::object idx_buffer = cmd_list_handle.attr("idx_buffer");

        result.draw_lists.push_back(
            ui::DrawList{
                .vertex_data = nb::cast<uintptr_t>(vtx_buffer.attr("data_address")()),
                .vertex_count = nb::cast<uint32_t>(vtx_buffer.attr("size")()),
                .index_data = nb::cast<uintptr_t>(idx_buffer.attr("data_address")()),
                .index_count = nb::cast<uint32_t>(idx_buffer.attr("size")()),
                .commands = std::span<const ui::DrawCommand>(list_commands.data(), list_commands.size()),
            }
        );
    }

    nb::object display_pos = draw_data_obj.attr("display_pos");
    nb::object display_size = draw_data_obj.attr("display_size");
    nb::object framebuffer_scale = draw_data_obj.attr("framebuffer_scale");

    // Detect index size from the binding module (e.g. imgui.INDEX_SIZE).
    uint32_t index_size = sizeof(uint32_t);
    nb::object imgui_mod = draw_data_obj.type().attr("__module__");
    nb::object mod = nb::module_::import_(nb::cast<const char*>(imgui_mod));
    if (nb::hasattr(mod, "INDEX_SIZE"))
        index_size = nb::cast<uint32_t>(mod.attr("INDEX_SIZE"));

    result.draw_data = ui::DrawData{
        .display_pos = float2(nb::cast<float>(display_pos.attr("x")), nb::cast<float>(display_pos.attr("y"))),
        .display_size = float2(nb::cast<float>(display_size.attr("x")), nb::cast<float>(display_size.attr("y"))),
        .framebuffer_scale
        = float2(nb::cast<float>(framebuffer_scale.attr("x")), nb::cast<float>(framebuffer_scale.attr("y"))),
        .draw_lists = std::span<const ui::DrawList>(result.draw_lists.data(), result.draw_lists.size()),
        .total_vtx_count = nb::cast<uint32_t>(draw_data_obj.attr("total_vtx_count")),
        .total_idx_count = nb::cast<uint32_t>(draw_data_obj.attr("total_idx_count")),
        .index_size = index_size,
    };
    return result;
}

} // namespace

SGL_PY_EXPORT(ui)
{
    using namespace sgl;

    nb::module_ ui = nb::module_::import_("slangpy.ui");

    nb::class_<ui::Context, Object>(ui, "Context", gc_helper_type_slots<ui::Context>(), D(Context))
        .def(nb::init<ref<Device>>(), "device"_a)
        .def("begin_frame", &ui::Context::begin_frame, "width"_a, "height"_a, D(Context, begin_frame))
        .def(
            "end_frame",
            nb::overload_cast<TextureView*, CommandEncoder*>(&ui::Context::end_frame),
            "texture_view"_a,
            "command_encoder"_a,
            D(Context, end_frame)
        )
        .def(
            "end_frame",
            nb::overload_cast<Texture*, CommandEncoder*>(&ui::Context::end_frame),
            "texture"_a,
            "command_encoder"_a,
            D(Context, end_frame, 2)
        )
        .def(
            "_render_marshaled_draw_data",
            [](ui::Context& self, nb::handle draw_data, TextureView* texture_view, CommandEncoder* command_encoder)
            {
                auto marshaled = marshal_draw_data(draw_data);
                self.render_draw_data(marshaled.draw_data, texture_view, command_encoder);
            },
            "draw_data"_a,
            "texture_view"_a,
            "command_encoder"_a
        )
        .def(
            "_render_marshaled_draw_data",
            [](ui::Context& self, nb::handle draw_data, Texture* texture, CommandEncoder* command_encoder)
            {
                auto marshaled = marshal_draw_data(draw_data);
                self.render_draw_data(marshaled.draw_data, texture, command_encoder);
            },
            "draw_data"_a,
            "texture"_a,
            "command_encoder"_a
        )
        .def("texture_id", &ui::Context::texture_id, "texture"_a)
        .def("handle_keyboard_event", &ui::Context::handle_keyboard_event, "event"_a, D(Context, handle_keyboard_event))
        .def("handle_mouse_event", &ui::Context::handle_mouse_event, "event"_a, D(Context, handle_mouse_event))
        .def_prop_ro("screen", &ui::Context::screen, D(Context, screen));
}
