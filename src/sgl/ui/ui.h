// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/ui/fwd.h"

#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/timer.h"

#include "sgl/device/fwd.h"
#include "sgl/device/formats.h"

#include <map>

struct ImGuiContext;
struct ImFont;
struct ImDrawData;

namespace sgl::ui {

class SGL_API Context : public Object {
    SGL_OBJECT(Context)
public:
    Context(ref<Device> device);
    ~Context();

    ref<Screen> screen() const { return m_screen; }

    ImFont* get_font(const char* name);

    void new_frame(uint32_t width, uint32_t height);
    void render(TextureView* texture_view, CommandEncoder* command_encoder);
    void render(Texture* texture, CommandEncoder* command_encoder);

    bool handle_keyboard_event(const KeyboardEvent& event);
    bool handle_mouse_event(const MouseEvent& event);

    void process_events();

private:
    void init_rasterizer();
    void init_sw_rasterizer();

    RenderPipeline* get_pipeline(Format format);

    void draw(
        ImDrawData* draw_data,
        Buffer* vertex_buffer,
        Buffer* index_buffer,
        TextureView* texture_view,
        CommandEncoder* command_encoder
    );

    void draw_sw(
        ImDrawData* draw_data,
        Buffer* vertex_buffer,
        Buffer* index_buffer,
        TextureView* texture_view,
        CommandEncoder* command_encoder
    );

    static constexpr uint32_t FRAME_COUNT = 3;

    ref<Device> m_device;
    ImGuiContext* m_imgui_context;

    ref<Screen> m_screen;

    uint32_t m_frame_index{0};
    Timer m_frame_timer;

    ref<Sampler> m_sampler;
    ref<Buffer> m_vertex_buffers[FRAME_COUNT];
    ref<Buffer> m_index_buffers[FRAME_COUNT];
    ref<ShaderProgram> m_program;
    ref<Texture> m_font_texture;
    ref<InputLayout> m_input_layout;
    std::map<Format, ref<RenderPipeline>> m_pipelines;
    ref<ComputePipeline> m_compute_pipeline;

    std::map<std::string, ImFont*> m_fonts;
};

} // namespace sgl::ui

// Extend ImGui with some additional convenience functions.
namespace ImGui {

SGL_API void PushFont(const char* name);

}
