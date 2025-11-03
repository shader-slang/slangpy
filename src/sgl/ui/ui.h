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

namespace sgl::ui {

class SGL_API Context : public Object {
    SGL_OBJECT(Context)
public:
    Context(ref<Device> device);
    ~Context();

    /// The main screen widget.
    ref<Screen> screen() const { return m_screen; }

    ImFont* get_font(const char* name);

    /// Begin a new ImGui frame and renders the main screen widget.
    /// ImGui widget calls are generally only valid between `begin_frame` and `end_frame`.
    /// \param width Render texture width
    /// \param height Render texture height
    void begin_frame(uint32_t width, uint32_t height);

    /// End the ImGui frame and renders the UI to the provided texture.
    /// \param texture_view Texture view to render to
    /// \param command_encoder Command encoder to encode commands to
    void end_frame(TextureView* texture_view, CommandEncoder* command_encoder);

    /// End the ImGui frame and renders the UI to the provided texture.
    /// \param texture Texture to render to
    /// \param command_encoder Command encoder to encode commands to
    void end_frame(Texture* texture, CommandEncoder* command_encoder);

    /// Pass a keyboard event to the UI context.
    /// \param event Keyboard event
    /// \return Returns true if event was consumed.
    bool handle_keyboard_event(const KeyboardEvent& event);

    /// Pass a mouse event to the UI context.
    /// \param event Mouse event
    /// \return Returns true if event was consumed.
    bool handle_mouse_event(const MouseEvent& event);

private:
    RenderPipeline* get_pipeline(Format format);

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

    std::map<std::string, ImFont*> m_fonts;

    std::map<Format, ref<RenderPipeline>> m_pipelines;
};

} // namespace sgl::ui

// Extend ImGui with some additional convenience functions.
namespace ImGui {

SGL_API void PushFont(const char* name);

}
