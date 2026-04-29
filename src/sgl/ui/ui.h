// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/ui/fwd.h"

#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/timer.h"

#include "sgl/device/fwd.h"
#include "sgl/device/formats.h"
#include "sgl/math/vector_types.h"

#include <map>
#include <span>

struct ImGuiContext;
struct ImFont;
struct ImTextureData;

namespace sgl::ui {

struct DrawCommand {
    float4 clip_rect;
    uint32_t elem_count;
    uint32_t idx_offset;
    uint32_t vtx_offset;
    ref<Texture> texture;
};

struct DrawList {
    uintptr_t vertex_data;
    uint32_t vertex_count;
    uintptr_t index_data;
    uint32_t index_count;
    std::span<const DrawCommand> commands;
};

struct DrawData {
    float2 display_pos;
    float2 display_size;
    float2 framebuffer_scale;
    std::span<const DrawList> draw_lists;
    uint32_t total_vtx_count;
    uint32_t total_idx_count;
    uint32_t index_size;
};

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
    /// \param window Window this UI context is rendered for (optional).
    void begin_frame(uint32_t width, uint32_t height, sgl::Window* window = nullptr);

    /// End the ImGui frame and renders the UI to the provided texture.
    /// \param texture_view Texture view to render to
    /// \param command_encoder Command encoder to encode commands to
    void end_frame(TextureView* texture_view, CommandEncoder* command_encoder);

    /// End the ImGui frame and renders the UI to the provided texture.
    /// \param texture Texture to render to
    /// \param command_encoder Command encoder to encode commands to
    void end_frame(Texture* texture, CommandEncoder* command_encoder);

    /// Render externally marshaled draw data to the provided texture view.
    /// @param draw_data Externally marshaled Dear ImGui draw data.
    /// @param texture_view Texture view to render to.
    /// @param command_encoder Command encoder used to record the render pass.
    void render_draw_data(const DrawData& draw_data, TextureView* texture_view, CommandEncoder* command_encoder);

    /// Render externally marshaled draw data to the provided texture.
    /// @param draw_data Externally marshaled Dear ImGui draw data.
    /// @param texture Texture to render to.
    /// @param command_encoder Command encoder used to record the render pass.
    void render_draw_data(const DrawData& draw_data, Texture* texture, CommandEncoder* command_encoder);

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

    // TODO: The frame count should not be hard-coded like this.
    // We should probably both control the number of buffers in the Context constructor
    // and pass in the frame to use in begin_frame().
    static constexpr uint32_t FRAME_COUNT = 4;

    ref<Device> m_device;
    ImGuiContext* m_imgui_context;

    ref<Screen> m_screen;

    uint32_t m_frame_index{0};
    Timer m_frame_timer;

    ref<Sampler> m_sampler;
    ref<Buffer> m_vertex_buffers[FRAME_COUNT];
    ref<Buffer> m_index_buffers[FRAME_COUNT];
    ref<ShaderProgram> m_program;
    ref<InputLayout> m_input_layout;

    std::map<std::string, ImFont*> m_fonts;
    std::map<ImTextureData*, ref<Texture>> m_textures;

    std::map<Format, ref<RenderPipeline>> m_pipelines;

    void update_texture(ImTextureData* tex);
    void update_mouse_cursor(sgl::Window* window);
};

} // namespace sgl::ui

// Extend ImGui with some additional convenience functions.
namespace ImGui {

SGL_API void PushFont(const char* name);

}
