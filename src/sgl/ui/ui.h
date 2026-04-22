// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/ui/fwd.h"

#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/timer.h"

#include "sgl/device/fwd.h"
#include "sgl/device/formats.h"

#include <cstdint>
#include <map>
#include <vector>

struct ImGuiContext;
struct ImFont;
struct ImTextureData;
struct ImDrawData;

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

    /// Pass a keyboard event to the UI context.
    /// \param event Keyboard event
    /// \return Returns true if event was consumed.
    bool handle_keyboard_event(const KeyboardEvent& event);

    /// Pass a mouse event to the UI context.
    /// \param event Mouse event
    /// \return Returns true if event was consumed.
    bool handle_mouse_event(const MouseEvent& event);

private:
    // TODO: The frame count should not be hard-coded like this.
    // We should probably both control the number of buffers in the Context constructor
    // and pass in the frame to use in begin_frame().
    static constexpr uint32_t FRAME_COUNT = 4;

    enum class RenderMode {
        disabled,
        rasterizer,
        sw_rasterizer,
    };

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
        TextureView* texture_view,
        CommandEncoder* command_encoder
    );

    void build_sw_draw_data(
        ImDrawData* draw_data,
        TextureView* texture_view,
        ref<Buffer>& triangle_buffer,
        ref<Buffer>& tile_header_buffer,
        ref<Buffer>& tile_index_buffer
    );

    void log_sw_frame_stats() const;

    struct SwTriangle {
        int32_t x0;
        int32_t y0;
        int32_t x1;
        int32_t y1;
        int32_t x2;
        int32_t y2;
        int32_t bbox_min_x;
        int32_t bbox_min_y;
        int32_t bbox_max_x;
        int32_t bbox_max_y;
        float inv_area;
        float u0;
        float v0;
        float u1;
        float v1;
        float u2;
        float v2;
        float c0_r;
        float c0_g;
        float c0_b;
        float c0_a;
        float c1_r;
        float c1_g;
        float c1_b;
        float c1_a;
        float c2_r;
        float c2_g;
        float c2_b;
        float c2_a;
    };

    struct SwTileHeader {
        uint32_t triangle_offset;
        uint32_t triangle_count;
    };

    struct SwDrawCommand {
        ref<Texture> texture;
        int32_t clip_min_x;
        int32_t clip_min_y;
        int32_t clip_max_x;
        int32_t clip_max_y;
        uint32_t tile_header_offset;
        uint32_t tile_grid_width;
        uint32_t tile_grid_height;
        uint32_t triangle_count;
    };

    struct SwFrameStats {
        uint32_t draw_command_count{0};
        uint64_t triangle_count{0};
        uint64_t clip_rect_pixels{0};
        uint64_t dispatched_pixels{0};
        double upload_cpu_ms{0.0};
        double preprocess_cpu_ms{0.0};
        double submit_cpu_ms{0.0};

        void reset()
        {
            *this = {};
        }
    };

    ref<Device> m_device;
    ImGuiContext* m_imgui_context;

    ref<Screen> m_screen;

    uint32_t m_frame_index{0};
    Timer m_frame_timer;

    RenderMode m_render_mode{RenderMode::disabled};

    ref<Sampler> m_sampler;
    ref<Buffer> m_vertex_buffers[FRAME_COUNT];
    ref<Buffer> m_index_buffers[FRAME_COUNT];
    ref<Buffer> m_sw_triangle_buffers[FRAME_COUNT];
    ref<Buffer> m_sw_tile_header_buffers[FRAME_COUNT];
    ref<Buffer> m_sw_tile_index_buffers[FRAME_COUNT];
    ref<ShaderProgram> m_program;
    ref<InputLayout> m_input_layout;
    std::map<Format, ref<RenderPipeline>> m_pipelines;
    ref<ComputePipeline> m_compute_pipeline;

    bool m_log_sw_stats{false};
    SwFrameStats m_sw_frame_stats;
    std::vector<SwDrawCommand> m_sw_draw_commands;
    std::vector<SwTriangle> m_sw_triangles;
    std::vector<SwTileHeader> m_sw_tile_headers;
    std::vector<uint32_t> m_sw_tile_triangle_indices;
    std::vector<uint32_t> m_sw_tile_counts;
    std::vector<uint32_t> m_sw_tile_write_offsets;

    std::map<std::string, ImFont*> m_fonts;
    std::map<ImTextureData*, ref<Texture>> m_textures;

    void update_texture(ImTextureData* tex);
    void update_mouse_cursor(sgl::Window* window);
};

} // namespace sgl::ui

// Extend ImGui with some additional convenience functions.
namespace ImGui {

SGL_API void PushFont(const char* name);

}
