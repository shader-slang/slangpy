// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ui.h"

#include "sgl/ui/widgets.h"

#include "sgl/core/error.h"
#include "sgl/core/input.h"
#include "sgl/core/logger.h"
#include "sgl/core/window.h"
#include "sgl/core/platform.h"

#include "sgl/device/device.h"
#include "sgl/device/sampler.h"
#include "sgl/device/resource.h"
#include "sgl/device/input_layout.h"
#include "sgl/device/shader.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/pipeline.h"

#include <imgui.h>
#include <cmrc/cmrc.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

CMRC_DECLARE(sgl_data);

namespace sgl::ui {

namespace {

constexpr int k_sw_raster_subpixel = 16;
constexpr int k_sw_raster_tile_size = 16;

int clamp_int(int value, int min_value, int max_value)
{
    return std::min(std::max(value, min_value), max_value);
}

int floor_div(int value, int divisor)
{
    SGL_ASSERT(divisor > 0);
    int quotient = value / divisor;
    int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
        --quotient;
    return quotient;
}

int ceil_div(int value, int divisor)
{
    return floor_div(value + divisor - 1, divisor);
}

float unpack_color_channel(uint32_t color, uint32_t shift)
{
    return float((color >> shift) & 0xffu) / 255.f;
}

bool env_flag_enabled(const char* name)
{
    auto value = platform::get_environment_variable(name);
    if (!value || value->empty())
        return false;
    return *value != "0" && *value != "false" && *value != "FALSE";
}

template<typename T>
void ensure_structured_buffer_capacity(ref<Device> device, ref<Buffer>& buffer, size_t element_count, const char* label)
{
    size_t capacity = std::max<size_t>(element_count, 1);
    capacity += std::max<size_t>(capacity / 2, 64);
    size_t required_size = capacity * sizeof(T);
    if (buffer && buffer->size() >= required_size)
        return;

    buffer = device->create_buffer({
        .size = required_size,
        .memory_type = MemoryType::upload,
        .usage = BufferUsage::shader_resource,
        .label = label,
    });
}

} // namespace

static void setup_style()
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

#if 1
    // Modified dark theme.

    ImGui::StyleColorsDark();

    colors[ImGuiCol_WindowBg] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
    colors[ImGuiCol_TabDimmed] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
#else
    // Custom theme.
    // Disabled for now since its tedious to maintain.

    colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
    colors[ImGuiCol_Border] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.20f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.09f, 0.12f, 0.14f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.09f, 0.12f, 0.14f, 0.65f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.39f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.18f, 0.22f, 0.25f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.09f, 0.21f, 0.31f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.37f, 0.61f, 1.00f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.29f, 0.55f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    // ImGuiCol_InputTextCursor
    colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    colors[ImGuiCol_Tab] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    // ImGuiCol_TabSelectedOverline
    colors[ImGuiCol_TabDimmed] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TabDimmedSelected] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    // ImGuiCol_TabDimmedSelectedOverline
    // ImGuiCol_DockingPreview
    // ImGuiCol_DockingEmptyBg
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    // ImGuiCol_TableHeaderBg
    // ImGuiCol_TableBorderStrong
    // ImGuiCol_TableBorderLight
    // ImGuiCol_TableRowBg
    // ImGuiCol_TableRowBgAlt
    // ImGuiCol_TextLink
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    // ImGuiCol_TreeLines
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    // ImGuiCol_DragDropTargetBg
    // ImGuiCol_UnsavedMarker
    colors[ImGuiCol_NavCursor] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

    style.WindowPadding = ImVec2(8.00f, 8.00f);
    style.FramePadding = ImVec2(5.00f, 2.00f);
    style.CellPadding = ImVec2(6.00f, 6.00f);
    style.ItemSpacing = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding = ImVec2(0.00f, 0.00f);
    style.IndentSpacing = 25;
    style.ScrollbarSize = 15;
    style.GrabMinSize = 10;
    style.WindowBorderSize = 1;
    style.ChildBorderSize = 1;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = 1;
    style.TabBorderSize = 1;
    style.WindowRounding = 7;
    style.ChildRounding = 4;
    style.FrameRounding = 3;
    style.PopupRounding = 4;
    style.ScrollbarRounding = 9;
    style.GrabRounding = 3;
    style.LogSliderDeadzone = 4;
    style.TabRounding = 4;
#endif
}

static ImGuiKey key_code_to_imgui_key(KeyCode key)
{
    static const std::unordered_map<KeyCode, ImGuiKey> map{
        {KeyCode::space, ImGuiKey_Space},
        {KeyCode::apostrophe, ImGuiKey_Apostrophe},
        {KeyCode::comma, ImGuiKey_Comma},
        {KeyCode::minus, ImGuiKey_Minus},
        {KeyCode::period, ImGuiKey_Period},
        {KeyCode::slash, ImGuiKey_Slash},
        {KeyCode::key0, ImGuiKey_0},
        {KeyCode::key1, ImGuiKey_1},
        {KeyCode::key2, ImGuiKey_2},
        {KeyCode::key3, ImGuiKey_3},
        {KeyCode::key4, ImGuiKey_4},
        {KeyCode::key5, ImGuiKey_5},
        {KeyCode::key6, ImGuiKey_6},
        {KeyCode::key7, ImGuiKey_7},
        {KeyCode::key8, ImGuiKey_8},
        {KeyCode::key9, ImGuiKey_9},
        {KeyCode::semicolon, ImGuiKey_Semicolon},
        {KeyCode::equal, ImGuiKey_Equal},
        {KeyCode::a, ImGuiKey_A},
        {KeyCode::b, ImGuiKey_B},
        {KeyCode::c, ImGuiKey_C},
        {KeyCode::d, ImGuiKey_D},
        {KeyCode::e, ImGuiKey_E},
        {KeyCode::f, ImGuiKey_F},
        {KeyCode::g, ImGuiKey_G},
        {KeyCode::h, ImGuiKey_H},
        {KeyCode::i, ImGuiKey_I},
        {KeyCode::j, ImGuiKey_J},
        {KeyCode::k, ImGuiKey_K},
        {KeyCode::l, ImGuiKey_L},
        {KeyCode::m, ImGuiKey_M},
        {KeyCode::n, ImGuiKey_N},
        {KeyCode::o, ImGuiKey_O},
        {KeyCode::p, ImGuiKey_P},
        {KeyCode::q, ImGuiKey_Q},
        {KeyCode::r, ImGuiKey_R},
        {KeyCode::s, ImGuiKey_S},
        {KeyCode::t, ImGuiKey_T},
        {KeyCode::u, ImGuiKey_U},
        {KeyCode::v, ImGuiKey_V},
        {KeyCode::w, ImGuiKey_W},
        {KeyCode::x, ImGuiKey_X},
        {KeyCode::y, ImGuiKey_Y},
        {KeyCode::z, ImGuiKey_Z},
        {KeyCode::left_bracket, ImGuiKey_LeftBracket},
        {KeyCode::backslash, ImGuiKey_Backslash},
        {KeyCode::right_bracket, ImGuiKey_RightBracket},
        {KeyCode::grave_accent, ImGuiKey_GraveAccent},
        {KeyCode::escape, ImGuiKey_Escape},
        {KeyCode::tab, ImGuiKey_Tab},
        {KeyCode::enter, ImGuiKey_Enter},
        {KeyCode::backspace, ImGuiKey_Backspace},
        {KeyCode::insert, ImGuiKey_Insert},
        {KeyCode::delete_, ImGuiKey_Delete},
        {KeyCode::left, ImGuiKey_LeftArrow},
        {KeyCode::right, ImGuiKey_RightArrow},
        {KeyCode::up, ImGuiKey_UpArrow},
        {KeyCode::down, ImGuiKey_DownArrow},
        {KeyCode::page_up, ImGuiKey_PageUp},
        {KeyCode::page_down, ImGuiKey_PageDown},
        {KeyCode::home, ImGuiKey_Home},
        {KeyCode::end, ImGuiKey_End},
        {KeyCode::caps_lock, ImGuiKey_CapsLock},
        {KeyCode::scroll_lock, ImGuiKey_ScrollLock},
        {KeyCode::num_lock, ImGuiKey_NumLock},
        {KeyCode::print_screen, ImGuiKey_PrintScreen},
        {KeyCode::pause, ImGuiKey_Pause},
        {KeyCode::f1, ImGuiKey_F1},
        {KeyCode::f2, ImGuiKey_F2},
        {KeyCode::f3, ImGuiKey_F3},
        {KeyCode::f4, ImGuiKey_F4},
        {KeyCode::f5, ImGuiKey_F5},
        {KeyCode::f6, ImGuiKey_F6},
        {KeyCode::f7, ImGuiKey_F7},
        {KeyCode::f8, ImGuiKey_F8},
        {KeyCode::f9, ImGuiKey_F9},
        {KeyCode::f10, ImGuiKey_F10},
        {KeyCode::f11, ImGuiKey_F11},
        {KeyCode::f12, ImGuiKey_F12},
        {KeyCode::keypad0, ImGuiKey_Keypad0},
        {KeyCode::keypad1, ImGuiKey_Keypad1},
        {KeyCode::keypad2, ImGuiKey_Keypad2},
        {KeyCode::keypad3, ImGuiKey_Keypad3},
        {KeyCode::keypad4, ImGuiKey_Keypad4},
        {KeyCode::keypad5, ImGuiKey_Keypad5},
        {KeyCode::keypad6, ImGuiKey_Keypad6},
        {KeyCode::keypad7, ImGuiKey_Keypad7},
        {KeyCode::keypad8, ImGuiKey_Keypad8},
        {KeyCode::keypad9, ImGuiKey_Keypad9},
        {KeyCode::keypad_divide, ImGuiKey_KeypadDivide},
        {KeyCode::keypad_multiply, ImGuiKey_KeypadMultiply},
        {KeyCode::keypad_subtract, ImGuiKey_KeypadSubtract},
        {KeyCode::keypad_add, ImGuiKey_KeypadAdd},
        {KeyCode::keypad_enter, ImGuiKey_KeypadEnter},
        {KeyCode::left_shift, ImGuiKey_LeftShift},
        {KeyCode::left_control, ImGuiKey_LeftCtrl},
        {KeyCode::left_alt, ImGuiKey_LeftAlt},
        {KeyCode::left_super, ImGuiKey_LeftSuper},
        {KeyCode::right_shift, ImGuiKey_RightShift},
        {KeyCode::right_control, ImGuiKey_RightCtrl},
        {KeyCode::right_alt, ImGuiKey_RightAlt},
        {KeyCode::right_super, ImGuiKey_RightSuper},
        {KeyCode::menu, ImGuiKey_Menu},
    };

    auto it = map.find(key);
    return it == map.end() ? ImGuiKey_None : it->second;
}

static std::string utf32_to_utf8(uint32_t utf32)
{
    std::string utf8;
    if (utf32 <= 0x7F) {
        utf8.resize(1);
        utf8[0] = static_cast<char>(utf32);
    } else if (utf32 <= 0x7FF) {
        utf8.resize(2);
        utf8[0] = static_cast<char>(0xC0 | (utf32 >> 6));
        utf8[1] = static_cast<char>(0x80 | (utf32 & 0x3F));
    } else if (utf32 <= 0xFFFF) {
        utf8.resize(3);
        utf8[0] = static_cast<char>(0xE0 | (utf32 >> 12));
        utf8[1] = static_cast<char>(0x80 | ((utf32 >> 6) & 0x3F));
        utf8[2] = static_cast<char>(0x80 | (utf32 & 0x3F));
    } else if (utf32 <= 0x10FFFF) {
        utf8.resize(4);
        utf8[0] = static_cast<char>(0xF0 | (utf32 >> 18));
        utf8[1] = static_cast<char>(0x80 | ((utf32 >> 12) & 0x3F));
        utf8[2] = static_cast<char>(0x80 | ((utf32 >> 6) & 0x3F));
        utf8[3] = static_cast<char>(0x80 | (utf32 & 0x3F));
    } else {
        SGL_THROW("Invalid UTF32 character");
    }
    return utf8;
}


Context::Context(ref<Device> device)
    : m_device(std::move(device))
{
    m_imgui_context = ImGui::CreateContext();
    ImGui::SetCurrentContext(m_imgui_context);

    m_screen = ref<Screen>(new Screen());

    ImGuiIO& io = ImGui::GetIO();
    io.UserData = this;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.IniFilename = nullptr;
    io.ConfigNavCaptureKeyboard = false;

    float scale_factor = platform::display_scale_factor();

    // Load an embedded font.
    auto load_embedded_font = [&](const char* name, const char* path)
    {
        ImFontConfig font_config;
        font_config.FontDataOwnedByAtlas = false;
        auto font_file = cmrc::sgl_data::get_filesystem().open(path);
        ImFont* font = io.Fonts->AddFontFromMemoryTTF(
            (void*)font_file.begin(),
            (int)font_file.size(),
            15.f * scale_factor,
            &font_config
        );
        m_fonts[name] = font;
    };

    // Setup fonts.
    load_embedded_font("default", "data/fonts/Montserrat-Regular.ttf");
    load_embedded_font("monospace", "data/fonts/Inconsolata-Regular.ttf");

    // Setup style.
    setup_style();
    ImGui::GetStyle().ScaleAllSizes(scale_factor);

    // Determine render mode.
    m_render_mode = m_device->has_feature(Feature::rasterization) ? RenderMode::rasterizer : RenderMode::sw_rasterizer;
    m_log_sw_stats = env_flag_enabled("SGL_IMGUI_SW_STATS");

    // Setup sampler.
    m_sampler = m_device->create_sampler({
        .min_filter = TextureFilteringMode::linear,
        .mag_filter = TextureFilteringMode::linear,
        .mip_filter = TextureFilteringMode::linear,
        .address_u = TextureAddressingMode::wrap,
        .address_v = TextureAddressingMode::wrap,
        .address_w = TextureAddressingMode::wrap,
        .mip_lod_bias = 0.f,
        .max_anisotropy = 1,
        .border_color = {0.f, 0.f, 0.f, 0.f},
        .min_lod = 0.f,
        .max_lod = 0.f,
    });

    switch (m_render_mode) {
    case RenderMode::disabled:
        break;
    case RenderMode::rasterizer:
        init_rasterizer();
        break;
    case RenderMode::sw_rasterizer:
        init_sw_rasterizer();
        break;
    }
}

Context::~Context()
{
    ImGui::SetCurrentContext(m_imgui_context);
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
        if (tex->Status != ImTextureStatus_Destroyed) {
            tex->SetTexID(ImTextureID_Invalid);
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    }
    m_textures.clear();
    ImGui::DestroyContext(m_imgui_context);
}

ImFont* Context::get_font(const char* name)
{
    auto it = m_fonts.find(name);
    return it == m_fonts.end() ? nullptr : it->second;
}

void Context::begin_frame(uint32_t width, uint32_t height, sgl::Window* window)
{
    ImGui::SetCurrentContext(m_imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
    io.DeltaTime = static_cast<float>(m_frame_timer.elapsed_s());
    m_frame_timer.reset();

    if (window)
        update_mouse_cursor(window);

    ImGui::NewFrame();

    m_screen->render();
}

void Context::end_frame(TextureView* texture_view, CommandEncoder* command_encoder)
{
    ImGui::SetCurrentContext(m_imgui_context);

    ImGui::Render();

    if (m_render_mode == RenderMode::disabled)
        return;

    ImDrawData* draw_data = ImGui::GetDrawData();

    // Update textures.
    if (draw_data->Textures != nullptr)
        for (ImTextureData* tex : *draw_data->Textures)
            if (tex->Status != ImTextureStatus_OK)
                update_texture(tex);

    if (draw_data->CmdListsCount > 0) {
        // Cycle through per-frame buffers.
        const uint32_t frame_index = m_frame_index;
        m_frame_index = (m_frame_index + 1) % FRAME_COUNT;

        ref<Buffer>& vertex_buffer = m_vertex_buffers[frame_index];
        ref<Buffer>& index_buffer = m_index_buffers[frame_index];
        ref<Buffer>& sw_triangle_buffer = m_sw_triangle_buffers[frame_index];
        ref<Buffer>& sw_tile_header_buffer = m_sw_tile_header_buffers[frame_index];
        ref<Buffer>& sw_tile_index_buffer = m_sw_tile_index_buffers[frame_index];

        m_sw_frame_stats.reset();

        switch (m_render_mode) {
        case RenderMode::disabled:
            break;
        case RenderMode::rasterizer:
        {
            Timer upload_timer;

            if (!vertex_buffer || vertex_buffer->size() < draw_data->TotalVtxCount * sizeof(ImDrawVert)) {
                vertex_buffer = m_device->create_buffer({
                    .size = draw_data->TotalVtxCount * sizeof(ImDrawVert) + 128 * 1024,
                    .memory_type = MemoryType::upload,
                    .usage = BufferUsage::vertex_buffer,
                    .label = "imgui vertex buffer",
                });
            }

            if (!index_buffer || index_buffer->size() < draw_data->TotalIdxCount * sizeof(ImDrawIdx)) {
                index_buffer = m_device->create_buffer({
                    .size = draw_data->TotalIdxCount * sizeof(ImDrawIdx) + 1024,
                    .memory_type = MemoryType::upload,
                    .usage = BufferUsage::index_buffer,
                    .label = "imgui index buffer",
                });
            }

            ImDrawVert* vertices = vertex_buffer->map<ImDrawVert>();
            ImDrawIdx* indices = index_buffer->map<ImDrawIdx>();
            for (int i = 0; i < draw_data->CmdListsCount; ++i) {
                const ImDrawList* cmd_list = draw_data->CmdLists[i];
                std::memcpy(vertices, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
                std::memcpy(indices, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
                vertices += cmd_list->VtxBuffer.Size;
                indices += cmd_list->IdxBuffer.Size;
            }
            vertex_buffer->unmap();
            index_buffer->unmap();
            m_sw_frame_stats.upload_cpu_ms = upload_timer.elapsed_ms();

            draw(draw_data, vertex_buffer, index_buffer, texture_view, command_encoder);
            break;
        }
        case RenderMode::sw_rasterizer:
        {
            Timer preprocess_timer;
            build_sw_draw_data(draw_data, texture_view, sw_triangle_buffer, sw_tile_header_buffer, sw_tile_index_buffer);
            m_sw_frame_stats.preprocess_cpu_ms = preprocess_timer.elapsed_ms();

            Timer submit_timer;
            draw_sw(texture_view, command_encoder);
            m_sw_frame_stats.submit_cpu_ms = submit_timer.elapsed_ms();

            if (m_log_sw_stats)
                log_sw_frame_stats();
            break;
        }
        }
    }
}

void Context::end_frame(Texture* texture, CommandEncoder* command_encoder)
{
    // TODO(slang-rhi) use default_view once it is available
    end_frame(texture->create_view({}), command_encoder);
}

bool Context::handle_keyboard_event(const KeyboardEvent& event)
{
    ImGui::SetCurrentContext(m_imgui_context);
    ImGuiIO& io = ImGui::GetIO();

    io.AddKeyEvent(ImGuiMod_Shift, event.has_modifier(KeyModifier::shift));
    io.AddKeyEvent(ImGuiMod_Ctrl, event.has_modifier(KeyModifier::ctrl));
    io.AddKeyEvent(ImGuiMod_Alt, event.has_modifier(KeyModifier::alt));

    switch (event.type) {
    case KeyboardEventType::key_press:
    case KeyboardEventType::key_release:
        io.AddKeyEvent(key_code_to_imgui_key(event.key), event.type == KeyboardEventType::key_press);
        break;
    case KeyboardEventType::key_repeat:
        break;
    case KeyboardEventType::input:
        io.AddInputCharactersUTF8(utf32_to_utf8(event.codepoint).c_str());
        break;
    }

    return io.WantCaptureKeyboard;
}

bool Context::handle_mouse_event(const MouseEvent& event)
{
    ImGui::SetCurrentContext(m_imgui_context);
    ImGuiIO& io = ImGui::GetIO();

    io.AddKeyEvent(ImGuiMod_Shift, event.has_modifier(KeyModifier::shift));
    io.AddKeyEvent(ImGuiMod_Ctrl, event.has_modifier(KeyModifier::ctrl));
    io.AddKeyEvent(ImGuiMod_Alt, event.has_modifier(KeyModifier::alt));

    switch (event.type) {
    case MouseEventType::button_down:
    case MouseEventType::button_up:
        io.AddMouseButtonEvent(static_cast<int>(event.button), event.type == MouseEventType::button_down);
        break;
    case MouseEventType::move:
        io.AddMousePosEvent(event.pos.x, event.pos.y);
        break;
    case MouseEventType::scroll:
        io.AddMouseWheelEvent(event.scroll.x, event.scroll.y);
        break;
    }

    return io.WantCaptureMouse;
}

void Context::init_rasterizer()
{
    // Setup program.
    m_program = m_device->load_program("sgl/ui/imgui.slang", {"vs_main", "fs_main"});

    // Setup vertex layout.
    m_input_layout = m_device->create_input_layout({
        .input_elements{
            {.semantic_name = "POSITION", .format = Format::rg32_float, .offset = offsetof(ImDrawVert, pos)},
            {.semantic_name = "TEXCOORD", .format = Format::rg32_float, .offset = offsetof(ImDrawVert, uv)},
            {.semantic_name = "COLOR", .format = Format::rgba8_unorm, .offset = offsetof(ImDrawVert, col)},
        },
        .vertex_streams{
            {.stride = sizeof(ImDrawVert)},
        },
    });
}

void Context::init_sw_rasterizer()
{
    m_program = m_device->load_program("sgl/ui/imguisw.slang", {"cs_main"});

    m_compute_pipeline = m_device->create_compute_pipeline({
        .program = m_program,
    });
}

RenderPipeline* Context::get_pipeline(Format format)
{
    auto it = m_pipelines.find(format);
    if (it != m_pipelines.end())
        return it->second;

    // Create pipeline.
    ref<RenderPipeline> pipeline = m_device->create_render_pipeline({
        .program = m_program,
        .input_layout = m_input_layout,
        .primitive_topology = PrimitiveTopology::triangle_list,
        .targets = {
            {
                .format = format,
                .color = {
                    .src_factor = BlendFactor::src_alpha,
                    .dst_factor = BlendFactor::inv_src_alpha,
                    .op = BlendOp::add,
                },
                .alpha = {
                    .src_factor = BlendFactor::one,
                    .dst_factor = BlendFactor::inv_src_alpha,
                    .op = BlendOp::add,
                },
                .enable_blend = true,
            },
        },
    });

    m_pipelines.emplace(format, pipeline);
    return pipeline;
}

void Context::update_texture(ImTextureData* tex)
{
    if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates) {
        SGL_ASSERT(tex->Format == ImTextureFormat_RGBA32);
        SubresourceData data[1] = {{
            .data = tex->GetPixels(),
            .size = size_t(tex->GetSizeInBytes()),
            .row_pitch = size_t(tex->GetPitch()),
            .slice_pitch = size_t(tex->GetSizeInBytes()),
        }};
        ref<Texture> gpu_texture = m_device->create_texture({
            .format = Format::rgba8_unorm,
            .width = narrow_cast<uint32_t>(tex->Width),
            .height = narrow_cast<uint32_t>(tex->Height),
            .usage = TextureUsage::shader_resource,
            .data = data,
        });
        m_textures[tex] = gpu_texture;
        tex->SetTexID(gpu_texture);
        tex->SetStatus(ImTextureStatus_OK);
    }
    if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0) {
        m_textures.erase(tex);
        tex->SetTexID(ImTextureID_Invalid);
        tex->SetStatus(ImTextureStatus_Destroyed);
    }
}

void Context::update_mouse_cursor(sgl::Window* window)
{
    ImGuiIO& io = ImGui::GetIO();
    ImGuiMouseCursor imgui_cursor = ImGui::GetMouseCursor();

    if (imgui_cursor == ImGuiMouseCursor_None) {
        window->set_cursor_mode(CursorMode::hidden);
    } else {
        window->set_cursor_mode(CursorMode::normal);
        CursorShape shape = CursorShape::arrow;
        switch (ImGui::GetMouseCursor()) {
        case ImGuiMouseCursor_Arrow:
            shape = CursorShape::arrow;
            break;
        case ImGuiMouseCursor_TextInput:
            shape = CursorShape::ibeam;
            break;
        case ImGuiMouseCursor_ResizeNS:
            shape = CursorShape::vresize;
            break;
        case ImGuiMouseCursor_ResizeEW:
            shape = CursorShape::hresize;
            break;
        case ImGuiMouseCursor_Hand:
            shape = CursorShape::hand;
            break;
        default:
            shape = CursorShape::arrow;
            break;
        }
        window->set_cursor_shape(shape);
    }
}

void Context::draw(
    ImDrawData* draw_data,
    Buffer* vertex_buffer,
    Buffer* index_buffer,
    TextureView* texture_view,
    CommandEncoder* command_encoder
)
{
    ImGuiIO& io = ImGui::GetIO();

    bool is_srgb_format = get_format_info(texture_view->format()).is_srgb_format();

    // Render command lists.
    auto pass_encoder = command_encoder->begin_render_pass({
        .color_attachments = {
            {
                .view = texture_view,
                .load_op = LoadOp::load,
                .store_op = StoreOp::store,
            },
        },
    });
    ShaderObject* shader_object = pass_encoder->bind_pipeline(get_pipeline(texture_view->desc().format));
    ShaderCursor shader_cursor = ShaderCursor(shader_object);
    shader_cursor["sampler"] = m_sampler;
    shader_cursor["scale"] = 2.f / float2(io.DisplaySize.x, -io.DisplaySize.y);
    shader_cursor["offset"] = float2(-1.f, 1.f);
    shader_cursor["is_srgb_format"] = is_srgb_format;
    ShaderOffset texture_offset = shader_cursor["texture"].offset();

    RenderState render_state = {
        .viewports = {Viewport::from_size(io.DisplaySize.x, io.DisplaySize.y)},
        .scissor_rects = {ScissorRect{}},
        .vertex_buffers = {vertex_buffer},
        .index_buffer = index_buffer,
        .index_format = sizeof(ImDrawIdx) == 2 ? IndexFormat::uint16 : IndexFormat::uint32,
    };

    int vertex_offset = 0;
    int index_offset = 0;
    ImVec2 clip_off = draw_data->DisplayPos;
    for (int n = 0; n < draw_data->CmdListsCount; n++) {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
            SGL_ASSERT(pcmd->UserCallback == nullptr);
            // Project scissor/clipping rectangles into framebuffer space.
            ScissorRect clip_rect{
                .min_x = uint32_t(pcmd->ClipRect.x - clip_off.x),
                .min_y = uint32_t(pcmd->ClipRect.y - clip_off.y),
                .max_x = uint32_t(pcmd->ClipRect.z - clip_off.x),
                .max_y = uint32_t(pcmd->ClipRect.w - clip_off.y),
            };
            if (clip_rect.max_x <= clip_rect.min_x || clip_rect.max_y <= clip_rect.min_y)
                continue;

            // Apply scissor/clipping rectangle, bind texture, draw.
            render_state.scissor_rects[0] = clip_rect;
            ref<Texture> texture = ref<Texture>(static_cast<Texture*>(pcmd->GetTexID()));
            shader_object->set_texture(texture_offset, texture);
            pass_encoder->set_render_state(render_state);
            pass_encoder->draw_indexed({
                .vertex_count = pcmd->ElemCount,
                .start_vertex_location = pcmd->VtxOffset + vertex_offset,
                .start_index_location = pcmd->IdxOffset + index_offset,
            });
        }
        index_offset += cmd_list->IdxBuffer.Size;
        vertex_offset += cmd_list->VtxBuffer.Size;
    }
    pass_encoder->end();
}

void Context::draw_sw(
    TextureView* texture_view,
    CommandEncoder* command_encoder
)
{
    bool is_srgb_format = get_format_info(texture_view->format()).is_srgb_format();

    if (m_sw_draw_commands.empty())
        return;

    auto pass_encoder = command_encoder->begin_compute_pass();
    ShaderObject* shader_object = pass_encoder->bind_pipeline(m_compute_pipeline);
    ShaderCursor shader_cursor = ShaderCursor(shader_object);
    shader_cursor["output_texture"] = ref(texture_view);
    shader_cursor["sampler"] = m_sampler;
    shader_cursor["triangles"] = m_sw_triangle_buffers[(m_frame_index + FRAME_COUNT - 1) % FRAME_COUNT];
    shader_cursor["tile_headers"] = m_sw_tile_header_buffers[(m_frame_index + FRAME_COUNT - 1) % FRAME_COUNT];
    shader_cursor["tile_triangle_indices"] = m_sw_tile_index_buffers[(m_frame_index + FRAME_COUNT - 1) % FRAME_COUNT];
    shader_cursor["is_srgb_format"] = is_srgb_format;
    ShaderOffset texture_offset = shader_cursor["texture"].offset();
    ShaderCursor entry_point_cursor = ShaderCursor(shader_object->get_entry_point(0));

    for (const SwDrawCommand& draw_command : m_sw_draw_commands) {
        shader_object->set_texture(texture_offset, draw_command.texture);
        entry_point_cursor["dispatch_origin"] = int2(draw_command.clip_min_x, draw_command.clip_min_y);
        entry_point_cursor["tile_header_offset"] = draw_command.tile_header_offset;
        entry_point_cursor["tile_grid_width"] = draw_command.tile_grid_width;
        pass_encoder->dispatch(
            uint3(
                uint32_t(draw_command.clip_max_x - draw_command.clip_min_x),
                uint32_t(draw_command.clip_max_y - draw_command.clip_min_y),
                1
            )
        );
    }

    pass_encoder->end();
}

void Context::build_sw_draw_data(
    ImDrawData* draw_data,
    TextureView* texture_view,
    ref<Buffer>& triangle_buffer,
    ref<Buffer>& tile_header_buffer,
    ref<Buffer>& tile_index_buffer
)
{
    m_sw_draw_commands.clear();
    m_sw_triangles.clear();
    m_sw_tile_headers.clear();
    m_sw_tile_triangle_indices.clear();

    uint32_t target_width = texture_view->texture()->width();
    uint32_t target_height = texture_view->texture()->height();
    ImVec2 clip_off = draw_data->DisplayPos;

    for (int list_index = 0; list_index < draw_data->CmdListsCount; ++list_index) {
        const ImDrawList* cmd_list = draw_data->CmdLists[list_index];
        for (int cmd_index = 0; cmd_index < cmd_list->CmdBuffer.Size; ++cmd_index) {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_index];
            SGL_ASSERT(pcmd->UserCallback == nullptr);

            int clip_min_x = clamp_int(int(std::floor(pcmd->ClipRect.x - clip_off.x)), 0, int(target_width));
            int clip_min_y = clamp_int(int(std::floor(pcmd->ClipRect.y - clip_off.y)), 0, int(target_height));
            int clip_max_x = clamp_int(int(std::ceil(pcmd->ClipRect.z - clip_off.x)), 0, int(target_width));
            int clip_max_y = clamp_int(int(std::ceil(pcmd->ClipRect.w - clip_off.y)), 0, int(target_height));
            if (clip_max_x <= clip_min_x || clip_max_y <= clip_min_y)
                continue;

            const uint32_t raw_triangle_count = pcmd->ElemCount / 3;
            const uint64_t clip_rect_pixels = uint64_t(clip_max_x - clip_min_x) * uint64_t(clip_max_y - clip_min_y);

            const uint32_t tile_grid_width = uint32_t((clip_max_x - clip_min_x + k_sw_raster_tile_size - 1) / k_sw_raster_tile_size);
            const uint32_t tile_grid_height = uint32_t((clip_max_y - clip_min_y + k_sw_raster_tile_size - 1) / k_sw_raster_tile_size);
            const size_t tile_count = size_t(tile_grid_width) * size_t(tile_grid_height);

            const size_t triangle_base = m_sw_triangles.size();
            m_sw_tile_counts.assign(tile_count, 0);

            for (uint32_t triangle_index = 0; triangle_index < raw_triangle_count; ++triangle_index) {
                const ImDrawIdx i0 = cmd_list->IdxBuffer[pcmd->IdxOffset + triangle_index * 3 + 0];
                const ImDrawIdx i1 = cmd_list->IdxBuffer[pcmd->IdxOffset + triangle_index * 3 + 1];
                const ImDrawIdx i2 = cmd_list->IdxBuffer[pcmd->IdxOffset + triangle_index * 3 + 2];

                const ImDrawVert& v0 = cmd_list->VtxBuffer[pcmd->VtxOffset + i0];
                const ImDrawVert& v1 = cmd_list->VtxBuffer[pcmd->VtxOffset + i1];
                const ImDrawVert& v2 = cmd_list->VtxBuffer[pcmd->VtxOffset + i2];

                const int x0 = int(v0.pos.x * float(k_sw_raster_subpixel));
                const int y0 = int(v0.pos.y * float(k_sw_raster_subpixel));
                const int x1 = int(v1.pos.x * float(k_sw_raster_subpixel));
                const int y1 = int(v1.pos.y * float(k_sw_raster_subpixel));
                const int x2 = int(v2.pos.x * float(k_sw_raster_subpixel));
                const int y2 = int(v2.pos.y * float(k_sw_raster_subpixel));

                const int area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                if (area == 0)
                    continue;

                int bbox_min_x = floor_div(std::min({x0, x1, x2}), k_sw_raster_subpixel);
                int bbox_min_y = floor_div(std::min({y0, y1, y2}), k_sw_raster_subpixel);
                int bbox_max_x = ceil_div(std::max({x0, x1, x2}), k_sw_raster_subpixel);
                int bbox_max_y = ceil_div(std::max({y0, y1, y2}), k_sw_raster_subpixel);

                bbox_min_x = clamp_int(bbox_min_x, clip_min_x, clip_max_x);
                bbox_min_y = clamp_int(bbox_min_y, clip_min_y, clip_max_y);
                bbox_max_x = clamp_int(bbox_max_x, clip_min_x, clip_max_x);
                bbox_max_y = clamp_int(bbox_max_y, clip_min_y, clip_max_y);
                if (bbox_max_x <= bbox_min_x || bbox_max_y <= bbox_min_y)
                    continue;

                SwTriangle triangle{
                    .x0 = x0,
                    .y0 = y0,
                    .x1 = x1,
                    .y1 = y1,
                    .x2 = x2,
                    .y2 = y2,
                    .bbox_min_x = bbox_min_x,
                    .bbox_min_y = bbox_min_y,
                    .bbox_max_x = bbox_max_x,
                    .bbox_max_y = bbox_max_y,
                    .inv_area = 1.f / float(area),
                    .u0 = v0.uv.x,
                    .v0 = v0.uv.y,
                    .u1 = v1.uv.x,
                    .v1 = v1.uv.y,
                    .u2 = v2.uv.x,
                    .v2 = v2.uv.y,
                    .c0_r = unpack_color_channel(v0.col, 0),
                    .c0_g = unpack_color_channel(v0.col, 8),
                    .c0_b = unpack_color_channel(v0.col, 16),
                    .c0_a = unpack_color_channel(v0.col, 24),
                    .c1_r = unpack_color_channel(v1.col, 0),
                    .c1_g = unpack_color_channel(v1.col, 8),
                    .c1_b = unpack_color_channel(v1.col, 16),
                    .c1_a = unpack_color_channel(v1.col, 24),
                    .c2_r = unpack_color_channel(v2.col, 0),
                    .c2_g = unpack_color_channel(v2.col, 8),
                    .c2_b = unpack_color_channel(v2.col, 16),
                    .c2_a = unpack_color_channel(v2.col, 24),
                };
                m_sw_triangles.push_back(triangle);

                const int tile_min_x = (bbox_min_x - clip_min_x) / k_sw_raster_tile_size;
                const int tile_min_y = (bbox_min_y - clip_min_y) / k_sw_raster_tile_size;
                const int tile_max_x = (bbox_max_x - clip_min_x - 1) / k_sw_raster_tile_size;
                const int tile_max_y = (bbox_max_y - clip_min_y - 1) / k_sw_raster_tile_size;
                for (int tile_y = tile_min_y; tile_y <= tile_max_y; ++tile_y) {
                    for (int tile_x = tile_min_x; tile_x <= tile_max_x; ++tile_x) {
                        const size_t tile_offset = size_t(tile_y) * tile_grid_width + size_t(tile_x);
                        ++m_sw_tile_counts[tile_offset];
                    }
                }
            }

            const uint32_t emitted_triangle_count = narrow_cast<uint32_t>(m_sw_triangles.size() - triangle_base);
            if (emitted_triangle_count == 0)
                continue;

            const size_t tile_header_offset = m_sw_tile_headers.size();
            m_sw_tile_headers.resize(tile_header_offset + tile_count);

            size_t tile_index_base = m_sw_tile_triangle_indices.size();
            for (size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
                m_sw_tile_headers[tile_header_offset + tile_index] = SwTileHeader{
                    .triangle_offset = narrow_cast<uint32_t>(tile_index_base),
                    .triangle_count = m_sw_tile_counts[tile_index],
                };
                tile_index_base += m_sw_tile_counts[tile_index];
            }

            const size_t first_tile_index = m_sw_tile_triangle_indices.size();
            m_sw_tile_triangle_indices.resize(tile_index_base);
            m_sw_tile_write_offsets.resize(tile_count);
            for (size_t tile_index = 0; tile_index < tile_count; ++tile_index)
                m_sw_tile_write_offsets[tile_index] = m_sw_tile_headers[tile_header_offset + tile_index].triangle_offset;

            for (uint32_t local_triangle_index = 0; local_triangle_index < emitted_triangle_count; ++local_triangle_index) {
                const SwTriangle& triangle = m_sw_triangles[triangle_base + local_triangle_index];
                const int tile_min_x = (triangle.bbox_min_x - clip_min_x) / k_sw_raster_tile_size;
                const int tile_min_y = (triangle.bbox_min_y - clip_min_y) / k_sw_raster_tile_size;
                const int tile_max_x = (triangle.bbox_max_x - clip_min_x - 1) / k_sw_raster_tile_size;
                const int tile_max_y = (triangle.bbox_max_y - clip_min_y - 1) / k_sw_raster_tile_size;
                for (int tile_y = tile_min_y; tile_y <= tile_max_y; ++tile_y) {
                    for (int tile_x = tile_min_x; tile_x <= tile_max_x; ++tile_x) {
                        const size_t tile_offset = size_t(tile_y) * tile_grid_width + size_t(tile_x);
                        const uint32_t write_offset = m_sw_tile_write_offsets[tile_offset]++;
                        m_sw_tile_triangle_indices[write_offset] = narrow_cast<uint32_t>(triangle_base + local_triangle_index);
                    }
                }
            }

            for (size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
                const SwTileHeader& tile_header = m_sw_tile_headers[tile_header_offset + tile_index];
                const uint32_t consumed = m_sw_tile_write_offsets[tile_index] - tile_header.triangle_offset;
                SGL_ASSERT(consumed == tile_header.triangle_count);
            }
            SGL_ASSERT(m_sw_tile_triangle_indices.size() - first_tile_index == tile_index_base - first_tile_index);

            m_sw_draw_commands.push_back(SwDrawCommand{
                .texture = ref<Texture>(static_cast<Texture*>(pcmd->GetTexID())),
                .clip_min_x = clip_min_x,
                .clip_min_y = clip_min_y,
                .clip_max_x = clip_max_x,
                .clip_max_y = clip_max_y,
                .tile_header_offset = narrow_cast<uint32_t>(tile_header_offset),
                .tile_grid_width = tile_grid_width,
                .tile_grid_height = tile_grid_height,
                .triangle_count = emitted_triangle_count,
            });

            ++m_sw_frame_stats.draw_command_count;
            m_sw_frame_stats.triangle_count += emitted_triangle_count;
            m_sw_frame_stats.clip_rect_pixels += clip_rect_pixels;
            m_sw_frame_stats.dispatched_pixels += clip_rect_pixels;
        }
    }

    if (m_sw_draw_commands.empty())
        return;

    ensure_structured_buffer_capacity<SwTriangle>(m_device, triangle_buffer, m_sw_triangles.size(), "imgui sw triangle buffer");
    ensure_structured_buffer_capacity<SwTileHeader>(m_device, tile_header_buffer, m_sw_tile_headers.size(), "imgui sw tile header buffer");
    ensure_structured_buffer_capacity<uint32_t>(m_device, tile_index_buffer, m_sw_tile_triangle_indices.size(), "imgui sw tile index buffer");

    triangle_buffer->set_data(m_sw_triangles.data(), m_sw_triangles.size() * sizeof(SwTriangle));
    tile_header_buffer->set_data(m_sw_tile_headers.data(), m_sw_tile_headers.size() * sizeof(SwTileHeader));
    tile_index_buffer->set_data(
        m_sw_tile_triangle_indices.data(),
        m_sw_tile_triangle_indices.size() * sizeof(uint32_t)
    );
}

void Context::log_sw_frame_stats() const
{
    log_debug(
        "ImGui SW raster: cmds={}, tris={}, clip_pixels={}, dispatched_pixels={}, upload_ms={:.3f}, preprocess_ms={:.3f}, submit_ms={:.3f}",
        m_sw_frame_stats.draw_command_count,
        m_sw_frame_stats.triangle_count,
        m_sw_frame_stats.clip_rect_pixels,
        m_sw_frame_stats.dispatched_pixels,
        m_sw_frame_stats.upload_cpu_ms,
        m_sw_frame_stats.preprocess_cpu_ms,
        m_sw_frame_stats.submit_cpu_ms
    );
}

} // namespace sgl::ui

namespace ImGui {

void PushFont(const char* name)
{
    sgl::ui::Context* ctx = static_cast<sgl::ui::Context*>(ImGui::GetIO().UserData);
    PushFont(ctx->get_font(name), 0.f);
}

} // namespace ImGui
