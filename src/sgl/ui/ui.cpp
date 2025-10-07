// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ui.h"

#include "sgl/ui/widgets.h"

#include "sgl/core/error.h"
#include "sgl/core/input.h"
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

#include <unordered_map>

CMRC_DECLARE(sgl_data);

namespace sgl::ui {

static void setup_style()
{
    ImGuiStyle& style = ImGui::GetStyle();

    ImVec4* colors = style.Colors;
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
    colors[ImGuiCol_Tab] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
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
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavNoCaptureKeyboard;
    io.IniFilename = nullptr;

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

    // Skip initializing rasterizer if device does not support rasterization.
    // TODO: This will be fixed later when adding sw rasterizer (wip).
    if (!m_device->has_feature(Feature::rasterization)) {
        log_warn("Rasterization is not available and UI will not be rendered!");
        return;
    }

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

    // Setup program.
    m_program = m_device->load_program("sgl/ui/imgui.slang", {"vs_main", "fs_main"});

    // Setup font texture.
    {
        uint8_t* pixels;
        int width;
        int height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        SubresourceData data[1] = {{
            .data = pixels,
            .size = size_t(width * height * 4),
            .row_pitch = size_t(width * 4),
            .slice_pitch = size_t(width * height * 4),
        }};
        m_font_texture = m_device->create_texture({
            .format = Format::rgba8_unorm,
            .width = narrow_cast<uint32_t>(width),
            .height = narrow_cast<uint32_t>(height),
            .usage = TextureUsage::shader_resource,
            .data = data,
        });

        io.Fonts->SetTexID(static_cast<ImTextureID>(m_font_texture.get()));
    }

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

Context::~Context()
{
    ImGui::DestroyContext(m_imgui_context);
}

ImFont* Context::get_font(const char* name)
{
    auto it = m_fonts.find(name);
    return it == m_fonts.end() ? nullptr : it->second;
}

void Context::new_frame(uint32_t width, uint32_t height)
{
    ImGui::SetCurrentContext(m_imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
    io.DeltaTime = static_cast<float>(m_frame_timer.elapsed_s());
    m_frame_timer.reset();

    ImGui::NewFrame();
}

void Context::render(TextureView* texture_view, CommandEncoder* command_encoder)
{
    ImGui::SetCurrentContext(m_imgui_context);
    ImGuiIO& io = ImGui::GetIO();

    bool is_srgb_format = get_format_info(texture_view->format()).is_srgb_format();

    m_screen->render();

    ImGui::Render();

    // Skip actual rendering device does not support it.
    if (!m_device->has_feature(Feature::rasterization))
        return;

    ImDrawData* draw_data = ImGui::GetDrawData();

    if (draw_data->CmdListsCount > 0) {
        // Cycle through vertex & index buffers.
        ref<Buffer>& vertex_buffer = m_vertex_buffers[m_frame_index];
        ref<Buffer>& index_buffer = m_index_buffers[m_frame_index];
        m_frame_index = (m_frame_index + 1) % FRAME_COUNT;

        // Allocate vertex buffer.
        if (!vertex_buffer || vertex_buffer->size() < draw_data->TotalVtxCount * sizeof(ImDrawVert)) {
            vertex_buffer = m_device->create_buffer({
                .size = draw_data->TotalVtxCount * sizeof(ImDrawVert) + 128 * 1024,
                .memory_type = MemoryType::upload,
                .usage = BufferUsage::vertex_buffer,
                .label = "imgui vertex buffer",
            });
        }

        // Allocate index buffer.
        if (!index_buffer || index_buffer->size() < draw_data->TotalIdxCount * sizeof(ImDrawIdx)) {
            index_buffer = m_device->create_buffer({
                .size = draw_data->TotalIdxCount * sizeof(ImDrawIdx) + 1024,
                .memory_type = MemoryType::upload,
                .usage = BufferUsage::index_buffer,
                .label = "imgui index buffer",
            });
        }

        // Upload vertex & index data.
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
}

void Context::render(Texture* texture, CommandEncoder* command_encoder)
{
    // TODO(slang-rhi) use default_view once it is available
    render(texture->create_view({}), command_encoder);
}

bool Context::handle_keyboard_event(const KeyboardEvent& event)
{
    ImGui::SetCurrentContext(m_imgui_context);
    ImGuiIO& io = ImGui::GetIO();

    io.AddKeyEvent(ImGuiKey_ModShift, event.has_modifier(KeyModifier::shift));
    io.AddKeyEvent(ImGuiKey_ModCtrl, event.has_modifier(KeyModifier::ctrl));
    io.AddKeyEvent(ImGuiKey_ModAlt, event.has_modifier(KeyModifier::alt));

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

    io.AddKeyEvent(ImGuiKey_ModShift, event.has_modifier(KeyModifier::shift));
    io.AddKeyEvent(ImGuiKey_ModCtrl, event.has_modifier(KeyModifier::ctrl));
    io.AddKeyEvent(ImGuiKey_ModAlt, event.has_modifier(KeyModifier::alt));

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

void Context::process_events()
{
    m_screen->dispatch_events();
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

} // namespace sgl::ui

namespace ImGui {

void PushFont(const char* name)
{
    sgl::ui::Context* ctx = static_cast<sgl::ui::Context*>(ImGui::GetIO().UserData);
    PushFont(ctx->get_font(name));
}

} // namespace ImGui
