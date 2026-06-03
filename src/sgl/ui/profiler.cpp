// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler.h"

#include "sgl/device/profiler.h"

#include <imgui.h>

#include <limits>

namespace sgl::ui {

namespace {

    constexpr uint32_t kInvalidProfilerId = std::numeric_limits<uint32_t>::max();

    const char* stats_node_name(const ProfilerStats& stats, const ProfilerStatsNode& node)
    {
        if (node.name_id < stats.names().size() && !stats.names()[node.name_id].name.empty())
            return stats.names()[node.name_id].name.c_str();
        if (node.source_id < stats.sources().size() && !stats.sources()[node.source_id].display_function.empty())
            return stats.sources()[node.source_id].display_function.c_str();
        return "zone";
    }

    void draw_stat_value(const ProfilerStatValue& value)
    {
        if (!value.valid) {
            ImGui::TextUnformatted("-");
            return;
        }
        ImGui::Text("%.3f", value.average_ms);
    }

    void draw_stats_node(const ProfilerStats& stats, uint32_t node_id)
    {
        if (node_id >= stats.nodes().size())
            return;

        const ProfilerStatsNode& node = stats.nodes()[node_id];
        const bool has_children = node.child_index_count > 0;
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanFullWidth;
        if (!has_children)
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const bool open
            = ImGui::TreeNodeEx(reinterpret_cast<void*>(uintptr_t(node.id)), flags, "%s", stats_node_name(stats, node));

        ImGui::TableNextColumn();
        draw_stat_value(node.cpu);
        ImGui::TableNextColumn();
        draw_stat_value(node.gpu);
        ImGui::TableNextColumn();
        ImGui::Text("%u", node.cpu.sample_count);
        ImGui::TableNextColumn();
        ImGui::Text("%u", node.pending_gpu_sample_count);

        if (has_children && open) {
            const uint32_t end = node.child_index_begin + node.child_index_count;
            for (uint32_t i = node.child_index_begin; i < end && i < stats.child_indices().size(); ++i)
                draw_stats_node(stats, stats.child_indices()[i]);
            ImGui::TreePop();
        }
    }

} // namespace

void render_profiler_window(Profiler* profiler)
{
    if (!profiler)
        profiler = current_profiler_or_null();

    if (!ImGui::Begin("Profiler")) {
        ImGui::End();
        return;
    }

    if (!profiler) {
        ImGui::TextUnformatted("No active profiler");
        ImGui::End();
        return;
    }

    ref<ProfilerStats> stats = profiler->stats_snapshot();
    ImGui::Text("Frames: %u  Window: %u", stats->completed_frame_count(), stats->window_size());

    const ImGuiTableFlags table_flags
        = ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("ProfilerStats", 5, table_flags)) {
        ImGui::TableSetupColumn("Zone", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("CPU avg ms", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("GPU avg ms", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Samples", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Pending GPU", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableHeadersRow();

        for (const ProfilerStatsNode& node : stats->nodes()) {
            if (node.parent_id == kInvalidProfilerId)
                draw_stats_node(*stats, node.id);
        }

        ImGui::EndTable();
    }

    ImGui::End();
}

} // namespace sgl::ui
