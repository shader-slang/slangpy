// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler_ui.h"

#include "sgl/ui/ui.h"
#include "sgl/utils/profiler.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace sgl::ui {
namespace {

    enum class ValueMode : uint32_t {
        mean,
        latest,
        p95,
    };
    constexpr std::array<const char*, 3> VALUE_MODE_LABELS{"Mean", "Latest", "P95"};

    enum class GraphMode : uint32_t {
        cpu,
        gpu,
    };
    constexpr std::array<const char*, 2> GRAPH_MODE_LABELS{"CPU", "GPU"};

    struct UIState {
        Profiler* profiler{nullptr};
        bool paused{false};
        ref<ProfilerFrameStats> stats;
        ValueMode value_mode{ValueMode::mean};
        GraphMode graph_mode{GraphMode::gpu};
        bool graph_open{true};
        bool diagnostics_open{false};
        int32_t hovered_entry{-1};
    };

    struct SelectedValue {
        double value{0.0};
        bool valid{false};
    };

    struct CallSummary {
        uint32_t minimum{0};
        uint32_t maximum{0};
        double mean{0.0};
    };

    struct GpuCoverage {
        uint64_t absent{0};
        uint64_t unavailable{0};
        uint64_t complete{0};
        uint64_t incomplete{0};
    };

    constexpr std::array PALETTE{
        IM_COL32(0, 114, 178, 255),
        IM_COL32(230, 159, 0, 255),
        IM_COL32(0, 158, 115, 255),
        IM_COL32(204, 121, 167, 255),
        IM_COL32(86, 180, 233, 255),
        IM_COL32(213, 94, 0, 255),
        IM_COL32(148, 103, 189, 255),
        IM_COL32(27, 158, 119, 255),
        IM_COL32(217, 95, 14, 255),
        IM_COL32(117, 112, 179, 255),
        IM_COL32(231, 138, 195, 255),
        IM_COL32(102, 166, 30, 255),
    };

    constexpr ImVec2 INITIAL_WINDOW_SIZE = ImVec2(800.f, 700.f);

    constexpr float TABLE_HIERARCHY_INDENT = 12.f;
    constexpr float TABLE_COLOR_SWATCH_SIZE = 12.f;
    constexpr uint32_t TABLE_MAX_HIERARCHY_DEPTH = 64;

    constexpr ImU32 SHARE_BAR_BACKGROUND_COLOR = IM_COL32(70, 74, 82, 100);
    constexpr float SHARE_BAR_ROUNDING = 2.f;

    constexpr float GRAPH_HEIGHT = 260.f;
    constexpr float GRAPH_LABEL_WIDTH = 80.f;
    constexpr float GRAPH_TOP_PADDING = 8.f;
    constexpr float GRAPH_RIGHT_PADDING = 4.f;
    constexpr float GRAPH_BOTTOM_PADDING = 8.f;
    constexpr float GRAPH_AXIS_LABEL_OFFSET = 2.f;
    constexpr float GRAPH_AXIS_LABEL_CENTER = 0.5f;
    constexpr float GRAPH_COLUMN_INSET = 0.5f;
    constexpr float GRAPH_OVERFLOW_MARKER_SIZE = 5.f;
    constexpr float GRAPH_INCOMPLETE_MARKER_OFFSET = 3.f;
    constexpr float GRAPH_LINE_THICKNESS = 2.f;
    constexpr uint32_t GRAPH_GRID_COUNT = 4;
    constexpr ImU32 GRAPH_BACKGROUND_COLOR = IM_COL32(22, 24, 29, 255);
    constexpr ImU32 GRAPH_GRID_COLOR = IM_COL32(120, 125, 135, 70);
    constexpr ImU32 GRAPH_LABEL_COLOR = IM_COL32(185, 188, 195, 255);
    constexpr ImU32 GRAPH_HIGHLIGHT_COLOR = IM_COL32(255, 255, 255, 230);
    constexpr ImU32 GRAPH_OVERFLOW_COLOR = IM_COL32(255, 90, 70, 255);
    constexpr ImU32 GRAPH_INCOMPLETE_COLOR = IM_COL32(190, 190, 195, 180);

    const ImVec4 DATA_LOSS_WARNING_COLOR(1.f, 0.48f, 0.35f, 1.f);

    constexpr uint32_t DIAGNOSTICS_LINE_COUNT = 3;

    UIState ui_state;

    bool valid_gpu_status(ProfilerFrameGpuStatus status)
    {
        return status == ProfilerFrameGpuStatus::absent || status == ProfilerFrameGpuStatus::complete;
    }

    CallSummary call_summary(const ProfilerFrameStats& stats, size_t entry_index)
    {
        CallSummary result;
        uint64_t total = 0;
        for (size_t sample_index = 0; sample_index < stats.sample_count(); ++sample_index) {
            const uint32_t calls = stats.sample(sample_index).call_count[entry_index];
            if (sample_index == 0)
                result.minimum = calls;
            else
                result.minimum = std::min(result.minimum, calls);
            result.maximum = std::max(result.maximum, calls);
            total += calls;
        }
        if (stats.sample_count() != 0)
            result.mean = double(total) / double(stats.sample_count());
        return result;
    }

    GpuCoverage gpu_coverage(const ProfilerFrameStats& stats, size_t entry_index)
    {
        GpuCoverage result;
        for (size_t sample_index = 0; sample_index < stats.sample_count(); ++sample_index) {
            switch (stats.sample(sample_index).gpu_status[entry_index]) {
            case ProfilerFrameGpuStatus::absent:
                ++result.absent;
                break;
            case ProfilerFrameGpuStatus::unavailable:
                ++result.unavailable;
                break;
            case ProfilerFrameGpuStatus::complete:
                ++result.complete;
                break;
            case ProfilerFrameGpuStatus::incomplete:
                ++result.incomplete;
                break;
            }
        }
        return result;
    }

    uint32_t latest_calls(const ProfilerFrameStats& stats, size_t entry_index)
    {
        return stats.sample_count() == 0 ? 0 : stats.sample(stats.sample_count() - 1).call_count[entry_index];
    }

    SelectedValue selected_time(const ProfilerFrameStats& stats, size_t entry_index, ValueMode mode, bool gpu)
    {
        const ProfilerFrameStatsEntry& entry = stats.entries()[entry_index];
        if (mode == ValueMode::latest) {
            if (stats.sample_count() == 0)
                return {};
            const ProfilerFrameStatsSampleView sample = stats.sample(stats.sample_count() - 1);
            if (gpu)
                return {sample.gpu_time_ms[entry_index], valid_gpu_status(sample.gpu_status[entry_index])};
            return {sample.cpu_time_ms[entry_index], true};
        }
        const ProfilerDurationStatistics& statistics = gpu ? entry.gpu_time_per_frame : entry.cpu_time_per_frame;
        if (statistics.count == 0)
            return {};
        return {mode == ValueMode::mean ? statistics.mean_ms : statistics.p95_ms, true};
    }

    ImU32 entry_color(const ProfilerFrameStats& stats, size_t entry_index)
    {
        std::vector<uint32_t> path;
        int32_t index = int32_t(entry_index);
        while (index >= 0 && size_t(index) < stats.entries().size() && path.size() <= stats.entries().size()) {
            path.push_back(stats.entries()[size_t(index)].site_id);
            index = stats.entries()[size_t(index)].parent_index;
        }
        uint64_t hash = 1469598103934665603ull;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            hash ^= *it;
            hash *= 1099511628211ull;
        }
        return PALETTE[size_t(hash % PALETTE.size())];
    }

    ImU32 muted_color(ImU32 color)
    {
        ImVec4 value = ImGui::ColorConvertU32ToFloat4(color);
        value.x = value.x * 0.35f + 0.35f;
        value.y = value.y * 0.35f + 0.35f;
        value.z = value.z * 0.35f + 0.35f;
        value.w = 0.55f;
        return ImGui::ColorConvertFloat4ToU32(value);
    }

    void show_duration_statistics(const char* label, const ProfilerDurationStatistics& statistics)
    {
        if (statistics.count == 0) {
            ImGui::Text("%s: n/a", label);
            return;
        }
        ImGui::Text(
            "%s/frame (%llu samples): min %.3f, max %.3f, mean %.3f, stddev %.3f ms",
            label,
            static_cast<unsigned long long>(statistics.count),
            statistics.minimum_ms,
            statistics.maximum_ms,
            statistics.mean_ms,
            statistics.standard_deviation_ms
        );
        ImGui::Text(
            "  P50 %.3f, P90 %.3f, P95 %.3f, P99 %.3f ms",
            statistics.p50_ms,
            statistics.p90_ms,
            statistics.p95_ms,
            statistics.p99_ms
        );
    }

    void show_call_statistics(const char* label, const ProfilerCallStatistics& statistics)
    {
        if (statistics.count == 0) {
            ImGui::Text("%s/call: n/a", label);
            return;
        }
        ImGui::Text(
            "%s/call (%llu calls): min %.3f, max %.3f, mean %.3f, stddev %.3f ms",
            label,
            static_cast<unsigned long long>(statistics.count),
            statistics.minimum_ms,
            statistics.maximum_ms,
            statistics.mean_ms,
            statistics.standard_deviation_ms
        );
    }

    void show_entry_tooltip(const ProfilerFrameStats& stats, size_t entry_index)
    {
        const ProfilerFrameStatsEntry& entry = stats.entries()[entry_index];
        const CallSummary calls = call_summary(stats, entry_index);
        const GpuCoverage coverage = gpu_coverage(stats, entry_index);
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(entry.name.c_str());
        ImGui::Separator();
        ImGui::Text("Calls/frame: min %u, max %u, mean %.2f", calls.minimum, calls.maximum, calls.mean);
        show_duration_statistics("CPU", entry.cpu_time_per_frame);
        show_call_statistics("CPU", entry.cpu_time_per_call);
        show_duration_statistics("GPU", entry.gpu_time_per_frame);
        show_call_statistics("GPU", entry.gpu_time_per_call);
        ImGui::Text(
            "GPU frames: %llu complete, %llu unavailable, %llu incomplete, %llu absent",
            static_cast<unsigned long long>(coverage.complete),
            static_cast<unsigned long long>(coverage.unavailable),
            static_cast<unsigned long long>(coverage.incomplete),
            static_cast<unsigned long long>(coverage.absent)
        );
        ImGui::EndTooltip();
    }

    bool item_hovered_with_tooltip(const ProfilerFrameStats& stats, size_t entry_index)
    {
        if (!ImGui::IsItemHovered())
            return false;
        show_entry_tooltip(stats, entry_index);
        return true;
    }

    bool draw_share_bar(const char* id, double fraction, ImU32 color)
    {
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 size(std::max(1.f, ImGui::GetContentRegionAvail().x), ImGui::GetTextLineHeight());
        ImGui::InvisibleButton(id, size);
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(
            origin,
            ImVec2(origin.x + size.x, origin.y + size.y),
            SHARE_BAR_BACKGROUND_COLOR,
            SHARE_BAR_ROUNDING
        );
        const float fill = float(std::clamp(fraction, 0.0, 1.0));
        if (fill > 0.f)
            draw->AddRectFilled(origin, ImVec2(origin.x + size.x * fill, origin.y + size.y), color, SHARE_BAR_ROUNDING);
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%.2f%% of summed root-zone time", fraction * 100.0);
            ImGui::EndTooltip();
            return true;
        }
        return false;
    }

    void draw_table(UIState& state, const ProfilerFrameStats& stats, float height, int32_t highlighted_entry)
    {
        double cpu_root_total = 0.0;
        double gpu_root_total = 0.0;
        for (size_t index = 0; index < stats.entries().size(); ++index) {
            if (stats.entries()[index].parent_index >= 0)
                continue;
            cpu_root_total += selected_time(stats, index, state.value_mode, false).value;
            const SelectedValue gpu = selected_time(stats, index, state.value_mode, true);
            if (gpu.valid)
                gpu_root_total += gpu.value;
        }

        if (!ImGui::BeginTable(
                "profiler_frame_stats",
                6,
                ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY
                    | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp,
                ImVec2(0.f, height)
            ))
            return;
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Zone", ImGuiTableColumnFlags_WidthStretch, 2.2f);
        ImGui::TableSetupColumn("Calls/frame", ImGuiTableColumnFlags_WidthStretch, 0.85f);
        ImGui::TableSetupColumn("CPU time", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("CPU share", ImGuiTableColumnFlags_WidthStretch, 0.9f);
        ImGui::TableSetupColumn("GPU time", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("GPU share", ImGuiTableColumnFlags_WidthStretch, 0.9f);
        ImGui::TableHeadersRow();

        for (size_t index = 0; index < stats.entries().size(); ++index) {
            const ProfilerFrameStatsEntry& entry = stats.entries()[index];
            const ImU32 color = entry_color(stats, index);
            ImGui::PushID(int(index));
            ImGui::TableNextRow();
            if (highlighted_entry == int32_t(index))
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, muted_color(color));

            bool row_hovered = false;
            ImGui::TableNextColumn();
            uint32_t depth = 0;
            int32_t parent = entry.parent_index;
            while (parent >= 0 && size_t(parent) < index && depth < TABLE_MAX_HIERARCHY_DEPTH) {
                ++depth;
                parent = stats.entries()[size_t(parent)].parent_index;
            }
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + float(depth) * TABLE_HIERARCHY_INDENT);
            ImGui::ColorButton(
                "##color",
                ImGui::ColorConvertU32ToFloat4(color),
                ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                ImVec2(TABLE_COLOR_SWATCH_SIZE, TABLE_COLOR_SWATCH_SIZE)
            );
            row_hovered |= item_hovered_with_tooltip(stats, index);
            ImGui::SameLine();
            ImGui::TextUnformatted(entry.name.c_str());
            row_hovered |= item_hovered_with_tooltip(stats, index);

            ImGui::TableNextColumn();
            ImGui::Text("%u", latest_calls(stats, index));
            row_hovered |= item_hovered_with_tooltip(stats, index);

            const SelectedValue cpu = selected_time(stats, index, state.value_mode, false);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f ms", cpu.value);
            row_hovered |= item_hovered_with_tooltip(stats, index);
            ImGui::TableNextColumn();
            row_hovered
                |= draw_share_bar("##cpu_share", cpu_root_total > 0.0 ? cpu.value / cpu_root_total : 0.0, color);

            const SelectedValue gpu = selected_time(stats, index, state.value_mode, true);
            ImGui::TableNextColumn();
            if (gpu.valid)
                ImGui::Text("%.3f ms", gpu.value);
            else
                ImGui::TextUnformatted("n/a");
            row_hovered |= item_hovered_with_tooltip(stats, index);
            ImGui::TableNextColumn();
            if (gpu.valid) {
                row_hovered
                    |= draw_share_bar("##gpu_share", gpu_root_total > 0.0 ? gpu.value / gpu_root_total : 0.0, color);
            } else {
                ImGui::TextDisabled("n/a");
                row_hovered |= item_hovered_with_tooltip(stats, index);
            }

            if (row_hovered)
                state.hovered_entry = int32_t(index);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    void draw_graph(UIState& state, const ProfilerFrameStats& stats, float height, int32_t highlighted_entry)
    {
        const bool gpu = state.graph_mode == GraphMode::gpu;
        if (stats.sample_count() == 0) {
            ImGui::TextUnformatted("No completed frame history");
            return;
        }
        if (stats.entries().empty()) {
            ImGui::TextUnformatted("No profiled zones in completed frame history");
            return;
        }

        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 size(std::max(1.f, ImGui::GetContentRegionAvail().x), height);
        ImGui::InvisibleButton("profiler_history", size);
        const bool graph_hovered = ImGui::IsItemHovered();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y), GRAPH_BACKGROUND_COLOR);

        const float plot_left = origin.x + GRAPH_LABEL_WIDTH;
        const float plot_right = origin.x + size.x - GRAPH_RIGHT_PADDING;
        const float plot_top = origin.y + GRAPH_TOP_PADDING;
        const float plot_bottom = origin.y + size.y - GRAPH_BOTTOM_PADDING;
        const float plot_width = std::max(1.f, plot_right - plot_left);
        const float plot_height = std::max(1.f, plot_bottom - plot_top);

        double maximum = 0.0;
        for (size_t sample_index = 0; sample_index < stats.sample_count(); ++sample_index) {
            const ProfilerFrameStatsSampleView sample = stats.sample(sample_index);
            double total = 0.0;
            for (size_t index = 0; index < stats.entries().size(); ++index) {
                if (stats.entries()[index].parent_index < 0)
                    total += gpu ? sample.gpu_time_ms[index] : sample.cpu_time_ms[index];
            }
            maximum = std::max(maximum, total);
        }
        if (maximum <= 0.0)
            maximum = 1.0;

        for (uint32_t grid = 0; grid <= GRAPH_GRID_COUNT; ++grid) {
            const float y = plot_bottom - plot_height * float(grid) / float(GRAPH_GRID_COUNT);
            draw->AddLine(ImVec2(plot_left, y), ImVec2(plot_right, y), GRAPH_GRID_COLOR);
            char label[32];
            snprintf(label, sizeof(label), "%.1f ms", maximum * double(grid) / double(GRAPH_GRID_COUNT));
            draw->AddText(
                ImVec2(origin.x + GRAPH_AXIS_LABEL_OFFSET, y - ImGui::GetTextLineHeight() * GRAPH_AXIS_LABEL_CENTER),
                GRAPH_LABEL_COLOR,
                label
            );
        }

        const float column_width = plot_width / float(stats.sample_count());
        int32_t hovered_entry = -1;
        size_t hovered_sample = 0;
        bool hovered_overflow = false;
        bool hovered_sample_incomplete = false;
        std::vector<bool> sample_incomplete_flags(gpu ? stats.sample_count() : 0);
        std::vector<double> band_bottom(stats.entries().size());
        std::vector<double> band_height(stats.entries().size());
        std::vector<double> child_used(stats.entries().size());
        std::vector<bool> overflow(stats.entries().size());
        const ImVec2 mouse = ImGui::GetIO().MousePos;

        for (size_t sample_index = 0; sample_index < stats.sample_count(); ++sample_index) {
            const ProfilerFrameStatsSampleView sample = stats.sample(sample_index);
            const float x0 = plot_left + column_width * float(sample_index) + GRAPH_COLUMN_INSET;
            const float x1 = plot_left + column_width * float(sample_index + 1) - GRAPH_COLUMN_INSET;
            std::fill(band_bottom.begin(), band_bottom.end(), 0.0);
            std::fill(band_height.begin(), band_height.end(), 0.0);
            std::fill(child_used.begin(), child_used.end(), 0.0);
            std::fill(overflow.begin(), overflow.end(), false);

            double root_bottom = 0.0;
            bool sample_incomplete = false;
            for (size_t index = 0; index < stats.entries().size(); ++index) {
                const ProfilerFrameStatsEntry& entry = stats.entries()[index];
                const double value = gpu ? sample.gpu_time_ms[index] : sample.cpu_time_ms[index];
                if (gpu && !valid_gpu_status(sample.gpu_status[index]))
                    sample_incomplete = true;
                if (entry.parent_index < 0) {
                    band_bottom[index] = root_bottom;
                    band_height[index] = value;
                    root_bottom += value;
                } else {
                    const size_t parent = size_t(entry.parent_index);
                    const double available = std::max(0.0, band_height[parent] - child_used[parent]);
                    band_bottom[index] = band_bottom[parent] + child_used[parent];
                    band_height[index] = std::min(value, available);
                    if (value > available + 1e-9)
                        overflow[parent] = true;
                    child_used[parent] += value;
                }

                if (band_height[index] <= 0.0)
                    continue;
                const float y0 = plot_bottom - float((band_bottom[index] + band_height[index]) / maximum) * plot_height;
                const float y1 = plot_bottom - float(band_bottom[index] / maximum) * plot_height;
                ImU32 color = entry_color(stats, index);
                if (gpu && !valid_gpu_status(sample.gpu_status[index]))
                    color = muted_color(color);
                draw->AddRectFilled(ImVec2(x0, y0), ImVec2(std::max(x0 + 1.f, x1), y1), color);
                if (highlighted_entry == int32_t(index))
                    draw->AddRect(
                        ImVec2(x0, y0),
                        ImVec2(std::max(x0 + 1.f, x1), y1),
                        GRAPH_HIGHLIGHT_COLOR,
                        0.f,
                        0,
                        GRAPH_LINE_THICKNESS
                    );
                if (graph_hovered && mouse.x >= x0 && mouse.x <= x1 && mouse.y >= y0 && mouse.y <= y1) {
                    hovered_entry = int32_t(index);
                    hovered_sample = sample_index;
                }
            }

            for (size_t index = 0; index < overflow.size(); ++index) {
                if (!overflow[index])
                    continue;
                const float y = plot_bottom - float((band_bottom[index] + band_height[index]) / maximum) * plot_height;
                draw->AddTriangleFilled(
                    ImVec2(x1 - GRAPH_OVERFLOW_MARKER_SIZE, y),
                    ImVec2(x1, y),
                    ImVec2(x1, y + GRAPH_OVERFLOW_MARKER_SIZE),
                    GRAPH_OVERFLOW_COLOR
                );
            }
            if (hovered_entry >= 0 && hovered_sample == sample_index)
                hovered_overflow = overflow[size_t(hovered_entry)];

            if (gpu)
                sample_incomplete_flags[sample_index] = sample_incomplete;
            if (hovered_entry >= 0 && hovered_sample == sample_index)
                hovered_sample_incomplete = sample_incomplete;
            if (gpu && sample_incomplete)
                draw->AddLine(
                    ImVec2(x0, plot_bottom + GRAPH_INCOMPLETE_MARKER_OFFSET),
                    ImVec2(x1, plot_bottom + GRAPH_INCOMPLETE_MARKER_OFFSET),
                    GRAPH_INCOMPLETE_COLOR,
                    GRAPH_LINE_THICKNESS
                );
        }

        if (hovered_entry >= 0) {
            state.hovered_entry = hovered_entry;
            const ProfilerFrameStatsSampleView sample = stats.sample(hovered_sample);
            const size_t index = size_t(hovered_entry);
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(stats.entries()[index].name.c_str());
            ImGui::Text(
                "Frame %u: %.3f ms (%u calls)",
                sample.frame_index,
                gpu ? sample.gpu_time_ms[index] : sample.cpu_time_ms[index],
                sample.call_count[index]
            );
            ImGui::Text("Frame time: %.3f ms", sample.frame_time_ms);
            if (gpu && sample.gpu_status[index] == ProfilerFrameGpuStatus::unavailable)
                ImGui::TextUnformatted("GPU timing was not requested; value is muted.");
            if (gpu && sample.gpu_status[index] == ProfilerFrameGpuStatus::incomplete)
                ImGui::TextUnformatted("GPU timing is incomplete; value is muted.");
            if (hovered_overflow)
                ImGui::TextUnformatted("Child totals overflow this parent and are clipped.");
            if (gpu && hovered_sample_incomplete)
                ImGui::TextUnformatted("This frame contains incomplete GPU samples.");
            ImGui::EndTooltip();
        } else if (gpu && graph_hovered && mouse.x >= plot_left && mouse.x <= plot_right) {
            const size_t sample_index
                = std::min(stats.sample_count() - 1, size_t(std::max(0.f, mouse.x - plot_left) / column_width));
            if (sample_incomplete_flags[sample_index]) {
                const ProfilerFrameStatsSampleView sample = stats.sample(sample_index);
                ImGui::BeginTooltip();
                ImGui::Text("Frame %u: incomplete GPU timing", sample.frame_index);
                ImGui::Text("Frame time: %.3f ms", sample.frame_time_ms);
                ImGui::TextUnformatted("Muted entries are unavailable or partial measurements.");
                ImGui::EndTooltip();
            }
        }
    }

    template<typename T, size_t N>
    void draw_selector(const char* label, T& value, const std::array<const char*, N>& labels, float width)
    {
        ImGui::SetNextItemWidth(width);
        if (ImGui::BeginCombo(label, labels[static_cast<size_t>(value)])) {
            for (size_t index = 0; index < labels.size(); ++index) {
                const bool selected = index == static_cast<size_t>(value);
                if (ImGui::Selectable(labels[index], selected))
                    value = static_cast<T>(index);
                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    void draw_value_selector(UIState& state)
    {
        draw_selector("Value", state.value_mode, VALUE_MODE_LABELS, 120.f);
    }

    void draw_graph_selector(UIState& state)
    {
        draw_selector("Mode", state.graph_mode, GRAPH_MODE_LABELS, 100.f);
    }

} // namespace

void render_profiler_window(Profiler* profiler)
{
    if (!profiler)
        profiler = current_profiler_or_null();
    if (ui_state.profiler != profiler)
        ui_state = UIState{.profiler = profiler};
    ImGui::SetNextWindowSize(INITIAL_WINDOW_SIZE, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Profiler", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::End();
        return;
    }
    if (!profiler) {
        ImGui::TextUnformatted("No active profiler");
        ImGui::End();
        return;
    }

    UIState& state = ui_state;
    bool enabled = profiler->enabled();
    if (ImGui::Checkbox("Enabled", &enabled))
        profiler->set_enabled(enabled);
    ImGui::SameLine();
    bool automatic = profiler->enable_auto_zones();
    if (ImGui::Checkbox("Automatic zones", &automatic))
        profiler->set_enable_auto_zones(automatic);
    ImGui::SameLine();
    ImGui::Checkbox("Pause display", &state.paused);

    if (!state.paused || !state.stats)
        state.stats = profiler->frame_stats_snapshot();
    if (!state.stats) {
        ImGui::TextUnformatted("No frame statistics available");
        ImGui::End();
        return;
    }

    const ProfilerDiagnostics& diagnostics = state.stats->diagnostics();
    const bool data_loss = diagnostics.producer_drop_count != 0 || diagnostics.hierarchy_loss_count != 0
        || diagnostics.gpu_query_exhaustion_count != 0;
    if (data_loss) {
        ImGui::TextColored(
            DATA_LOSS_WARNING_COLOR,
            "Warning: profiler data loss detected; expand Diagnostics for details."
        );
    }

    ImGui::PushFont("monospace");
    const double latest_frame_ms = state.stats->latest_frame_ms();
    if (latest_frame_ms > 0.0) {
        ImGui::Text(
            "FPS: %.1f  Frame time: %.3f ms latest  %.3f ms mean  %.3f ms P95",
            1000.0 / latest_frame_ms,
            latest_frame_ms,
            state.stats->frame_time().mean_ms,
            state.stats->frame_time().p95_ms
        );
    } else {
        ImGui::TextUnformatted("FPS: n/a  Frame time: n/a");
    }
    draw_value_selector(state);

    const bool graph_has_data = state.stats->sample_count() != 0 && !state.stats->entries().empty();
    const float graph_content_height = graph_has_data ? GRAPH_HEIGHT : ImGui::GetTextLineHeight();
    float bottom_height = ImGui::GetFrameHeightWithSpacing();
    if (state.graph_open)
        bottom_height += ImGui::GetFrameHeightWithSpacing() + graph_content_height + ImGui::GetStyle().ItemSpacing.y;
    bottom_height += ImGui::GetFrameHeight();
    if (state.diagnostics_open)
        bottom_height += DIAGNOSTICS_LINE_COUNT * ImGui::GetTextLineHeightWithSpacing();

    const int32_t previous_hovered = state.hovered_entry;
    state.hovered_entry = -1;
    const float table_height = std::max(1.f, ImGui::GetContentRegionAvail().y - bottom_height);
    draw_table(state, *state.stats, table_height, previous_hovered);

    ImGui::SetNextItemOpen(state.graph_open, ImGuiCond_Always);
    state.graph_open = ImGui::CollapsingHeader("Graph");
    if (state.graph_open) {
        draw_graph_selector(state);
        draw_graph(
            state,
            *state.stats,
            GRAPH_HEIGHT,
            state.hovered_entry >= 0 ? state.hovered_entry : previous_hovered
        );
    }

    ImGui::SetNextItemOpen(state.diagnostics_open, ImGuiCond_Always);
    state.diagnostics_open = ImGui::CollapsingHeader("Diagnostics");
    if (state.diagnostics_open) {
        ImGui::Text(
            "Pending frames: %llu  pending GPU zones: %llu",
            static_cast<unsigned long long>(state.stats->pending_frame_count()),
            static_cast<unsigned long long>(diagnostics.pending_gpu_zone_count)
        );
        ImGui::Text(
            "Dropped producer events: %llu  hierarchy losses: %llu  GPU query exhaustion: %llu",
            static_cast<unsigned long long>(diagnostics.producer_drop_count),
            static_cast<unsigned long long>(diagnostics.hierarchy_loss_count),
            static_cast<unsigned long long>(diagnostics.gpu_query_exhaustion_count)
        );
        ImGui::Text(
            "Thread event queue high-water mark: %llu",
            static_cast<unsigned long long>(diagnostics.thread_event_queue_high_water_mark)
        );
    }
    ImGui::PopFont();
    ImGui::End();
}

} // namespace sgl::ui
