// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "profiler_ui.h"

#include "sgl/ui/ui.h"
#include "sgl/utils/profiler.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <chrono>
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
        std::vector<ImU32> entry_colors;
        ValueMode value_mode{ValueMode::mean};
        GraphMode graph_mode{GraphMode::gpu};
        bool graph_open{true};
        bool diagnostics_open{false};
        std::vector<uint32_t> selected_path;
        int32_t selected_entry{-1};
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

    constexpr ImVec2 INITIAL_WINDOW_SIZE = ImVec2(940.f, 740.f);

    constexpr float TABLE_HIERARCHY_INDENT = 12.f;
    constexpr float TABLE_COLOR_SWATCH_SIZE = 12.f;
    constexpr uint32_t TABLE_MAX_HIERARCHY_DEPTH = 64;


    constexpr float GRAPH_HEIGHT = 260.f;
    constexpr float GRAPH_LABEL_WIDTH = 80.f;
    constexpr float GRAPH_TOP_PADDING = 8.f;
    constexpr float GRAPH_RIGHT_PADDING = 4.f;
    constexpr float GRAPH_BOTTOM_PADDING = 8.f;
    constexpr float GRAPH_AXIS_LABEL_OFFSET = 2.f;
    constexpr float GRAPH_AXIS_LABEL_CENTER = 0.5f;
    constexpr float GRAPH_COLUMN_INSET = 0.5f;
    constexpr float GRAPH_INCOMPLETE_MARKER_OFFSET = 3.f;
    constexpr float GRAPH_LINE_THICKNESS = 2.f;
    constexpr uint32_t GRAPH_GRID_COUNT = 4;
    constexpr ImU32 GRAPH_BACKGROUND_COLOR = IM_COL32(22, 24, 29, 255);
    constexpr ImU32 GRAPH_GRID_COLOR = IM_COL32(120, 125, 135, 70);
    constexpr ImU32 GRAPH_LABEL_COLOR = IM_COL32(185, 188, 195, 255);
    constexpr ImU32 GRAPH_INCOMPLETE_COLOR = IM_COL32(190, 190, 195, 180);

    const ImVec4 DATA_LOSS_WARNING_COLOR(1.f, 0.48f, 0.35f, 1.f);

    constexpr uint32_t DIAGNOSTICS_LINE_COUNT = 4;

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

    double selected_calls(const ProfilerFrameStats& stats, size_t entry_index, ValueMode mode)
    {
        if (stats.sample_count() == 0)
            return 0.0;
        if (mode == ValueMode::latest)
            return stats.sample(stats.sample_count() - 1).call_count[entry_index];
        if (mode == ValueMode::mean)
            return call_summary(stats, entry_index).mean;
        std::vector<uint32_t> counts;
        counts.reserve(stats.sample_count());
        for (size_t i = 0; i < stats.sample_count(); ++i)
            counts.push_back(stats.sample(i).call_count[entry_index]);
        std::sort(counts.begin(), counts.end());
        const double position = 0.95 * double(counts.size() - 1);
        const size_t lower = size_t(position);
        const size_t upper = std::min(lower + 1, counts.size() - 1);
        return double(counts[lower]) + (double(counts[upper]) - double(counts[lower])) * (position - double(lower));
    }

    std::vector<uint32_t> entry_path(const ProfilerFrameStats& stats, size_t index)
    {
        std::vector<uint32_t> path;
        for (int32_t current = int32_t(index); current >= 0; current = stats.entries()[size_t(current)].parent_index)
            path.push_back(stats.entries()[size_t(current)].site_id);
        return path;
    }

    void update_selection(UIState& state)
    {
        state.selected_entry = -1;
        for (size_t i = 0; i < state.stats->entry_count(); ++i) {
            if (!state.selected_path.empty() && state.stats->entries()[i].site_id == state.selected_path.front()
                && entry_path(*state.stats, i) == state.selected_path) {
                state.selected_entry = int32_t(i);
                return;
            }
        }
        if (state.stats->entries().empty())
            return;
        state.selected_entry = 0;
        state.selected_path = entry_path(*state.stats, 0);
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

    std::vector<ImU32> make_entry_colors(const ProfilerFrameStats& stats)
    {
        std::vector<uint64_t> hashes(stats.entries().size());
        std::vector<ImU32> colors(stats.entries().size());
        for (size_t index = 0; index < stats.entries().size(); ++index) {
            const ProfilerFrameStatsEntry& entry = stats.entries()[index];
            uint64_t hash = entry.parent_index < 0 ? 1469598103934665603ull : hashes[size_t(entry.parent_index)];
            hash ^= entry.site_id;
            hash *= 1099511628211ull;
            hashes[index] = hash;
            colors[index] = PALETTE[size_t(hash % PALETTE.size())];
        }
        return colors;
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
        ImGui::TextUnformatted("Times are inclusive; parent and child rows are not additive.");
        ImGui::TextUnformatted("GPU means and percentiles exclude unavailable and incomplete frames.");
        ImGui::TextUnformatted("Absent occurrences contribute zero. CPU time includes waits inside the scope.");
        ImGui::EndTooltip();
    }

    bool item_hovered_with_tooltip(const ProfilerFrameStats& stats, size_t entry_index)
    {
        if (!ImGui::IsItemHovered())
            return false;
        show_entry_tooltip(stats, entry_index);
        return true;
    }

    void draw_table(UIState& state, float height)
    {
        const ProfilerFrameStats& stats = *state.stats;
        if (!ImGui::BeginTable(
                "profiler_frame_stats",
                4,
                ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY
                    | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp,
                ImVec2(0.f, height)
            ))
            return;
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Zone (select to graph)", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableSetupColumn("Calls/frame", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("CPU ms/frame", ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableSetupColumn("GPU ms/frame", ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableHeadersRow();

        for (size_t index = 0; index < stats.entries().size(); ++index) {
            const ProfilerFrameStatsEntry& entry = stats.entries()[index];
            ImGui::PushID(int(index));
            ImGui::TableNextRow();
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
                ImGui::ColorConvertU32ToFloat4(state.entry_colors[index]),
                ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                ImVec2(TABLE_COLOR_SWATCH_SIZE, TABLE_COLOR_SWATCH_SIZE)
            );
            ImGui::SameLine();
            if (ImGui::Selectable(entry.name.c_str(), state.selected_entry == int32_t(index))) {
                state.selected_entry = int32_t(index);
                state.selected_path = entry_path(stats, index);
            }
            item_hovered_with_tooltip(stats, index);

            ImGui::TableNextColumn();
            ImGui::Text("%.2f", selected_calls(stats, index, state.value_mode));
            item_hovered_with_tooltip(stats, index);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", selected_time(stats, index, state.value_mode, false).value);
            item_hovered_with_tooltip(stats, index);
            ImGui::TableNextColumn();
            const SelectedValue gpu = selected_time(stats, index, state.value_mode, true);
            if (gpu.valid)
                ImGui::Text("%.3f", gpu.value);
            else
                ImGui::TextUnformatted("n/a");
            item_hovered_with_tooltip(stats, index);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    void draw_graph(UIState& state, float height)
    {
        const ProfilerFrameStats& stats = *state.stats;
        if (stats.sample_count() == 0 || state.selected_entry < 0) {
            ImGui::TextUnformatted("No completed zone history. Record zones inside profile_frame().");
            return;
        }
        const size_t root = size_t(state.selected_entry);
        const bool gpu = state.graph_mode == GraphMode::gpu;
        // Entries are ordered parent before child. Restrict the stack to the selected subtree.
        std::vector<size_t> entries;
        std::vector<bool> included(stats.entry_count(), false);
        included[root] = true;
        for (size_t index = root; index < stats.entry_count(); ++index) {
            const int32_t parent = stats.entries()[index].parent_index;
            if (parent >= 0 && included[size_t(parent)])
                included[index] = true;
            if (included[index])
                entries.push_back(index);
        }

        // Partition inclusive durations instead of summing parents and children twice.
        // GPU hierarchy need not imply containment: expand to fit children rather than
        // clipping or rescaling their measurements to a missing/smaller parent.
        std::vector<double> bands(stats.sample_count() * entries.size());
        std::vector<double> child_totals(stats.entry_count());
        std::vector<bool> missing(stats.sample_count(), false);
        double maximum = 0.0;
        for (size_t i = 0; i < stats.sample_count(); ++i) {
            const auto sample = stats.sample(i);
            std::fill(child_totals.begin(), child_totals.end(), 0.0);
            for (size_t j = entries.size(); j-- > 0;) {
                const size_t index = entries[j];
                const bool valid = !gpu || valid_gpu_status(sample.gpu_status[index]);
                missing[i] = missing[i] || !valid;
                const double measured = valid ? (gpu ? sample.gpu_time_ms[index] : sample.cpu_time_ms[index]) : 0.0;
                const double total = std::max(measured, child_totals[index]);
                bands[i * entries.size() + j] = total - child_totals[index];
                if (index == root)
                    maximum = std::max(maximum, total);
                else
                    child_totals[size_t(stats.entries()[index].parent_index)] += total;
            }
        }
        if (maximum <= 0.0)
            maximum = 1.0;
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
        for (uint32_t grid = 0; grid <= GRAPH_GRID_COUNT; ++grid) {
            const float y = plot_bottom - plot_height * float(grid) / float(GRAPH_GRID_COUNT);
            draw->AddLine(ImVec2(plot_left, y), ImVec2(plot_right, y), GRAPH_GRID_COLOR);
            char label[32];
            snprintf(label, sizeof(label), "%.3g ms", maximum * double(grid) / double(GRAPH_GRID_COUNT));
            draw->AddText(
                ImVec2(origin.x + GRAPH_AXIS_LABEL_OFFSET, y - ImGui::GetTextLineHeight() * GRAPH_AXIS_LABEL_CENTER),
                GRAPH_LABEL_COLOR,
                label
            );
        }
        const float column_width = plot_width / float(stats.sample_count());
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const bool plot_hovered = graph_hovered && mouse.x >= plot_left && mouse.x <= plot_right;
        const size_t hovered_sample
            = plot_hovered ? std::min(stats.sample_count() - 1, size_t((mouse.x - plot_left) / column_width)) : 0;
        size_t hovered_entry = root;
        double hovered_band = 0.0;
        for (size_t i = 0; i < stats.sample_count(); ++i) {
            const float inset = std::min(GRAPH_COLUMN_INSET, column_width * 0.1f);
            const float x0 = plot_left + column_width * float(i) + inset;
            const float x1 = plot_left + column_width * float(i + 1) - inset;
            double offset = 0.0;
            for (size_t j = 0; j < entries.size(); ++j) {
                const double band = bands[i * entries.size() + j];
                if (band <= 0.0)
                    continue;
                const float bottom = plot_bottom - float(offset / maximum) * plot_height;
                offset += band;
                const float top = plot_bottom - float(offset / maximum) * plot_height;
                draw->AddRectFilled(ImVec2(x0, top), ImVec2(x1, bottom), state.entry_colors[entries[j]]);
                if (plot_hovered && i == hovered_sample && mouse.y >= top && mouse.y <= bottom) {
                    hovered_entry = entries[j];
                    hovered_band = band;
                }
            }
            if (missing[i]) {
                draw->AddLine(
                    ImVec2(x0, plot_bottom + GRAPH_INCOMPLETE_MARKER_OFFSET),
                    ImVec2(x1, plot_bottom + GRAPH_INCOMPLETE_MARKER_OFFSET),
                    GRAPH_INCOMPLETE_COLOR,
                    GRAPH_LINE_THICKNESS
                );
            }
        }
        if (plot_hovered) {
            const auto sample = stats.sample(hovered_sample);
            std::string path = stats.entries()[hovered_entry].name;
            for (int32_t parent = stats.entries()[hovered_entry].parent_index; parent >= 0;
                 parent = stats.entries()[size_t(parent)].parent_index)
                path = stats.entries()[size_t(parent)].name + " / " + path;
            ImGui::BeginTooltip();
            ImGui::Text("Frame %u: %s", sample.frame_index, path.c_str());
            if (gpu && !valid_gpu_status(sample.gpu_status[hovered_entry]))
                ImGui::Text("GPU timing: %s", enum_to_string(sample.gpu_status[hovered_entry]).c_str());
            else
                ImGui::Text(
                    "%s (inclusive): %.3f ms",
                    gpu ? "GPU measured" : "CPU elapsed",
                    gpu ? sample.gpu_time_ms[hovered_entry] : sample.cpu_time_ms[hovered_entry]
                );
            if (hovered_band > 0.0)
                ImGui::Text("Band after accounting for descendants: %.3f ms", hovered_band);
            ImGui::Text("Calls: %u; CPU frame scope: %.3f ms", sample.call_count[hovered_entry], sample.frame_time_ms);
            ImGui::TextUnformatted("Bands partition inclusive times; missing or smaller parents do not clip children.");
            if (gpu) {
                ImGui::TextUnformatted(
                    "Grouped GPU measurements, not total GPU frame time or measured exclusive time."
                );
                if (missing[hovered_sample])
                    ImGui::TextUnformatted("Some GPU measurements in this subtree are unavailable or incomplete.");
            }
            ImGui::EndTooltip();
        }
    }

    SelectedValue selected_frame_time(const ProfilerFrameStats& stats, ValueMode mode)
    {
        if (stats.sample_count() == 0)
            return {};
        if (mode == ValueMode::latest)
            return {stats.sample(stats.sample_count() - 1).frame_time_ms, true};
        const auto& distribution = stats.frame_time();
        return {mode == ValueMode::mean ? distribution.mean_ms : distribution.p95_ms, distribution.count != 0};
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
    ImGui::SameLine();
    draw_value_selector(state);

    if (!state.paused || !state.stats) {
        ref<ProfilerFrameStats> stats = profiler->frame_stats_snapshot();
        if (stats != state.stats) {
            state.stats = std::move(stats);
            state.entry_colors = state.stats ? make_entry_colors(*state.stats) : std::vector<ImU32>();
            if (state.stats)
                update_selection(state);
        }
    }
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
    const auto& interval = state.stats->frame_interval();
    if (interval.count != 0 && interval.mean_ms > 0.0)
        ImGui::Text("FPS: %.1f", 1000.0 / interval.mean_ms);
    else
        ImGui::TextUnformatted("FPS: n/a");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted("FPS = 1000 / mean interval between profile_frame starts.");
        ImGui::TextUnformatted("Includes time outside CPU frame scopes. Requires consecutive recorded frames.");
        ImGui::TextUnformatted("Always uses the mean, regardless of the Value selector.");
        ImGui::EndTooltip();
    }
    ImGui::SameLine();
    const SelectedValue frame = selected_frame_time(*state.stats, state.value_mode);
    if (frame.valid)
        ImGui::Text("| CPU frame: %.3f ms", frame.value);
    else
        ImGui::TextUnformatted("| CPU frame: n/a");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted("CPU elapsed time inside profile_frame(), including waits within the scope.");
        ImGui::TextUnformatted("Mean, Latest, or P95 follows the Value selector, as do the table values.");
        ImGui::EndTooltip();
    }

    const bool graph_has_data = state.stats->sample_count() != 0 && !state.stats->entries().empty();
    const float graph_content_height = graph_has_data ? GRAPH_HEIGHT : ImGui::GetTextLineHeight();
    float bottom_height = ImGui::GetFrameHeightWithSpacing();
    if (state.graph_open)
        bottom_height += ImGui::GetFrameHeightWithSpacing() + graph_content_height + ImGui::GetStyle().ItemSpacing.y;
    bottom_height += ImGui::GetFrameHeight();
    if (state.diagnostics_open)
        bottom_height += DIAGNOSTICS_LINE_COUNT * ImGui::GetTextLineHeightWithSpacing();

    const float table_height = std::max(1.f, ImGui::GetContentRegionAvail().y - bottom_height);
    draw_table(state, table_height);

    ImGui::SetNextItemOpen(state.graph_open, ImGuiCond_Always);
    state.graph_open = ImGui::CollapsingHeader("Selected zone history (oldest to newest)");
    if (state.graph_open) {
        draw_graph_selector(state);
        draw_graph(state, GRAPH_HEIGHT);
    }

    ImGui::SetNextItemOpen(state.diagnostics_open, ImGuiCond_Always);
    state.diagnostics_open = ImGui::CollapsingHeader("Diagnostics");
    if (state.diagnostics_open) {
        if (state.stats->sample_count() != 0) {
            const auto latest = state.stats->sample(state.stats->sample_count() - 1);
            const uint64_t now = uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                )
                    .count()
            );
            const double age_ms
                = std::max(0.0, double(now - std::min(now, latest.frame_start_ns)) * 1e-6 - latest.frame_time_ms);
            ImGui::Text(
                "Statistics through frame %u; scope ended %.1f ms ago%s",
                latest.frame_index,
                age_ms,
                state.paused ? " (display paused)" : ""
            );
        } else {
            ImGui::TextUnformatted("No completed statistics");
        }
        ImGui::Text(
            "Frames awaiting GPU: %llu  pending GPU zones: %llu",
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
