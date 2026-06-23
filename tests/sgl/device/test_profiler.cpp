// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/query.h"
#include "sgl/utils/profiler.h"

#include <chrono>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>

using namespace sgl;

TEST_SUITE_BEGIN("device");

namespace {

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

size_t count_occurrences(std::string_view value, std::string_view pattern)
{
    size_t count = 0;
    size_t pos = 0;
    while ((pos = value.find(pattern, pos)) != std::string_view::npos) {
        ++count;
        pos += pattern.size();
    }
    return count;
}

bool trace_has_name(const ProfilerTrace& trace, std::string_view name)
{
    for (const ProfilerZoneRecord& zone : trace.zones()) {
        if (zone.name_id < trace.names().size() && trace.names()[zone.name_id].name == name)
            return true;
    }
    return false;
}

const ProfilerZoneRecord* find_trace_zone(const ProfilerTrace& trace, std::string_view name)
{
    for (const ProfilerZoneRecord& zone : trace.zones()) {
        if (zone.name_id < trace.names().size() && trace.names()[zone.name_id].name == name)
            return &zone;
    }
    return nullptr;
}

size_t count_trace_zones(const ProfilerTrace& trace, std::string_view name, ProfilerTimelineType timeline_type)
{
    size_t count = 0;
    for (const ProfilerZoneRecord& zone : trace.zones()) {
        if (zone.name_id >= trace.names().size() || trace.names()[zone.name_id].name != name)
            continue;
        if (zone.timeline_id >= trace.timelines().size() || trace.timelines()[zone.timeline_id].type != timeline_type)
            continue;
        ++count;
    }
    return count;
}

const ProfilerZoneRecord*
find_trace_zone(const ProfilerTrace& trace, std::string_view name, ProfilerTimelineType timeline_type)
{
    for (const ProfilerZoneRecord& zone : trace.zones()) {
        if (zone.name_id >= trace.names().size() || trace.names()[zone.name_id].name != name)
            continue;
        if (zone.timeline_id >= trace.timelines().size() || trace.timelines()[zone.timeline_id].type != timeline_type)
            continue;
        return &zone;
    }
    return nullptr;
}

const ProfilerStatsNode* find_stats_node(const ProfilerStats& stats, std::string_view name)
{
    for (const ProfilerStatsNode& node : stats.nodes()) {
        if (node.name_id < stats.names().size() && stats.names()[node.name_id].name == name)
            return &node;
    }
    return nullptr;
}

ref<ProfilerTrace>
wait_for_trace_zones(Profiler& profiler, std::string_view name, ProfilerTimelineType timeline_type, size_t min_count)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    ref<ProfilerTrace> trace;
    do {
        profiler.tick();
        trace = profiler.trace_snapshot();
        if (count_trace_zones(*trace, name, timeline_type) >= min_count)
            return trace;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);

    return trace;
}

ref<ProfilerStats> wait_for_gpu_stats_sample(Profiler& profiler, std::string_view name, uint32_t min_count)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    ref<ProfilerStats> stats;
    do {
        profiler.tick();
        stats = profiler.stats_snapshot();
        const ProfilerStatsNode* node = find_stats_node(*stats, name);
        if (node && node->gpu.valid && node->gpu.sample_count >= min_count)
            return stats;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);

    return stats;
}

ref<ProfilerStats> wait_for_cpu_stats_sample(Profiler& profiler, std::string_view name, uint32_t min_count)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    ref<ProfilerStats> stats;
    do {
        stats = profiler.stats_snapshot();
        const ProfilerStatsNode* node = find_stats_node(*stats, name);
        if (node && node->cpu.valid && node->cpu.sample_count >= min_count)
            return stats;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);

    return stats;
}

} // namespace

TEST_CASE("profiler scope macro is callable")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();
        CHECK(current_profiler_or_null() == nullptr);

        {
            ProfilerScope activation(profiler.get());
            SGL_PROFILE_SCOPE("native_static_zone");
        }

        profiler->tick();
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler scope macro supports explicit interned and dynamic names")
{
    CommandEncoder* encoder = nullptr;
    const char* name = Profiler::intern_name("native_interned_zone");
    std::string dynamic_name = "native_dynamic_zone";

    REQUIRE(name != nullptr);
    CHECK(name == Profiler::intern_name("native_interned_zone"));
    CHECK(std::string_view(name) == "native_interned_zone");

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());

        {
            SGL_PROFILE_SCOPE(name);
            SGL_PROFILE_SCOPE(name, encoder);
            SGL_PROFILE_SCOPE(dynamic_name.c_str(), nullptr, ProfilerZoneFlags::copy_name);
            SGL_PROFILE_SCOPE(dynamic_name.c_str(), encoder, ProfilerZoneFlags::copy_name);
        }

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(trace_has_name(*trace, "native_interned_zone"));
        CHECK(trace_has_name(*trace, "native_dynamic_zone"));
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler function and frame macros record trace data")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_FUNCTION();
        }

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(trace->frames().size() == 1);
        CHECK(trace->zones().size() == 1);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler explicit zone and frame tokens record trace data")
{
    static const ProfilerSourceLocation source_location = {
        __FILE__,
        __LINE__,
        SGL_PRETTY_FUNC,
    };

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();

        ProfilerFrameToken frame_token = profiler->begin_frame(&source_location, "token_frame");
        REQUIRE(frame_token.profiler == profiler.get());

        ProfilerZoneToken zone_token
            = profiler->begin_zone(&source_location, "token_zone", nullptr, ProfilerZoneFlags::none);
        REQUIRE(zone_token.profiler == profiler.get());

        profiler->end_zone(zone_token);
        profiler->end_frame(frame_token);

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(trace->frames().size() == 1);
        CHECK(trace_has_name(*trace, "token_zone"));
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("current profiler free functions are LIFO")
{
    {
        ref<Profiler> profiler_a = make_ref<Profiler>();
        Profiler* profiler_a_ptr = profiler_a.get();
        CHECK(current_profiler_or_null() == nullptr);
        push_current_profiler(profiler_a_ptr);
        CHECK(current_profiler() == profiler_a_ptr);

        {
            ref<Profiler> profiler_b = make_ref<Profiler>();
            Profiler* profiler_b_ptr = profiler_b.get();
            CHECK(current_profiler() == profiler_a_ptr);

            push_current_profiler(profiler_b_ptr);
            CHECK(current_profiler() == profiler_b_ptr);
            CHECK(pop_current_profiler() == profiler_b_ptr);
            CHECK(current_profiler() == profiler_a_ptr);
        }

        CHECK(current_profiler() == profiler_a_ptr);
        CHECK(pop_current_profiler() == profiler_a_ptr);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler destruction removes stack entries")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();
        Profiler* profiler_ptr = profiler.get();

        CHECK(current_profiler_or_null() == nullptr);
        push_current_profiler(profiler_ptr);
        CHECK(current_profiler() == profiler_ptr);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("current profiler is thread local")
{
    bool worker_saw_profiler = false;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        ProfilerScope activation(profiler.get());
        CHECK(current_profiler() == profiler.get());

        std::thread thread(
            [&]()
            {
                worker_saw_profiler = current_profiler_or_null() == profiler.get();
                SGL_PROFILE_SCOPE("worker_zone");
            }
        );
        thread.join();

        profiler->tick();
        CHECK(current_profiler() == profiler.get());
    }

    CHECK(!worker_saw_profiler);
    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler scope is movable")
{
    {
        ref<Profiler> profiler_a = make_ref<Profiler>();
        ref<Profiler> profiler_b = make_ref<Profiler>();
        CHECK(current_profiler_or_null() == nullptr);

        {
            ProfilerScope scope_a(profiler_a.get());
            CHECK(current_profiler() == profiler_a.get());

            ProfilerScope scope_b(profiler_b.get());
            CHECK(current_profiler() == profiler_b.get());

            scope_b = std::move(scope_a);
            CHECK(current_profiler() == profiler_a.get());
        }

        CHECK(current_profiler_or_null() == nullptr);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler interned source locations use structured keys")
{
    const ProfilerSourceLocation* first = Profiler::intern_source_location("c", 7, "d");
    const ProfilerSourceLocation* first_again = Profiler::intern_source_location("c", 7, "d");
    const ProfilerSourceLocation* second = Profiler::intern_source_location("b\nc", 7, "d");

    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    CHECK(first == first_again);
    CHECK(first != second);
    CHECK(std::string_view(first->file) == "c");
    CHECK(first->line == 7);
    CHECK(std::string_view(first->function) == "d");
    CHECK(std::string_view(second->file) == "b\nc");
    CHECK(second->line == 7);
    CHECK(std::string_view(second->function) == "d");
}

TEST_CASE("profiler descriptor defaults and validation")
{
    ProfilerDesc desc;
    CHECK(!desc.trace_enabled_on_start);
    CHECK(desc.stats_window_size == 120);
    CHECK(desc.gpu_query_pool_size == 64 * 1024);
    CHECK(desc.auto_zones_enabled);
    CHECK(!desc.debug_groups_enabled);

    desc.stats_window_size = 0;
    CHECK_THROWS(make_ref<Profiler>(desc));

    desc = {};
    desc.gpu_query_pool_size = 0;
    CHECK_THROWS(make_ref<Profiler>(desc));
}

TEST_CASE("profiler retained settings are mutable")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();

        CHECK(profiler->enabled());
        profiler->set_enabled(false);
        CHECK(!profiler->enabled());

        CHECK(profiler->auto_zones_enabled());
        profiler->set_auto_zones_enabled(false);
        CHECK(!profiler->auto_zones_enabled());

        CHECK(!profiler->debug_groups_enabled());
        profiler->set_debug_groups_enabled(true);
        CHECK(profiler->debug_groups_enabled());

        CHECK(profiler->stats_window_size() == 120);
        profiler->set_stats_window_size(4);
        CHECK(profiler->stats_window_size() == 4);
        CHECK_THROWS(profiler->set_stats_window_size(0));

        CHECK(&profiler->desc() != nullptr);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler records flat cpu trace and stats")
{
    ref<ProfilerTrace> retained_trace;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("outer");
            {
                SGL_PROFILE_SCOPE("inner");
            }
        }

        retained_trace = profiler->trace_snapshot();
        CHECK(retained_trace->frames().size() == 1);
        CHECK(retained_trace->zones().size() == 2);
        CHECK(retained_trace->root_indices().size() == 1);
        CHECK(retained_trace->child_indices().size() == 1);
        CHECK(trace_has_name(*retained_trace, "outer"));
        CHECK(trace_has_name(*retained_trace, "inner"));

        ref<ProfilerStats> stats = profiler->stats_snapshot();
        CHECK(stats->completed_frame_count() == 1);
        CHECK(stats->nodes().size() >= 2);

        profiler->clear_trace();
        ref<ProfilerTrace> cleared_trace = profiler->trace_snapshot();
        CHECK(cleared_trace->zones().empty());

        const std::filesystem::path path = testing::get_case_temp_directory() / "cpu-flat-trace.json";
        retained_trace->write_to_json(path);
        const std::string json = read_text_file(path);
        CHECK(json.find("\"name\":\"outer\"") != std::string::npos);
    }

    CHECK(retained_trace);
    CHECK(retained_trace->zones().size() == 2);
    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler records stats by default without retaining trace")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();
        ProfilerScope activation(profiler.get());

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("stats_only_zone");
        }

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(trace->zones().empty());

        ref<ProfilerStats> stats = profiler->stats_snapshot();
        CHECK(stats->completed_frame_count() == 1);
        CHECK(!stats->nodes().empty());
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler stats snapshot is a cached read")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    ref<ProfilerStats> first = profiler->stats_snapshot();
    ref<ProfilerStats> second = profiler->stats_snapshot();
    CHECK(first.get() == second.get());
}

TEST_CASE("profiler records stats without an active frame")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();

        {
            ProfilerScope activation(profiler.get());
            SGL_PROFILE_SCOPE("frameless_stats_zone");
        }

        ref<ProfilerStats> stats = wait_for_cpu_stats_sample(*profiler, "frameless_stats_zone", 1);
        const ProfilerStatsNode* node = find_stats_node(*stats, "frameless_stats_zone");
        REQUIRE(node);
        CHECK(node->cpu.valid);
        CHECK(node->cpu.sample_count == 1);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler stats window is sample based")
{
    {
        ProfilerDesc desc;
        desc.stats_window_size = 2;
        ref<Profiler> profiler = make_ref<Profiler>(desc);

        {
            ProfilerScope activation(profiler.get());
            for (int i = 0; i < 3; ++i) {
                SGL_PROFILE_SCOPE("sample_window_zone");
            }
        }

        profiler->flush();
        ref<ProfilerStats> stats = profiler->stats_snapshot();
        const ProfilerStatsNode* node = find_stats_node(*stats, "sample_window_zone");
        REQUIRE(node);
        CHECK(node->cpu.valid);
        CHECK(node->cpu.sample_count == 2);
        CHECK(stats->window_size() == 2);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler stop trace drops zones still open at stop")
{
    static const ProfilerSourceLocation source_location = {
        __FILE__,
        __LINE__,
        SGL_PRETTY_FUNC,
    };

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();

        {
            ProfilerScope activation(profiler.get());
            SGL_PROFILE_SCOPE("completed_before_stop");
        }

        ProfilerZoneToken open_zone
            = profiler->begin_zone(&source_location, "open_at_stop", nullptr, ProfilerZoneFlags::none);
        REQUIRE(open_zone.profiler == profiler.get());

        profiler->stop_trace();
        profiler->end_zone(open_zone);

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(trace_has_name(*trace, "completed_before_stop"));
        CHECK(!trace_has_name(*trace, "open_at_stop"));
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler clear trace drops currently open zones without corrupting later trace")
{
    static const ProfilerSourceLocation source_location = {
        __FILE__,
        __LINE__,
        SGL_PRETTY_FUNC,
    };

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();

        ProfilerZoneToken open_zone
            = profiler->begin_zone(&source_location, "open_during_clear", nullptr, ProfilerZoneFlags::none);
        REQUIRE(open_zone.profiler == profiler.get());
        profiler->clear_trace();
        profiler->end_zone(open_zone);

        {
            ProfilerScope activation(profiler.get());
            SGL_PROFILE_SCOPE("after_clear_zone");
        }

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(!trace_has_name(*trace, "open_during_clear"));
        CHECK(trace_has_name(*trace, "after_clear_zone"));
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler frames are thread local")
{
    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("main_thread_zone");

            std::thread thread(
                [&]()
                {
                    ProfilerScope worker_activation(profiler.get());
                    SGL_PROFILE_SCOPE("worker_thread_zone");
                }
            );
            thread.join();
        }

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        const ProfilerZoneRecord* main_zone = find_trace_zone(*trace, "main_thread_zone");
        const ProfilerZoneRecord* worker_zone = find_trace_zone(*trace, "worker_thread_zone");

        REQUIRE(main_zone);
        REQUIRE(worker_zone);
        CHECK(main_zone->frame_id != invalid_id);
        CHECK(worker_zone->frame_id == invalid_id);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler records gpu zones in trace json")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());
        ref<CommandEncoder> encoder = device->create_command_encoder();

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("gpu_zone", encoder.get());
        }

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        profiler->tick();
        device->wait_for_submit(submit_id);

        ref<ProfilerTrace> trace = wait_for_trace_zones(*profiler, "gpu_zone", ProfilerTimelineType::gpu, 1);
        const std::filesystem::path path = testing::get_case_temp_directory() / "gpu-zones.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(json.find("\"cat\":\"sgl.cpu\"") != std::string::npos);
        CHECK(json.find("\"cat\":\"sgl.gpu\"") != std::string::npos);
        CHECK(count_occurrences(json, "\"name\":\"gpu_zone\"") >= 2);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("timestamp query state distinguishes reset pending and resolved")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query))
        return;

    ref<QueryPool> query_pool = device->create_query_pool({
        .type = QueryType::timestamp,
        .count = 2,
    });
    CHECK(query_pool->get_result_state(0, 1) == QueryResultState::reset);

    ref<CommandEncoder> encoder = device->create_command_encoder();
    encoder->write_timestamp(query_pool.get(), 0);

    const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
    const QueryResultState submitted_state = query_pool->get_result_state(0, 1);
    CHECK((submitted_state == QueryResultState::pending || submitted_state == QueryResultState::resolved));

    device->wait_for_submit(submit_id);
    CHECK(query_pool->get_result_state(0, 1) == QueryResultState::resolved);
}

TEST_CASE_GPU("profiler keeps gpu stats when snapshot happens before submit")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        ProfilerScope activation(profiler.get());
        ref<CommandEncoder> encoder = device->create_command_encoder();

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("pre_submit_gpu_stats_zone", encoder.get());
        }

        profiler->flush();
        ref<ProfilerStats> pre_submit_stats = profiler->stats_snapshot();
        const ProfilerStatsNode* pre_submit_node = find_stats_node(*pre_submit_stats, "pre_submit_gpu_stats_zone");
        REQUIRE(pre_submit_node);
        CHECK(!pre_submit_node->gpu.valid);

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        device->wait_for_submit(submit_id);

        ref<ProfilerStats> stats = wait_for_gpu_stats_sample(*profiler, "pre_submit_gpu_stats_zone", 1);
        const ProfilerStatsNode* node = find_stats_node(*stats, "pre_submit_gpu_stats_zone");
        REQUIRE(node);
        CHECK(node->gpu.valid);
        CHECK(node->gpu.sample_count >= 1);
        CHECK(node->pending_gpu_sample_count == 0);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler records gpu zones from multiple command buffers")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        static constexpr size_t command_buffer_count = 16;

        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());
        std::vector<ref<CommandBuffer>> command_buffers;
        std::vector<CommandBuffer*> command_buffer_ptrs;
        command_buffers.reserve(command_buffer_count);
        command_buffer_ptrs.reserve(command_buffer_count);

        {
            SGL_PROFILE_FRAME();
            for (size_t i = 0; i < command_buffer_count; ++i) {
                ref<CommandEncoder> encoder = device->create_command_encoder();
                {
                    SGL_PROFILE_SCOPE("multi_encoder_gpu_zone", encoder.get());
                }

                ref<CommandBuffer> command_buffer = encoder->finish();
                command_buffer_ptrs.push_back(command_buffer.get());
                command_buffers.push_back(std::move(command_buffer));
            }
        }

        const uint64_t submit_id = device->submit_command_buffers(command_buffer_ptrs);
        device->wait_for_submit(submit_id);

        ref<ProfilerTrace> trace = wait_for_trace_zones(
            *profiler,
            "multi_encoder_gpu_zone",
            ProfilerTimelineType::gpu,
            command_buffer_count
        );
        const std::filesystem::path path = testing::get_case_temp_directory() / "multi-buffer-gpu-zones.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(count_occurrences(json, "\"name\":\"multi_encoder_gpu_zone\"") >= command_buffer_count * 2);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler skips gpu timing when query pool is exhausted")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        static constexpr size_t zone_count = 20;

        ProfilerDesc desc;
        desc.gpu_query_pool_size = 2;
        ref<Profiler> profiler = make_ref<Profiler>(desc);
        profiler->start_trace();
        ProfilerScope activation(profiler.get());
        ref<CommandEncoder> encoder = device->create_command_encoder();

        {
            SGL_PROFILE_FRAME();
            for (size_t i = 0; i < zone_count; ++i) {
                SGL_PROFILE_SCOPE("many_gpu_zones", encoder.get());
            }
        }

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        device->wait_for_submit(submit_id);
        profiler->flush();

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(count_trace_zones(*trace, "many_gpu_zones", ProfilerTimelineType::cpu) == zone_count);
        CHECK(count_trace_zones(*trace, "many_gpu_zones", ProfilerTimelineType::gpu) <= 1);

        const std::filesystem::path path = testing::get_case_temp_directory() / "exhausted-gpu-query-pool.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(count_occurrences(json, "\"cat\":\"sgl.cpu\",\"name\":\"many_gpu_zones\"") == zone_count);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler ignores discarded gpu recordings")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());

        {
            ref<CommandEncoder> discarded_encoder = device->create_command_encoder();
            {
                SGL_PROFILE_SCOPE("discarded_gpu_zone", discarded_encoder.get());
            }
        }

        ref<CommandEncoder> encoder = device->create_command_encoder();
        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("submitted_after_discard_gpu_zone", encoder.get());
        }

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        device->wait_for_submit(submit_id);

        ref<ProfilerTrace> trace
            = wait_for_trace_zones(*profiler, "submitted_after_discard_gpu_zone", ProfilerTimelineType::gpu, 1);
        const std::filesystem::path path = testing::get_case_temp_directory() / "discarded-gpu-zones.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(json.find("\"cat\":\"sgl.gpu\",\"name\":\"discarded_gpu_zone\"") == std::string::npos);
        CHECK(json.find("\"cat\":\"sgl.gpu\",\"name\":\"submitted_after_discard_gpu_zone\"") != std::string::npos);

        ref<ProfilerStats> stats = profiler->stats_snapshot();
        for (const ProfilerStatsNode& node : stats->nodes())
            CHECK(node.pending_gpu_sample_count == 0);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler keeps gpu zones ticked before submit until command submit")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());
        ref<CommandEncoder> encoder = device->create_command_encoder();

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("late_submit_gpu_zone", encoder.get());
        }

        profiler->tick();
        profiler->flush();

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        device->wait_for_submit(submit_id);
        profiler->tick();
        profiler->flush();

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(trace_has_name(*trace, "late_submit_gpu_zone"));

        const std::filesystem::path path = testing::get_case_temp_directory() / "late-submit-gpu-zone.json";
        trace->write_to_json(path);
        const std::string json = read_text_file(path);
        CHECK(json.find("\"cat\":\"sgl.cpu\"") != std::string::npos);
        CHECK(json.find("\"cat\":\"sgl.gpu\",\"name\":\"late_submit_gpu_zone\"") != std::string::npos);

        ref<ProfilerStats> stats = profiler->stats_snapshot();
        const ProfilerStatsNode* node = find_stats_node(*stats, "late_submit_gpu_zone");
        REQUIRE(node);
        CHECK(node->gpu.valid);
        CHECK(node->pending_gpu_sample_count == 0);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler resolves gpu hierarchy per command encoder")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());
        ref<CommandEncoder> outer_encoder = device->create_command_encoder();
        ref<CommandEncoder> inner_encoder = device->create_command_encoder();

        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("outer_stream_gpu_zone", outer_encoder.get());
            {
                SGL_PROFILE_SCOPE("inner_stream_gpu_zone", inner_encoder.get());
            }
        }

        std::vector<ref<CommandBuffer>> command_buffers;
        std::vector<CommandBuffer*> command_buffer_ptrs;
        command_buffers.push_back(outer_encoder->finish());
        command_buffer_ptrs.push_back(command_buffers.back().get());
        command_buffers.push_back(inner_encoder->finish());
        command_buffer_ptrs.push_back(command_buffers.back().get());

        const uint64_t submit_id = device->submit_command_buffers(command_buffer_ptrs);
        device->wait_for_submit(submit_id);

        (void)wait_for_trace_zones(*profiler, "outer_stream_gpu_zone", ProfilerTimelineType::gpu, 1);
        ref<ProfilerTrace> trace
            = wait_for_trace_zones(*profiler, "inner_stream_gpu_zone", ProfilerTimelineType::gpu, 1);

        const ProfilerZoneRecord* outer_gpu
            = find_trace_zone(*trace, "outer_stream_gpu_zone", ProfilerTimelineType::gpu);
        const ProfilerZoneRecord* inner_gpu
            = find_trace_zone(*trace, "inner_stream_gpu_zone", ProfilerTimelineType::gpu);
        REQUIRE(outer_gpu);
        REQUIRE(inner_gpu);
        CHECK(outer_gpu->child_index_count == 0);
        CHECK(inner_gpu->child_index_count == 0);

        bool outer_gpu_root = false;
        bool inner_gpu_root = false;
        for (uint32_t root_index : trace->root_indices()) {
            if (root_index >= trace->zones().size())
                continue;
            const ProfilerZoneRecord& zone = trace->zones()[root_index];
            if (zone.timeline_id >= trace->timelines().size()
                || trace->timelines()[zone.timeline_id].type != ProfilerTimelineType::gpu)
                continue;
            if (zone.name_id < trace->names().size() && trace->names()[zone.name_id].name == "outer_stream_gpu_zone")
                outer_gpu_root = true;
            if (zone.name_id < trace->names().size() && trace->names()[zone.name_id].name == "inner_stream_gpu_zone")
                inner_gpu_root = true;
        }
        CHECK(outer_gpu_root);
        CHECK(inner_gpu_root);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler drops gpu state when device closes before profiler")
{
    Device* fixture_device = ctx.device;
    if (!fixture_device->has_feature(Feature::timestamp_query)
        || !fixture_device->has_feature(Feature::timestamp_calibration))
        return;

    {
        DeviceDesc desc = fixture_device->desc();
        desc.type = fixture_device->type();
        desc.adapter_luid = fixture_device->info().adapter_luid;
        desc.enable_hot_reload = false;
        desc.label = "profiler_close_callback_test_device";

        ref<Device> device = make_ref<Device>(desc);
        if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration)) {
            device->close();
            return;
        }

        ref<Profiler> profiler = make_ref<Profiler>();
        profiler->start_trace();
        ProfilerScope activation(profiler.get());

        ref<CommandEncoder> encoder = device->create_command_encoder();
        {
            SGL_PROFILE_FRAME();
            SGL_PROFILE_SCOPE("closed_device_gpu_zone", encoder.get());
        }

        device->submit_command_buffer(encoder->finish());
        device->close();

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        CHECK(trace_has_name(*trace, "closed_device_gpu_zone"));

        const std::filesystem::path path = testing::get_case_temp_directory() / "closed-device-gpu-zones.json";
        trace->write_to_json(path);
        const std::string json = read_text_file(path);
        CHECK(json.find("\"cat\":\"sgl.cpu\"") != std::string::npos);
        CHECK(json.find("\"cat\":\"sgl.gpu\"") == std::string::npos);

        ref<ProfilerStats> stats = profiler->stats_snapshot();
        for (const ProfilerStatsNode& node : stats->nodes())
            CHECK(node.pending_gpu_sample_count == 0);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_SUITE_END();
