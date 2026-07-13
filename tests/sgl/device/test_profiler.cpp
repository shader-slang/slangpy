// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/core/platform.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/utils/profiler.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <thread>

using namespace sgl;

TEST_SUITE_BEGIN("profiler");

namespace {

void record_nested()
{
    SGL_PROFILE_FRAME("frame");
    SGL_PROFILE_ZONE("outer");
    {
        SGL_PROFILE_ZONE("inner");
    }
}

void record_native_function_site()
{
    SGL_PROFILE_FUNCTION();
}

bool supports_gpu_timestamps(Device* device)
{
    return device->has_feature(Feature::timestamp_query) && device->has_feature(Feature::timestamp_calibration);
}

} // namespace

TEST_CASE("descriptor validation")
{
    ProfilerDesc invalid_desc;
    invalid_desc.thread_event_capacity = 3;
    CHECK_THROWS_WITH_AS(Profiler{invalid_desc}, doctest::Contains("power of two"), std::runtime_error);
    invalid_desc.thread_event_capacity = 8;
    invalid_desc.frame_stats_window_size = 0;
    CHECK_THROWS_WITH_AS(Profiler{invalid_desc}, doctest::Contains("frame_stats_window_size"), std::runtime_error);
    invalid_desc.frame_stats_window_size = 120;
    invalid_desc.gpu_query_pool_size = 3;
    CHECK_THROWS_WITH_AS(Profiler{invalid_desc}, doctest::Contains("even"), std::runtime_error);
}

TEST_CASE("current profiler stack is nested and thread local")
{
    CHECK(current_profiler_or_null() == nullptr);
    ref<Profiler> profiler = make_ref<Profiler>();
    CHECK(current_profiler() == profiler);
    ref<Profiler> secondary = make_ref<Profiler>();
    CHECK(current_profiler() == profiler);
    {
        ProfilerScope outer(secondary);
        CHECK(current_profiler() == secondary);
        {
            ProfilerScope inner(secondary);
            CHECK(current_profiler() == secondary);
        }
        Profiler* worker_current = profiler;
        std::thread worker(
            [&]
            {
                worker_current = current_profiler_or_null();
            }
        );
        worker.join();
        CHECK(worker_current == nullptr);
    }
    CHECK(current_profiler() == profiler);
    secondary = nullptr;
    CHECK(current_profiler() == profiler);
    profiler = nullptr;
    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("site metadata is interned and native function names are compact")
{
    std::string file = "shared-profiler-source.cpp";
    std::string function = "public: static void __cdecl example::Widget::render(class sgl::CommandEncoder *)";
    std::string name = function;
    const uint32_t function_site = Profiler::register_site(file, 10, function, name);
    const uint32_t zone_site = Profiler::register_site(file, 11, function, "render zone");
    CHECK(Profiler::register_site(file, 10, function, name) == function_site);

    file.assign("replaced file");
    function.assign("replaced function");
    name.assign("replaced name");

    ref<Profiler> profiler = make_ref<Profiler>();
    profiler->start_capture();
    profiler->end_zone(profiler->begin_zone(function_site));
    profiler->end_zone(profiler->begin_zone(zone_site));
    record_native_function_site();
    ref<ProfilerTrace> trace = profiler->stop_capture();

    const ProfilerSite& function_metadata = trace->sites().at(function_site - 1);
    const ProfilerSite& zone_metadata = trace->sites().at(zone_site - 1);
    CHECK(function_metadata.file == "shared-profiler-source.cpp");
    CHECK(function_metadata.function == "example::Widget::render");
    CHECK(function_metadata.name == "example::Widget::render");
    CHECK(zone_metadata.name == "render zone");
    CHECK(function_metadata.file.data() == zone_metadata.file.data());
    CHECK(function_metadata.function.data() == zone_metadata.function.data());
    const auto native_function = std::find_if(
        trace->sites().begin(),
        trace->sites().end(),
        [](const ProfilerSite& site)
        {
            return site.name.ends_with("record_native_function_site");
        }
    );
    REQUIRE(native_function != trace->sites().end());
    CHECK(native_function->name == native_function->function);
    CHECK(native_function->name.find('(') == std::string_view::npos);

    const uint32_t gcc_site = Profiler::register_site(
        "shared-profiler-source.cpp",
        12,
        "void example::Widget::update(int) [with T = float]",
        "update"
    );
    const uint32_t apple_clang_site = Profiler::register_site(
        "shared-profiler-source.cpp",
        13,
        "void (anonymous namespace)::example::Widget::update(int)",
        "update"
    );
    profiler->start_capture();
    profiler->end_zone(profiler->begin_zone(gcc_site));
    profiler->end_zone(profiler->begin_zone(apple_clang_site));
    trace = profiler->stop_capture();
    CHECK(trace->sites().at(gcc_site - 1).function == "example::Widget::update");
    CHECK(trace->sites().at(apple_clang_site - 1).function == "example::Widget::update");
}

TEST_CASE("capture records hierarchy frames queries and immutable chunks")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    profiler->start_capture();
    record_nested();
    ref<ProfilerTrace> trace = profiler->stop_capture();
    REQUIRE(trace->zone_count() == 2);
    REQUIRE(trace->frames().size() == 1);
    REQUIRE(trace->zone_chunks().size() == 1);
    const auto& chunk = trace->zone_chunks().front();
    CHECK(chunk->parent_index()[0] == 1);
    CHECK(chunk->parent_index()[1] == -1);
    CHECK(chunk->frame_index().front() != ~uint32_t{0});
    REQUIRE(trace->timelines().size() == 1);
    CHECK(trace->timelines().front().thread_id == platform::current_thread_id());

    ref<ProfilerZoneSelection> selection = trace->query_zones("inner");
    CHECK(selection->count() == 1);
    CHECK(selection->statistics().count == 1);

    const uint64_t original_count = trace->zone_count();
    {
        SGL_PROFILE_ZONE("later");
    }
    profiler->flush();
    CHECK(trace->zone_count() == original_count);
    CHECK(profiler->live_snapshot()->zone_count() == 3);
}

TEST_CASE("disabled profiler does not record")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    const uint32_t site = Profiler::register_site(__FILE__, __LINE__, __func__, "deep");
    profiler->start_capture();
    profiler->set_enabled(false);
    CHECK(!profiler->begin_zone(site).profiler);
    CHECK(profiler->stop_capture()->zone_count() == 0);
}

TEST_CASE("zone stack overflow drops only the excess zone")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    const uint32_t site = Profiler::register_site(__FILE__, __LINE__, __func__, "deep");
    profiler->start_capture();
    std::vector<ProfilerZoneToken> tokens;
    for (uint32_t i = 0; i < 65; ++i)
        tokens.push_back(profiler->begin_zone(site));
    CHECK(!tokens.back().profiler);
    for (auto it = tokens.rbegin(); it != tokens.rend(); ++it)
        profiler->end_zone(*it);
    ref<ProfilerTrace> trace = profiler->stop_capture();
    CHECK(trace->zone_count() == 64);
    CHECK(trace->diagnostics().producer_drop_count == 1);
}

TEST_CASE("capture collects concurrent producers without drops")
{
    ProfilerDesc desc;
    desc.thread_event_capacity = 1024;
    desc.live_event_capacity = 10000;
    ref<Profiler> profiler = make_ref<Profiler>(desc);
    profiler->start_capture();
    std::vector<std::thread> workers;
    for (uint32_t thread_index = 0; thread_index < 4; ++thread_index) {
        workers.emplace_back(
            [&]
            {
                ProfilerScope scope(profiler);
                for (uint32_t i = 0; i < 200; ++i) {
                    SGL_PROFILE_ZONE("parallel");
                }
            }
        );
    }
    for (std::thread& worker : workers)
        worker.join();
    ref<ProfilerTrace> trace = profiler->stop_capture();
    CHECK(trace->zone_count() == 800);
    CHECK(trace->diagnostics().producer_drop_count == 0);
    CHECK(trace->timelines().size() == 4);
}

TEST_CASE("frame statistics align repeated and intermittent zones")
{
    ProfilerDesc desc;
    desc.frame_stats_window_size = 2;
    ref<Profiler> profiler = make_ref<Profiler>(desc);
    const uint32_t frame_site = Profiler::register_site(__FILE__, __LINE__, __func__, "stats frame");
    const uint32_t outer_site = Profiler::register_site(__FILE__, __LINE__, __func__, "stats outer");
    const uint32_t inner_site = Profiler::register_site(__FILE__, __LINE__, __func__, "stats inner");

    auto record_frame = [&](uint32_t child_count)
    {
        ProfilerFrameToken frame = profiler->begin_frame(frame_site);
        ProfilerZoneToken outer = profiler->begin_zone(outer_site);
        for (uint32_t i = 0; i < child_count; ++i)
            profiler->end_zone(profiler->begin_zone(inner_site));
        profiler->end_zone(outer);
        profiler->end_frame(frame);
    };

    record_frame(4);
    record_frame(0);
    profiler->flush();

    ref<ProfilerFrameStats> first_snapshot = profiler->frame_stats_snapshot();
    const ProfilerFrameStats& first = *first_snapshot;
    CHECK(first.sample_count() == 2);
    REQUIRE(first.entries().size() == 2);
    const ProfilerFrameStatsEntry& outer = first.entries()[0];
    const ProfilerFrameStatsEntry& inner = first.entries()[1];
    CHECK(outer.name == "stats outer");
    CHECK(inner.name == "stats inner");
    CHECK(inner.parent_index == 0);
    CHECK(inner.cpu_time_per_frame.count == 2);
    CHECK(inner.cpu_time_per_call.count == 4);
    CHECK(inner.gpu_time_per_frame.count == 0);
    CHECK(first.diagnostics().producer_drop_count == 0);
    CHECK(first.sample_frame_index().size() == 2);
    CHECK(first.sample_frame_index()[0] < first.sample_frame_index()[1]);
    CHECK(first.sample_call_count().size() == first.sample_count() * first.entry_count());
    CHECK(first.sample_cpu_time_ms().size() == first.sample_count() * first.entry_count());
    CHECK(first.sample_gpu_time_ms().size() == first.sample_count() * first.entry_count());
    CHECK(first.sample_gpu_status().size() == first.sample_count() * first.entry_count());
    const ProfilerFrameStatsSampleView first_sample = first.sample(0);
    const ProfilerFrameStatsSampleView second_sample = first.sample(1);
    CHECK(first_sample.call_count[1] == 4);
    CHECK(second_sample.call_count[1] == 0);
    CHECK(first_sample.cpu_time_ms[1] >= 0.0);
    CHECK(second_sample.cpu_time_ms[1] == 0.0);
    CHECK(first_sample.gpu_status[1] == ProfilerFrameGpuStatus::unavailable);
    CHECK(second_sample.gpu_status[1] == ProfilerFrameGpuStatus::absent);

    record_frame(2);
    profiler->flush();
    ref<ProfilerFrameStats> second_snapshot = profiler->frame_stats_snapshot();
    CHECK(first_snapshot->entries()[1].cpu_time_per_call.count == 4);
    CHECK(second_snapshot->sample_count() == 2);
    CHECK(second_snapshot->entries()[1].cpu_time_per_call.count == 2);
    CHECK(second_snapshot->sample(0).frame_index == first.sample(1).frame_index);
    CHECK(second_snapshot->sample(0).call_count[1] == 0);
    CHECK(second_snapshot->sample(1).call_count[1] == 2);
    CHECK(first_snapshot->sample(0).call_count[1] == 4);

    const std::string text = second_snapshot->to_string();
    CHECK(text.find("ProfilerFrameStats(pending_frame_count=0") != std::string::npos);
    CHECK(text.find("sample_count=2") != std::string::npos);
    CHECK(text.find("stats inner: calls/frame mean=1") != std::string::npos);

    profiler->clear_frame_stats();
    CHECK(profiler->frame_stats_snapshot()->sample_count() == 0);
    CHECK(first_snapshot->sample(0).call_count[1] == 4);
}

TEST_CASE("frame statistics hierarchy ordering is deterministic preorder")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    const uint32_t frame_site = Profiler::register_site(__FILE__, __LINE__, __func__, "ordered frame");
    const uint32_t root_b_site = Profiler::register_site(__FILE__, __LINE__, __func__, "root b");
    const uint32_t root_a_site = Profiler::register_site(__FILE__, __LINE__, __func__, "root a");
    const uint32_t child_z_site = Profiler::register_site(__FILE__, __LINE__, __func__, "child z");
    const uint32_t child_a_site = Profiler::register_site(__FILE__, __LINE__, __func__, "child a");

    ProfilerFrameToken frame = profiler->begin_frame(frame_site);
    ProfilerZoneToken root_b = profiler->begin_zone(root_b_site);
    profiler->end_zone(profiler->begin_zone(child_z_site));
    profiler->end_zone(profiler->begin_zone(child_a_site));
    profiler->end_zone(root_b);
    profiler->end_zone(profiler->begin_zone(root_a_site));
    profiler->end_frame(frame);
    profiler->flush();

    ref<ProfilerFrameStats> stats = profiler->frame_stats_snapshot();
    const ProfilerFrameStats& ordered = *stats;
    REQUIRE(ordered.entries().size() == 4);
    CHECK(ordered.entries()[0].name == "root a");
    CHECK(ordered.entries()[0].parent_index == -1);
    CHECK(ordered.entries()[1].name == "root b");
    CHECK(ordered.entries()[1].parent_index == -1);
    CHECK(ordered.entries()[2].name == "child a");
    CHECK(ordered.entries()[2].parent_index == 1);
    CHECK(ordered.entries()[3].name == "child z");
    CHECK(ordered.entries()[3].parent_index == 1);
}

TEST_CASE("global frames include zones from other threads and exclude unframed zones")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    const uint32_t frame_a = Profiler::register_site(__FILE__, __LINE__, __func__, "frame a");
    const uint32_t frame_b = Profiler::register_site(__FILE__, __LINE__, __func__, "frame b");
    const uint32_t zone = Profiler::register_site(__FILE__, __LINE__, __func__, "framed zone");
    profiler->end_zone(profiler->begin_zone(zone));

    ProfilerFrameToken frame = profiler->begin_frame(frame_a);
    REQUIRE(frame.profiler == profiler);
    std::atomic<bool> worker_started{false};
    std::atomic<bool> finish_worker{false};
    std::atomic<bool> worker_frame_rejected{false};
    std::thread worker(
        [&]
        {
            worker_frame_rejected.store(!profiler->begin_frame(frame_b).profiler, std::memory_order_relaxed);
            ProfilerZoneToken token = profiler->begin_zone(zone);
            worker_started.store(true, std::memory_order_release);
            while (!finish_worker.load(std::memory_order_acquire))
                std::this_thread::yield();
            profiler->end_zone(token);
        }
    );
    while (!worker_started.load(std::memory_order_acquire))
        std::this_thread::yield();
    CHECK(worker_frame_rejected.load(std::memory_order_relaxed));
    profiler->end_frame(frame);
    CHECK(!profiler->begin_frame(frame_b).profiler);
    finish_worker.store(true, std::memory_order_release);
    worker.join();

    frame = profiler->begin_frame(frame_b);
    REQUIRE(frame.profiler == profiler);
    profiler->end_zone(profiler->begin_zone(zone));
    profiler->end_frame(frame);
    profiler->flush();
    ref<ProfilerFrameStats> stats = profiler->frame_stats_snapshot();
    CHECK(stats->sample_count() == 2);
    REQUIRE(stats->entries().size() == 1);
    CHECK(stats->entries().front().cpu_time_per_call.count == 2);
    CHECK(stats->sample(0).call_count.front() == 1);
    CHECK(stats->sample(1).call_count.front() == 1);
}

TEST_CASE("capture stops at its memory bound")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    profiler->start_capture({64});
    const uint32_t site = Profiler::register_site(__FILE__, __LINE__, __func__, "bounded");
    for (uint32_t i = 0; i < 32; ++i) {
        ProfilerZoneToken token = profiler->begin_zone(site);
        profiler->end_zone(token);
    }
    profiler->flush();
    CHECK(!profiler->capture_active());
    ref<ProfilerTrace> trace = profiler->stop_capture();
    CHECK(trace->truncated());
    CHECK(trace->stop_reason() == ProfilerCaptureStopReason::memory_limit);
    CHECK(trace->memory_bytes() <= 64);
}

TEST_CASE("trace JSON is streamed with escaping")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    profiler->start_capture();
    const uint32_t site = Profiler::register_site(__FILE__, __LINE__, __func__, "json \"zone\"");
    profiler->end_zone(profiler->begin_zone(site));
    ref<ProfilerTrace> trace = profiler->stop_capture();
    const auto path = testing::get_case_temp_directory() / "trace.json";
    trace->write_to_json(path);
    std::ifstream stream(path);
    const std::string contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    CHECK(contents.find("json \\\"zone\\\"") != std::string::npos);
    CHECK(contents.find("\"ph\":\"X\"") != std::string::npos);
}

TEST_CASE_GPU("GPU timestamps follow command recording submission")
{
    if (!supports_gpu_timestamps(ctx.device))
        return;

    ref<Profiler> profiler = make_ref<Profiler>();
    profiler->start_capture();
    ref<CommandEncoder> encoder = ctx.device->create_command_encoder();
    const uint32_t frame_site = Profiler::register_site(__FILE__, __LINE__, __func__, "gpu frame");
    const uint32_t site = Profiler::register_site(__FILE__, __LINE__, __func__, "gpu");
    ProfilerFrameToken frame = profiler->begin_frame(frame_site);
    ProfilerZoneToken token = profiler->begin_zone(site, encoder);
    profiler->end_zone(token);
    profiler->end_frame(frame);
    profiler->flush();
    ref<ProfilerFrameStats> pending_stats = profiler->frame_stats_snapshot();
    CHECK(pending_stats->pending_frame_count() == 1);
    CHECK(pending_stats->sample_count() == 0);
    profiler->tick();
    ctx.device->submit_command_buffer(encoder->finish());
    ctx.device->wait();
    profiler->tick();

    encoder = ctx.device->create_command_encoder();
    token = profiler->begin_zone(site, encoder);
    profiler->end_zone(token);
    ctx.device->submit_command_buffer(encoder->finish());
    ctx.device->wait();
    profiler->tick();

    std::thread later_cpu_thread(
        [&]
        {
            const uint32_t cpu_site = Profiler::register_site(__FILE__, __LINE__, __func__, "later cpu timeline");
            profiler->end_zone(profiler->begin_zone(cpu_site));
        }
    );
    later_cpu_thread.join();

    ref<ProfilerTrace> trace = profiler->stop_capture();
    CHECK(trace->query_zones("gpu", ProfilerTimelineType::cpu)->count() == 2);
    CHECK(trace->query_zones("gpu", ProfilerTimelineType::gpu)->count() == 2);
    CHECK(trace->query_zones("later cpu timeline", ProfilerTimelineType::cpu)->count() == 1);
    CHECK(trace->diagnostics().pending_gpu_zone_count == 0);
    REQUIRE(trace->timelines().size() == 3);
    CHECK(trace->timelines()[0].type == ProfilerTimelineType::cpu);
    CHECK(trace->timelines()[1].type == ProfilerTimelineType::gpu);
    CHECK(trace->timelines()[2].type == ProfilerTimelineType::cpu);
    ref<ProfilerFrameStats> stats = profiler->frame_stats_snapshot();
    CHECK(stats->sample_count() == 1);
    REQUIRE(stats->entries().size() == 1);
    CHECK(stats->entries().front().gpu_time_per_call.count == 1);
    CHECK(stats->entries().front().gpu_time_per_frame.count == 1);
    CHECK(stats->sample(0).gpu_status.front() == ProfilerFrameGpuStatus::complete);
}

TEST_CASE_GPU("GPU hierarchy is scoped to a command recording")
{
    if (!supports_gpu_timestamps(ctx.device))
        return;

    ref<Profiler> profiler = make_ref<Profiler>();
    profiler->start_capture();
    const uint32_t outer_site = Profiler::register_site(__FILE__, __LINE__, __func__, "gpu outer");
    const uint32_t inner_site = Profiler::register_site(__FILE__, __LINE__, __func__, "gpu inner");

    ref<CommandEncoder> encoder = ctx.device->create_command_encoder();
    ProfilerZoneToken outer = profiler->begin_zone(outer_site, encoder);
    ProfilerZoneToken inner = profiler->begin_zone(inner_site, encoder);
    profiler->end_zone(inner);
    profiler->end_zone(outer);
    ctx.device->submit_command_buffer(encoder->finish());

    ref<CommandEncoder> encoder_a = ctx.device->create_command_encoder();
    ref<CommandEncoder> encoder_b = ctx.device->create_command_encoder();
    outer = profiler->begin_zone(outer_site, encoder_a);
    inner = profiler->begin_zone(inner_site, encoder_b);
    profiler->end_zone(inner);
    profiler->end_zone(outer);
    ctx.device->submit_command_buffer(encoder_a->finish());
    ctx.device->submit_command_buffer(encoder_b->finish());

    ctx.device->wait();
    profiler->tick();
    ref<ProfilerTrace> trace = profiler->stop_capture();
    ref<ProfilerZoneSelection> gpu_zones = trace->query_zones({}, ProfilerTimelineType::gpu);
    REQUIRE(gpu_zones->count() == 4);
    uint32_t gpu_child_count = 0;
    for (uint32_t index : gpu_zones->indices()) {
        uint32_t base = 0;
        for (const auto& chunk : trace->zone_chunks()) {
            if (index < base + chunk->size()) {
                if (chunk->parent_index()[index - base] >= 0)
                    ++gpu_child_count;
                break;
            }
            base += uint32_t(chunk->size());
        }
    }
    CHECK(gpu_child_count == 1);
}

TEST_CASE_GPU("GPU query exhaustion preserves CPU zones")
{
    if (!supports_gpu_timestamps(ctx.device))
        return;

    ProfilerDesc desc;
    desc.gpu_query_pool_size = 2;
    ref<Profiler> profiler = make_ref<Profiler>(desc);
    profiler->start_capture();
    const uint32_t frame_site = Profiler::register_site(__FILE__, __LINE__, __func__, "exhausted frame");
    const uint32_t site = Profiler::register_site(__FILE__, __LINE__, __func__, "exhausted gpu");
    ref<CommandEncoder> encoder = ctx.device->create_command_encoder();
    ProfilerFrameToken frame = profiler->begin_frame(frame_site);
    ProfilerZoneToken outer = profiler->begin_zone(site, encoder);
    ProfilerZoneToken inner = profiler->begin_zone(site, encoder);
    profiler->end_zone(inner);
    profiler->end_zone(outer);
    profiler->end_frame(frame);
    ctx.device->submit_command_buffer(encoder->finish());
    ctx.device->wait();
    profiler->tick();
    ref<ProfilerTrace> trace = profiler->stop_capture();
    CHECK(trace->query_zones("exhausted gpu", ProfilerTimelineType::cpu)->count() == 2);
    CHECK(trace->query_zones("exhausted gpu", ProfilerTimelineType::gpu)->count() == 1);
    CHECK(trace->diagnostics().gpu_query_exhaustion_count == 1);
    ref<ProfilerFrameStats> stats = profiler->frame_stats_snapshot();
    REQUIRE(stats->entries().size() == 2);
    uint64_t gpu_call_count = 0;
    for (const ProfilerFrameStatsEntry& entry : stats->entries())
        gpu_call_count += entry.gpu_time_per_call.count;
    CHECK(gpu_call_count == 1);
    const ProfilerFrameStatsSampleView sample = stats->sample(0);
    REQUIRE(sample.gpu_status.size() == 2);
    CHECK(std::count(sample.gpu_status.begin(), sample.gpu_status.end(), ProfilerFrameGpuStatus::complete) == 1);
    CHECK(std::count(sample.gpu_status.begin(), sample.gpu_status.end(), ProfilerFrameGpuStatus::incomplete) == 1);
}

TEST_CASE_GPU("discarded GPU work and pending bounds complete frame statistics")
{
    if (!supports_gpu_timestamps(ctx.device))
        return;

    ProfilerDesc desc;
    desc.frame_stats_window_size = 1;
    ref<Profiler> profiler = make_ref<Profiler>(desc);
    const uint32_t frame_site = Profiler::register_site(__FILE__, __LINE__, __func__, "pending frame");
    const uint32_t zone_site = Profiler::register_site(__FILE__, __LINE__, __func__, "pending gpu");
    auto record_pending_frame = [&]()
    {
        ref<CommandEncoder> encoder = ctx.device->create_command_encoder();
        ProfilerFrameToken frame = profiler->begin_frame(frame_site);
        profiler->end_zone(profiler->begin_zone(zone_site, encoder));
        profiler->end_frame(frame);
        return encoder->finish();
    };

    ref<CommandBuffer> first = record_pending_frame();
    profiler->flush();
    CHECK(profiler->frame_stats_snapshot()->pending_frame_count() == 1);
    ref<CommandBuffer> second = record_pending_frame();
    profiler->flush();
    ref<ProfilerFrameStats> bounded = profiler->frame_stats_snapshot();
    CHECK(bounded->sample_count() == 1);
    CHECK(bounded->pending_frame_count() == 1);
    REQUIRE(bounded->entries().size() == 1);
    CHECK(bounded->sample(0).gpu_status.front() == ProfilerFrameGpuStatus::incomplete);

    first = nullptr;
    second = nullptr;
    profiler->flush();
    ref<ProfilerFrameStats> discarded = profiler->frame_stats_snapshot();
    CHECK(discarded->sample_count() == 1);
    CHECK(discarded->pending_frame_count() == 0);
    REQUIRE(discarded->entries().size() == 1);
    CHECK(discarded->sample(0).gpu_status.front() == ProfilerFrameGpuStatus::incomplete);

    ref<CommandBuffer> third = record_pending_frame();
    profiler->flush();
    CHECK(profiler->frame_stats_snapshot()->pending_frame_count() == 1);
    profiler->clear_frame_stats();
    CHECK(profiler->frame_stats_snapshot()->sample_count() == 0);
    third = nullptr;
    profiler->flush();
    CHECK(profiler->frame_stats_snapshot()->sample_count() == 0);
}

TEST_CASE_GPU("device close settles pending frame statistics")
{
    DeviceDesc device_desc = ctx.device->desc();
    device_desc.label = "profiler-device-close";
    ref<Device> device = Device::create(device_desc);
    if (!supports_gpu_timestamps(device)) {
        device->close();
        return;
    }

    ref<Profiler> profiler = make_ref<Profiler>();
    const uint32_t frame_site = Profiler::register_site(__FILE__, __LINE__, __func__, "device close frame");
    const uint32_t zone_site = Profiler::register_site(__FILE__, __LINE__, __func__, "device close gpu");
    ref<CommandEncoder> encoder = device->create_command_encoder();
    ProfilerFrameToken frame = profiler->begin_frame(frame_site);
    profiler->end_zone(profiler->begin_zone(zone_site, encoder));
    profiler->end_frame(frame);
    ref<CommandBuffer> command_buffer = encoder->finish();
    profiler->flush();
    CHECK(profiler->frame_stats_snapshot()->pending_frame_count() == 1);

    device->close();
    profiler->flush();
    ref<ProfilerFrameStats> stats = profiler->frame_stats_snapshot();
    CHECK(stats->pending_frame_count() == 0);
    REQUIRE(stats->entries().size() == 1);
    CHECK(stats->sample(0).gpu_status.front() == ProfilerFrameGpuStatus::incomplete);
    command_buffer = nullptr;
}

TEST_SUITE_END();
