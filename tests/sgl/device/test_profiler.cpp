// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/command.h"
#include "sgl/device/device.h"
#include "sgl/device/profiler.h"

#include <fstream>
#include <iterator>
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

} // namespace

TEST_CASE("profiler static zone macro is callable")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();
        CHECK(current_profiler() == profiler.get());

        {
            SGL_PROFILER_ZONE("native_static_zone");
        }

        profiler->tick();
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler zone macro supports explicit interned and dynamic names")
{
    CommandEncoder* encoder = nullptr;
    const char* name = Profiler::intern_name("native_interned_zone");
    std::string dynamic_name = "native_dynamic_zone";

    REQUIRE(name != nullptr);
    CHECK(name == Profiler::intern_name("native_interned_zone"));
    CHECK(std::string_view(name) == "native_interned_zone");

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        CHECK(current_profiler() == profiler.get());

        SGL_PROFILER_ZONE(name);
        SGL_PROFILER_ZONE(name, encoder);
        SGL_PROFILER_ZONE(dynamic_name.c_str(), nullptr, ProfilerZoneFlags::copy_name);
        SGL_PROFILER_ZONE(dynamic_name.c_str(), encoder, ProfilerZoneFlags::copy_name);

        profiler->tick();
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("current profiler free functions are LIFO")
{
    {
        ref<Profiler> profiler_a = make_ref<Profiler>();
        Profiler* profiler_a_ptr = profiler_a.get();
        CHECK(current_profiler() == profiler_a_ptr);

        {
            ref<Profiler> profiler_b = make_ref<Profiler>();
            Profiler* profiler_b_ptr = profiler_b.get();
            CHECK(current_profiler() == profiler_b_ptr);

            push_current_profiler(profiler_a_ptr);
            CHECK(current_profiler() == profiler_a_ptr);

            push_current_profiler(profiler_b_ptr);
            CHECK(current_profiler() == profiler_b_ptr);
            CHECK(pop_current_profiler() == profiler_b_ptr);
            CHECK(current_profiler() == profiler_a_ptr);
            CHECK(pop_current_profiler() == profiler_a_ptr);
            CHECK(current_profiler() == profiler_b_ptr);
        }

        CHECK(current_profiler() == profiler_a_ptr);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler destruction removes stack entries")
{
    {
        ref<Profiler> profiler = make_ref<Profiler>();
        Profiler* profiler_ptr = profiler.get();

        CHECK(current_profiler() == profiler_ptr);
        push_current_profiler(profiler_ptr);
        CHECK(current_profiler() == profiler_ptr);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("current profiler is application-wide")
{
    bool worker_saw_profiler = false;

    {
        ref<Profiler> profiler = make_ref<Profiler>();
        CHECK(current_profiler() == profiler.get());

        std::thread thread(
            [&]()
            {
                worker_saw_profiler = current_profiler_or_null() == profiler.get();
                SGL_PROFILER_ZONE("worker_zone");
            }
        );
        thread.join();

        profiler->tick();
        CHECK(current_profiler() == profiler.get());
    }

    CHECK(worker_saw_profiler);
    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler scope is movable")
{
    {
        ref<Profiler> profiler_a = make_ref<Profiler>();
        ref<Profiler> profiler_b = make_ref<Profiler>();
        CHECK(current_profiler() == profiler_b.get());

        {
            ProfilerScope scope_a(profiler_a.get());
            CHECK(current_profiler() == profiler_a.get());

            ProfilerScope scope_b(profiler_b.get());
            CHECK(current_profiler() == profiler_b.get());

            scope_b = std::move(scope_a);
            CHECK(current_profiler() == profiler_a.get());
        }

        CHECK(current_profiler() == profiler_b.get());
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

        CHECK(&profiler->desc() != nullptr);
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
        ref<CommandEncoder> encoder = device->create_command_encoder();

        {
            SGL_PROFILER_ZONE("gpu_zone", encoder.get());
        }

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        profiler->tick();
        device->wait_for_submit(submit_id);
        profiler->tick();

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        const std::filesystem::path path = testing::get_case_temp_directory() / "gpu-zones.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(json.find("\"cat\":\"sgl.cpu\"") != std::string::npos);
        CHECK(json.find("\"cat\":\"sgl.gpu\"") != std::string::npos);
        CHECK(count_occurrences(json, "\"name\":\"gpu_zone\"") >= 2);
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
        std::vector<ref<CommandBuffer>> command_buffers;
        std::vector<CommandBuffer*> command_buffer_ptrs;
        command_buffers.reserve(command_buffer_count);
        command_buffer_ptrs.reserve(command_buffer_count);

        for (size_t i = 0; i < command_buffer_count; ++i) {
            ref<CommandEncoder> encoder = device->create_command_encoder();
            {
                SGL_PROFILER_ZONE("multi_encoder_gpu_zone", encoder.get());
            }

            ref<CommandBuffer> command_buffer = encoder->finish();
            command_buffer_ptrs.push_back(command_buffer.get());
            command_buffers.push_back(std::move(command_buffer));
        }

        const uint64_t submit_id = device->submit_command_buffers(command_buffer_ptrs);
        device->wait_for_submit(submit_id);
        profiler->tick();

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        const std::filesystem::path path = testing::get_case_temp_directory() / "multi-buffer-gpu-zones.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(count_occurrences(json, "\"name\":\"multi_encoder_gpu_zone\"") >= command_buffer_count * 2);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE_GPU("profiler records gpu zones across query blocks")
{
    Device* device = ctx.device;
    if (!device->has_feature(Feature::timestamp_query) || !device->has_feature(Feature::timestamp_calibration))
        return;

    {
        static constexpr size_t zone_count = 160;

        ref<Profiler> profiler = make_ref<Profiler>();
        ref<CommandEncoder> encoder = device->create_command_encoder();

        for (size_t i = 0; i < zone_count; ++i) {
            SGL_PROFILER_ZONE("many_gpu_zones", encoder.get());
        }

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        device->wait_for_submit(submit_id);
        profiler->tick();

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        const std::filesystem::path path = testing::get_case_temp_directory() / "many-gpu-zones.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(count_occurrences(json, "\"name\":\"many_gpu_zones\"") >= zone_count * 2);
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

        {
            ref<CommandEncoder> discarded_encoder = device->create_command_encoder();
            {
                SGL_PROFILER_ZONE("discarded_gpu_zone", discarded_encoder.get());
            }
        }

        ref<CommandEncoder> encoder = device->create_command_encoder();
        {
            SGL_PROFILER_ZONE("submitted_after_discard_gpu_zone", encoder.get());
        }

        const uint64_t submit_id = device->submit_command_buffer(encoder->finish());
        device->wait_for_submit(submit_id);
        profiler->tick();

        ref<ProfilerTrace> trace = profiler->trace_snapshot();
        const std::filesystem::path path = testing::get_case_temp_directory() / "discarded-gpu-zones.json";
        trace->write_to_json(path);

        const std::string json = read_text_file(path);
        CHECK(json.find("\"cat\":\"sgl.gpu\",\"name\":\"discarded_gpu_zone\"") == std::string::npos);
        CHECK(json.find("\"cat\":\"sgl.gpu\",\"name\":\"submitted_after_discard_gpu_zone\"") != std::string::npos);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_SUITE_END();
