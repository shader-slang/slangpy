// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/profiler.h"

using namespace sgl;

TEST_SUITE_BEGIN("device");

TEST_CASE("profiler static zone macro is callable")
{
    ref<Profiler> profiler = make_ref<Profiler>();

    {
        ProfilerScope scope(profiler.get());
        {
            SGL_PROFILER_ZONE("native_static_zone");
        }
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler zone macro supports optional name encoder and flags")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    CommandEncoder* encoder = nullptr;
    ProfilerZoneFlags flags = ProfilerZoneFlags::cpu | ProfilerZoneFlags::gpu;

    {
        ProfilerScope scope(profiler.get());

        SGL_PROFILER_ZONE();
        SGL_PROFILER_ZONE("native_literal_zone");
        SGL_PROFILER_ZONE(nullptr, encoder);
        SGL_PROFILER_ZONE("native_literal_gpu_zone", encoder);
        SGL_PROFILER_ZONE(nullptr, nullptr, flags);
        SGL_PROFILER_ZONE("native_literal_flagged_zone", nullptr, flags);
        SGL_PROFILER_ZONE(nullptr, encoder, flags);
        SGL_PROFILER_ZONE("native_literal_gpu_flagged_zone", encoder, flags);
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler zone macro supports explicit interned and dynamic names")
{
    ref<Profiler> profiler = make_ref<Profiler>();
    CommandEncoder* encoder = nullptr;
    const char* name = Profiler::intern_name("native_interned_zone");
    std::string dynamic_name = "native_dynamic_zone";

    REQUIRE(name != nullptr);
    CHECK(name == Profiler::intern_name("native_interned_zone"));
    CHECK(std::string_view(name) == "native_interned_zone");

    {
        ProfilerScope scope(profiler.get());

        SGL_PROFILER_ZONE(name);
        SGL_PROFILER_ZONE(name, encoder);
        SGL_PROFILER_ZONE(name, nullptr, ProfilerZoneFlags::cpu);
        SGL_PROFILER_ZONE(name, encoder, ProfilerZoneFlags::cpu | ProfilerZoneFlags::gpu);
        SGL_PROFILER_ZONE(dynamic_name.c_str(), nullptr, ProfilerZoneFlags::copy_name);
        SGL_PROFILER_ZONE(dynamic_name.c_str(), encoder, ProfilerZoneFlags::copy_name);
        SGL_PROFILER_ZONE(dynamic_name.c_str(), nullptr, ProfilerZoneFlags::copy_name | ProfilerZoneFlags::cpu);
        SGL_PROFILER_ZONE(
            dynamic_name.c_str(),
            encoder,
            ProfilerZoneFlags::copy_name | ProfilerZoneFlags::cpu | ProfilerZoneFlags::gpu
        );
    }

    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("current profiler free functions are LIFO")
{
    ref<Profiler> profiler_a = make_ref<Profiler>();
    ref<Profiler> profiler_b = make_ref<Profiler>();

    push_current_profiler(profiler_a.get());
    CHECK(current_profiler() == profiler_a.get());

    push_current_profiler(profiler_b.get());
    CHECK(current_profiler() == profiler_b.get());
    CHECK(pop_current_profiler() == profiler_b.get());
    CHECK(current_profiler() == profiler_a.get());
    CHECK(pop_current_profiler() == profiler_a.get());
    CHECK(current_profiler_or_null() == nullptr);
}

TEST_CASE("profiler scope is movable")
{
    ref<Profiler> profiler_a = make_ref<Profiler>();
    ref<Profiler> profiler_b = make_ref<Profiler>();

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

TEST_SUITE_END();
