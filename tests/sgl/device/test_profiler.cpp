// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/profiler.h"

#include <thread>

using namespace sgl;

TEST_SUITE_BEGIN("device");

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

TEST_SUITE_END();
