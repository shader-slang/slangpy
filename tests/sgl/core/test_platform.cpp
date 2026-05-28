// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/platform.h"

using namespace sgl::platform;

TEST_SUITE_BEGIN("platform");

TEST_CASE("is_same_path")
{
    CHECK(is_same_path("foo", "foo"));
    CHECK(is_same_path("foo/", "foo/."));
    CHECK(is_same_path("foo/", "foo/./"));
    CHECK(is_same_path("foo/", "foo/./."));
    CHECK(is_same_path("foo", "./foo"));
    CHECK(is_same_path("foo/", "./foo/."));
    CHECK(is_same_path("foo/", "./foo/./"));
    CHECK(is_same_path("foo/", "./foo/./."));

    CHECK_FALSE(is_same_path("foo", "bar"));
}

TEST_CASE("has_extension")
{
    CHECK_EQ(has_extension("foo.exr", "exr"), true);
    CHECK_EQ(has_extension("foo.exr", ".exr"), true);
    CHECK_EQ(has_extension("foo.Exr", "exr"), true);
    CHECK_EQ(has_extension("foo.Exr", ".exr"), true);
    CHECK_EQ(has_extension("foo.Exr", "exR"), true);
    CHECK_EQ(has_extension("foo.Exr", ".exR"), true);
    CHECK_EQ(has_extension("foo.EXR", "exr"), true);
    CHECK_EQ(has_extension("foo.EXR", ".exr"), true);
    CHECK_EQ(has_extension("foo.xr", "exr"), false);
    CHECK_EQ(has_extension("/foo/png", ""), true);
    CHECK_EQ(has_extension("/foo/png", "exr"), false);
    CHECK_EQ(has_extension("/foo/.profile", ""), true);
}

TEST_CASE("get_extension_from_path")
{
    CHECK_EQ(get_extension_from_path("foo.exr"), "exr");
    CHECK_EQ(get_extension_from_path("foo.Exr"), "exr");
    CHECK_EQ(get_extension_from_path("foo.EXR"), "exr");
    CHECK_EQ(get_extension_from_path("foo"), "");
    CHECK_EQ(get_extension_from_path("/foo/.profile"), "");
}

TEST_CASE("junction")
{
    std::filesystem::path path = sgl::testing::get_case_temp_directory();
    std::filesystem::path target = path / "junction_target";
    std::filesystem::path link = path / "junction_link";

    // Create junction_target/test
    std::filesystem::create_directories(target / "test");

    // Create junction from junction_link to junction_target
    CHECK(create_junction(link, target));

    // Check that junction was successfully created by accessing junction_link/test
    CHECK(std::filesystem::exists(link / "test"));

    // Delete junction
    CHECK(delete_junction(link));

    // Check that junction was deleted
    CHECK_FALSE(std::filesystem::exists(link));

    // Delete junction_target/test
    std::filesystem::remove_all(target);
}

TEST_CASE("environment")
{
    auto path = get_environment_variable("PATH");
    CHECK(path.has_value());
    MESSAGE("PATH:", path.value());
}

TEST_CASE("backtrace")
{
    StackTrace trace = backtrace();
    CHECK_GT(trace.size(), 0);

    ResolvedStackTrace resolved_trace = resolve_stacktrace(trace);
    CHECK_EQ(resolved_trace.size(), trace.size());

    std::string formatted_trace = format_stacktrace(resolved_trace);
    CHECK_GT(formatted_trace.size(), 0);
}

TEST_SUITE_END();
