// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/kernel.h"
#include "sgl/device/shader_coverage_internal.h"
#include <algorithm>
#include <fstream>
#include <filesystem>

using namespace sgl;

/// Setup code for the shader test writes out some simple modules.
static void setup_testshader_files(const std::filesystem::path& dir)
{
    // Use local static to ensure setup only occurs once.
    static bool is_done = false;
    if (!is_done) {
        is_done = true;
        // _testshader_simple.slang is a single compute shader + struct.
        {
            std::ofstream shader(dir / "_testshader_simple.slang");
            shader << R"SHADER(
struct Foo {
    uint a;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main_a(uint3 tid : SV_DispatchThreadID, uniform Foo foo)
{
}
)SHADER";
            shader.close();
        }

        // _testshader_struct.slang imports sgl.device.print and defines a struct.
        {
            std::ofstream shader(dir / "_testshader_struct.slang");
            shader << R"SHADER(
import sgl.device.print;
struct Foo {
    uint a;
};
)SHADER";
            shader.close();
        }

        // _testshader_dependent.slang is a compute shader dependent on _testshader_struct.
        {
            std::ofstream shader(dir / "_testshader_dependent.slang");
            shader << R"SHADER(
import _testshader_struct;
[shader("compute")]
[numthreads(1, 1, 1)]
void main_a(uint3 tid : SV_DispatchThreadID, uniform Foo foo)
{
}
)SHADER";
            shader.close();
        }
    }
}

TEST_SUITE_BEGIN("device");

TEST_CASE_GPU("shader")
{
    auto dir = testing::get_case_temp_directory();

    // Perform 1-time setup that creates shader files for these test cases.
    setup_testshader_files(dir);

    // Just verify module loads.
    SUBCASE("load_module")
    {
        ref<SlangModule> module = ctx.device->load_module((dir / "_testshader_simple.slang").string());
        CHECK(module);
    }

    // Load a module with no external dependencies and verify it only depends on itself.
    SUBCASE("single_module_dependency")
    {
        ref<SlangModule> module = ctx.device->load_module((dir / "_testshader_simple.slang").string());
        CHECK_EQ(module->slang_module()->getDependencyFileCount(), 1);
        std::filesystem::path path0 = module->slang_module()->getDependencyFilePath(0);
        CHECK_EQ(path0.filename(), "_testshader_simple.slang");
    }

    // Load a module with a 2-stage dependency chain and verify all 3 dependencies.
    SUBCASE("multi_module_dependency")
    {
        ref<SlangModule> module = ctx.device->load_module((dir / "_testshader_dependent.slang").string());
        REQUIRE_EQ(module->slang_module()->getDependencyFileCount(), 3);
        std::vector<std::filesystem::path> paths{
            module->slang_module()->getDependencyFilePath(0),
            module->slang_module()->getDependencyFilePath(1),
            module->slang_module()->getDependencyFilePath(2),
        };
        std::sort(
            paths.begin(),
            paths.end(),
            [](const auto& a, const auto& b)
            {
                return a.filename() < b.filename();
            }
        );
        CHECK_EQ(paths[0].filename(), "_testshader_dependent.slang");
        CHECK_EQ(paths[1].filename(), "_testshader_struct.slang");
        CHECK_EQ(paths[2].filename(), "print.slang");
    }
}

TEST_CASE_GPU("coverage_capture_finishes_after_close")
{
    if (ctx.device->type() != DeviceType::vulkan && ctx.device->type() != DeviceType::cuda)
        return;

    DeviceDesc desc{.type = ctx.device->type(), .enable_debug_layers = true};
    desc.compiler_options.coverage = ShaderCoverageOptions{.counter_width = 32};
    auto device = Device::create(desc);
    auto module = device->load_module_from_source(
        "coverage_lifetime",
        "[shader(\"compute\")][numthreads(1,1,1)] void compute_main() {}"
    );
    auto program = device->link_program({module}, {module->entry_point("compute_main")});
    device->create_compute_kernel({.program = program})->dispatch({3, 1, 1}, {});
    auto collector = device->shader_coverage();
    auto baseline = collector->snapshot();

    // Submit both captures before completing either. Close clears the registry
    // before finish() accesses readback, so only the capture's owned resources remain.
    auto first = collector->_begin_capture(true, true);
    auto second = collector->_begin_capture(true, false);
    device->close();
    CHECK_THROWS_WITH(collector->snapshot(), doctest::Contains("Device is closed"));
    auto after_reset = second.finish();
    auto before_reset = first.finish();

    REQUIRE_EQ(baseline.programs.size(), 1);
    REQUIRE_EQ(before_reset.programs.size(), 1);
    REQUIRE_EQ(after_reset.programs.size(), 1);
    CHECK(
        std::any_of(
            baseline.programs[0].counters.begin(),
            baseline.programs[0].counters.end(),
            [](uint64_t count)
            {
                return count != 0;
            }
        )
    );
    CHECK(before_reset.programs[0].counters == baseline.programs[0].counters);
    CHECK(
        std::all_of(
            after_reset.programs[0].counters.begin(),
            after_reset.programs[0].counters.end(),
            [](uint64_t count)
            {
                return count == 0;
            }
        )
    );
    CHECK_EQ(before_reset.capture_id + 1, after_reset.capture_id);
    CHECK_EQ(before_reset.interval_id + 1, after_reset.interval_id);
}

TEST_CASE_GPU("internal_shader_session")
{
    CHECK_EQ(ctx.device->_internal_slang_session(), ctx.device->slang_session());
    if (ctx.device->type() != DeviceType::vulkan && ctx.device->type() != DeviceType::cuda)
        return;

    DeviceDesc desc{.type = ctx.device->type()};
    desc.compiler_options.coverage = ShaderCoverageOptions{.counter_width = 32};
    desc.compiler_options.defines["INTERNAL_SESSION_TEST"] = "1";
    auto device = Device::create(desc);
    auto* application = device->slang_session();
    auto* internal = device->_internal_slang_session();
    CHECK_NE(internal, application);
    CHECK_EQ(internal, device->_internal_slang_session());
    CHECK(application->desc().compiler_options.coverage.has_value());
    CHECK_FALSE(internal->desc().compiler_options.coverage.has_value());
    CHECK(internal->desc().compiler_options.defines == application->desc().compiler_options.defines);
    CHECK(internal->desc().compiler_options.include_paths == application->desc().compiler_options.include_paths);
    CHECK(internal->desc().cache_path == application->desc().cache_path);
    CHECK_EQ(internal->desc().add_default_include_paths, application->desc().add_default_include_paths);
    device->close();
    CHECK_THROWS_WITH(device->_internal_slang_session(), doctest::Contains("Device is closed"));
}

TEST_SUITE_END();
