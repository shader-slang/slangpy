// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
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
        CHECK_EQ(module->slang_module()->getDependencyFileCount(), 3);
        std::vector<std::filesystem::path> paths{
            module->slang_module()->getDependencyFilePath(0),
            module->slang_module()->getDependencyFilePath(1),
            module->slang_module()->getDependencyFilePath(2),
        };
        std::sort(paths.begin(), paths.end(), [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
        CHECK_EQ(paths[0].filename(), "_testshader_dependent.slang");
        CHECK_EQ(paths[1].filename(), "_testshader_struct.slang");
        CHECK_EQ(paths[2].filename(), "print.slang");
    }
}

TEST_SUITE_END();
