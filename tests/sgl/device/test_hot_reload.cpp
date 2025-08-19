// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/format.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/kernel.h"
#include "sgl/device/hot_reload.h"
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>

using namespace sgl;

// Writes out a shader with a line in that is "outbuffer[tid.x] = <set_to>;".
struct WriteShaderDesc {
    std::filesystem::path path;
    std::string set_to;
    std::string kernel_name{"compute_main"};
    std::string param_name{"outbuffer"};
    std::vector<std::string> imports;
};
static void write_shader(const WriteShaderDesc& desc)
{
    std::string imports;
    for (auto& i : desc.imports) {
        imports += "import " + i + ";\n";
    }

    std::string formatted = fmt::format(
        R"SHADER(
{3}
RWStructuredBuffer<uint> {0};
[shader("compute")]
[numthreads(1, 1, 1)]
void {1}(uint3 tid : SV_DispatchThreadID)
{{
if(tid.x<1024)
    {0}[tid.x] = {2};
}}
)SHADER",
        desc.param_name,
        desc.kernel_name,
        desc.set_to,
        imports
    );

    std::ofstream shader(desc.path);
    shader << formatted;
    shader.close();
}

// Writes out a shader module with a function that returns <set_to>.
struct WriteModuleDesc {
    std::filesystem::path path;
    std::string func_name{"func"};
    std::string set_to;
    std::vector<std::string> imports;
};
static void write_module(const WriteModuleDesc& desc)
{
    std::string imports;
    for (auto& i : desc.imports) {
        imports += "import " + i + ";\n";
    }

    std::string formatted = fmt::format(
        R"SHADER(
{0}
int {1}()
{{
    return {2};
}}
)SHADER",
        imports,
        desc.func_name,
        desc.set_to
    );

    std::ofstream shader(desc.path);
    shader << formatted;
    shader.close();
}

static bool run_and_verify(
    testing::GpuTestContext& ctx,
    ref<ComputeKernel> kernel,
    uint32_t expected_value,
    bool expect_success = true
)
{
    static int g_zeros[1024];
    static int g_results[1024];
    memset(g_zeros, 0, sizeof(g_zeros));
    ref<Buffer> buffer = ctx.device->create_buffer({
        .element_count = 1024,
        .struct_size = 4,
        .usage = BufferUsage::shader_resource | BufferUsage::unordered_access,
        .data = g_zeros,
        .data_size = sizeof(g_zeros),
    });

    kernel->dispatch(uint3(1024, 1, 1), [&buffer](ShaderCursor cursor) { cursor["outbuffer"] = buffer; });

    memset(g_results, 0, sizeof(g_results));
    buffer->get_data(g_results, sizeof(g_results));

    bool all_correct = true;
    for (auto x : g_results) {
        if (expect_success)
            all_correct = all_correct && x == (int)expected_value;
        else
            all_correct = all_correct && x != (int)expected_value;
    }
    return all_correct;
}

TEST_SUITE_BEGIN("hot_reload");

TEST_CASE_GPU("verify test case works")
{
    auto path = testing::get_case_temp_directory() / "verify.slang";
    write_shader({.path = path, .set_to = "1"});
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));
    CHECK(run_and_verify(ctx, kernel, 2, false));
}

TEST_CASE_GPU("change program and recreate")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "changeprog.slang";
    write_shader({.path = path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Re-write the shader, and verify it still returns 1, as hasn't reloaded yet.
    write_shader({.path = path, .set_to = "2"});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Force a reload, and verify the result is now 2.
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(run_and_verify(ctx, kernel, 2));

    // Hot reload should not report error
    CHECK(!ctx.device->_hot_reload()->last_build_failed());
}

TEST_CASE_GPU("change program with error and recreate")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "changeprogerror.slang";
    write_shader({.path = path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Re-write the shader, and verify it still returns 1, as hasn't reloaded yet.
    write_shader({.path = path, .set_to = "1adsda"});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Force a reload
    ctx.device->_hot_reload()->recreate_all_sessions();

    // Hot reload should now report error
    CHECK(ctx.device->_hot_reload()->last_build_failed());

    // Program should still be valid and return 1
    CHECK(run_and_verify(ctx, kernel, 1));
}

TEST_CASE_GPU("change kernel name and recreate")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "changekernelname.slang";
    write_shader({.path = path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Re-write the shader, and verify it still returns 1, as hasn't reloaded yet.
    write_shader({.path = path, .set_to = "1", .kernel_name = "main2"});

    // Force a reload
    ctx.device->_hot_reload()->recreate_all_sessions();

    // Hot reload should report error, as entry point name has changed so kernel is invalid
    CHECK(ctx.device->_hot_reload()->last_build_failed());

    // Program should still be valid and return 1.
    CHECK(run_and_verify(ctx, kernel, 1));
}

TEST_CASE_GPU("change buffer name and fail to use recreated program")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "changebuffer.slang";
    write_shader({.path = path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Re-write the shader, and verify it still returns 1, as hasn't reloaded yet.
    write_shader({.path = path, .set_to = "2", .param_name = "outbuffer2"});

    // Force a reload, which should succeed.
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(!ctx.device->_hot_reload()->last_build_failed());

    // Verify should throw exception, as the normal parameter name is now wrong.
    CHECK_THROWS(run_and_verify(ctx, kernel, 2));
}

TEST_CASE_GPU("change program with invalid imports and recreate")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "badimport.slang";
    write_shader({.path = path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Re-write shader with an import that doesn't exist and check build fails.
    write_shader({.path = path, .set_to = "1", .imports = {"blabla"}});
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(ctx.device->_hot_reload()->last_build_failed());

    // Program should still be valid and return 1
    CHECK(run_and_verify(ctx, kernel, 1));
}

TEST_CASE_GPU("change program with correct module import and recreate")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "goodimport.slang";
    write_shader({.path = path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Create a module with a function that returns 2.
    std::filesystem::path abs_module_path = testing::get_case_temp_directory() / "goodimportmodule.slang";
    write_module({.path = abs_module_path, .set_to = "2"});

    // Re-write shader with a valid import and check build succeeds.
    write_shader({.path = path, .set_to = "func()", .imports = {"goodimportmodule"}});
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(!ctx.device->_hot_reload()->last_build_failed());

    // Program should now be valid and return 2.
    CHECK(run_and_verify(ctx, kernel, 2));
}

TEST_CASE_GPU("leave program but change the module it imports")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "changeimported.slang";
    write_shader({.path = path, .set_to = "func()", .imports = {"changeimportedmodule"}});
    std::filesystem::path abs_module_path = testing::get_case_temp_directory() / "changeimportedmodule.slang";
    write_module({.path = abs_module_path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Recreate the module with a new value and recompile
    write_module({.path = abs_module_path, .set_to = "2"});
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(!ctx.device->_hot_reload()->last_build_failed());

    // Program should now be valid and return 2.
    CHECK(run_and_verify(ctx, kernel, 2));
}

TEST_CASE_GPU("leave program then break the module it imports")
{
    // Disable auto detect changes so can test explicit reload.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "breakimported.slang";
    write_shader({.path = path, .set_to = "func()", .imports = {"breakimportedmodule"}});
    std::filesystem::path abs_module_path = testing::get_case_temp_directory() / "breakimportedmodule.slang";
    write_module({.path = abs_module_path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Recreate the module with a new value and expect failed recompile.
    write_module({.path = abs_module_path, .set_to = "blabla"});
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(ctx.device->_hot_reload()->last_build_failed());

    // Program should still be valid and return 1.
    CHECK(run_and_verify(ctx, kernel, 1));
}

TEST_CASE_GPU("change program with basic additional source")
{
    // Disable auto detection.
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Create a new session explicitly with no include paths.
    SlangCompilerOptions opts = ctx.device->desc().compiler_options;
    opts.include_paths.clear();
    ref<sgl::SlangSession> session = ctx.device->create_slang_session({
        .compiler_options = opts,
        .add_default_include_paths = true,
    });

    // Write a teeny 'config' file that predeclares the constant in additional source
    std::ofstream config_module(testing::get_case_temp_directory() / "addsourcebasicconfig.slang");
    config_module << "extern static const uint SOMENUMBER;\n";
    config_module.close();

    // Build the additional source code that actually sets the number
    std::string addsource = "export static const uint SOMENUMBER = 1;\n";

    // Write first version of shader that reads a value from the additional source.
    auto path = testing::get_case_temp_directory() / "addsourcebasic.slang";
    write_shader({
        .path = path,
        .set_to = "SOMENUMBER",
        .imports = {"addsourcebasicconfig"},
    });

    // Load program + kernel with the extra source, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"}, addsource);
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Modify the shader to add one to the returned number and verify returns 2
    write_shader({
        .path = path,
        .set_to = "SOMENUMBER+1",
        .imports = {"addsourcebasicconfig"},
    });
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(!ctx.device->_hot_reload()->last_build_failed());
    CHECK(run_and_verify(ctx, kernel, 2));
}

TEST_CASE_GPU("load module separately from program")
{
    // Disable auto detection
    ctx.device->_hot_reload()->set_auto_detect_changes(false);

    // Create module in subdirectory.
    std::filesystem::path inc_path = testing::get_case_temp_directory() / "sepmod_inc";
    std::filesystem::create_directories(inc_path);
    std::filesystem::path mod_path = inc_path / "mod.slang";
    write_module({
        .path = mod_path,
        .set_to = "1",
    });

    // Create a new session with access to the subdirectory.
    SlangCompilerOptions opts = ctx.device->desc().compiler_options;
    opts.include_paths.clear();
    opts.include_paths.push_back(inc_path);
    ref<sgl::SlangSession> session = ctx.device->create_slang_session({
        .compiler_options = opts,
        .add_default_include_paths = true,
    });

    // Write version of shader that reads a value from the module.
    auto path = testing::get_case_temp_directory() / "sepmod.slang";
    write_shader({
        .path = path,
        .set_to = "func()",
        .imports = {"mod"},
    });

    // Load module then program independently and verify result.
    ref<SlangModule> module = session->load_module(mod_path.string());
    ref<ShaderProgram> program = session->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Modify the module to return 2 and verify the result.
    write_module({
        .path = mod_path,
        .set_to = "2",
    });
    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(!ctx.device->_hot_reload()->last_build_failed());
    CHECK(run_and_verify(ctx, kernel, 2));
}


TEST_CASE_GPU("change program and auto detect changes")
{
    // Enable auto detection and wipe any existing monitors to ensure test is from a 'clean slate'.
    ctx.device->_hot_reload()->set_auto_detect_changes(true);
    ctx.device->_hot_reload()->set_auto_detect_delay(25);
    ctx.device->_hot_reload()->_clear_file_watches();

    // Write first version of shader that outputs 1.
    auto path = testing::get_case_temp_directory() / "detectchangeprog.slang";
    write_shader({.path = path, .set_to = "1"});

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = ctx.device->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Re-write the shader so it returns 2 and wait for it to reload.
    ctx.device->_hot_reload()->_reset_reloaded();
    write_shader({.path = path, .set_to = "2"});
    for (int i = 0; i < 400 && !ctx.device->_hot_reload()->_has_reloaded(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        ctx.device->_hot_reload()->update();
    }

    // Verify the result is now 2.
    INFO("Has reloaded: " << ctx.device->_hot_reload()->_has_reloaded());
    INFO("Has error: " << ctx.device->_hot_reload()->last_build_failed());
    CHECK(run_and_verify(ctx, kernel, 2));

    // Hot reload should not report error.
    CHECK(!ctx.device->_hot_reload()->last_build_failed());
}

/// SKIPPED: This test is flaky on CI, and needs to be reworked.
TEST_CASE_GPU("create multi directory session and monitor for changes" * doctest::skip())
{
    // Enable auto detection and wipe any existing monitors to ensure test is from a 'clean slate'.
    ctx.device->_hot_reload()->set_auto_detect_changes(true);
    ctx.device->_hot_reload()->set_auto_detect_delay(25);
    ctx.device->_hot_reload()->_clear_file_watches();

    // Create 2 directories, and a module inside each one.
    std::filesystem::path inc_path_0 = testing::get_case_temp_directory() / "md_inc_0";
    std::filesystem::path inc_path_1 = testing::get_case_temp_directory() / "md_inc_1";
    std::filesystem::create_directories(inc_path_0);
    std::filesystem::create_directories(inc_path_1);
    std::filesystem::path mod_path_0 = inc_path_0 / "mod0.slang";
    std::filesystem::path mod_path_1 = inc_path_1 / "mod1.slang";
    write_module({
        .path = mod_path_0,
        .func_name = "mod0func",
        .set_to = "2",
    });
    write_module({
        .path = mod_path_1,
        .func_name = "mod1func",
        .set_to = "3",
    });

    // Create a new session with access to both directories.
    SlangCompilerOptions opts = ctx.device->desc().compiler_options;
    opts.include_paths.push_back(inc_path_0);
    opts.include_paths.push_back(inc_path_1);
    ref<sgl::SlangSession> session = ctx.device->create_slang_session({
        .compiler_options = opts,
        .add_default_include_paths = true,
    });

    // Write first version of shader that imports but doesn't use both modules.
    auto path = testing::get_case_temp_directory() / "multidirchangeprog.slang";
    write_shader({
        .path = path,
        .set_to = "1",
        .imports = {"mod0", "mod1"},
    });

    // Load program + kernel, and verify returns 1.
    ref<ShaderProgram> program = session->load_program(path.string(), {"compute_main"});
    ref<ComputeKernel> kernel = ctx.device->create_compute_kernel({.program = program});
    CHECK(run_and_verify(ctx, kernel, 1));

    // Re-write the shader to call mod0 and check changes are detected.
    write_shader({
        .path = path,
        .set_to = "mod0func()",
        .imports = {"mod0", "mod1"},
    });
    ctx.device->_hot_reload()->_reset_reloaded();
    for (int i = 0; i < 400 && !ctx.device->_hot_reload()->_has_reloaded(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        ctx.device->_hot_reload()->update();
    }
    CHECK(run_and_verify(ctx, kernel, 2));

    // Modify module 0 to return a different number and check.
    write_module({
        .path = mod_path_0,
        .func_name = "mod0func",
        .set_to = "10",
    });
    ctx.device->_hot_reload()->_reset_reloaded();
    for (int i = 0; i < 400 && !ctx.device->_hot_reload()->_has_reloaded(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        ctx.device->_hot_reload()->update();
    }
    CHECK(run_and_verify(ctx, kernel, 10));

    // Modify shader to use mod1, AND modify mod1, and check.
    write_shader({
        .path = path,
        .set_to = "mod1func()",
        .imports = {"mod0", "mod1"},
    });
    write_module({
        .path = mod_path_1,
        .func_name = "mod1func",
        .set_to = "20",
    });
    ctx.device->_hot_reload()->_reset_reloaded();
    for (int i = 0; i < 400 && !ctx.device->_hot_reload()->_has_reloaded(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        ctx.device->_hot_reload()->update();
    }
    CHECK(run_and_verify(ctx, kernel, 20));
}

TEST_SUITE_END();
