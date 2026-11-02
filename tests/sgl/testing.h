// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <doctest/doctest.h>
#include <filesystem>

namespace sgl {

class Device;

} // namespace sgl

namespace sgl::testing {

/// Get name of running test suite (note: defined in sgl_tests.cpp).
std::string get_current_test_suite_name();

/// Get name of running test case (note: defined in sgl_tests.cpp).
std::string get_current_test_case_name();

/// Get global temp directory for tests.
std::filesystem::path get_test_temp_directory();

/// Get temp directory for current test suite.
std::filesystem::path get_suite_temp_directory();

/// Get temp directory for current test case.
std::filesystem::path get_case_temp_directory();

void static_init();
void static_shutdown();

struct TestOptions {
    bool skip_device_tests{false};
};

/// Return the test runner options.
TestOptions& options();

struct GpuTestContext {
    Device* device;
};

void run_gpu_test(void (*func)(GpuTestContext&));

void release_cached_devices();

/// Return whether device-backed tests are enabled.
bool device_tests_enabled();

/// Record that the current test was skipped at runtime.
void report_skip(const doctest::detail::TestCase* test_case, const char* reason);

/// Return the runtime skip reason for a test, or nullptr if it was not skipped.
const char* get_skip_message(const doctest::TestCaseData* test_case);

} // namespace sgl::testing


#define DOCTEST_TEST_CASE_GPU(f, name)                                                                                 \
    static void f(::sgl::testing::GpuTestContext& ctx);                                                                \
    TEST_CASE(name)                                                                                                    \
    {                                                                                                                  \
        ::sgl::testing::run_gpu_test(f);                                                                               \
    }                                                                                                                  \
    static void f(::sgl::testing::GpuTestContext& ctx)


#define TEST_CASE_GPU(name) DOCTEST_TEST_CASE_GPU(DOCTEST_ANONYMOUS(gpu_test), name)


// doctest does not support skipping tests at runtime. This macro records the skip for the custom reporter and returns
// from the current test function. The reason must be a string literal, and the macro must be used in the main scope of
// the test function.
#define SKIP(reason)                                                                                                   \
    do {                                                                                                               \
        ::sgl::testing::report_skip(::doctest::getContextOptions()->currentTest, "" reason);                           \
        return;                                                                                                        \
    } while (false)
