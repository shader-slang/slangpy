// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// doctest unconditionally includes <ciso646> on clang; libstdc++ >= 13 makes that
// a deprecation #warning in C++20+, which -Werror promotes to a fatal error.
#if defined(__clang__) && __cplusplus >= 202002L
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#endif
#include <doctest/doctest.h>
#if defined(__clang__) && __cplusplus >= 202002L
#pragma clang diagnostic pop
#endif
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

struct GpuTestContext {
    Device* device;
};

void run_gpu_test(void (*func)(GpuTestContext&));

void release_cached_devices();

} // namespace sgl::testing


#define DOCTEST_TEST_CASE_GPU(f, name)                                                                                 \
    static void f(::sgl::testing::GpuTestContext& ctx);                                                                \
    TEST_CASE(name)                                                                                                    \
    {                                                                                                                  \
        ::sgl::testing::run_gpu_test(f);                                                                               \
    }                                                                                                                  \
    static void f(::sgl::testing::GpuTestContext& ctx)


#define TEST_CASE_GPU(name) DOCTEST_TEST_CASE_GPU(DOCTEST_ANONYMOUS(gpu_test), name)
