// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/sgl.h"
#include "sgl/device/device.h"
#include "sgl/device/agility_sdk.h"
#include "sgl/core/object.h"
#include "sgl/core/logger.h"
#include "sgl/core/error.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

SGL_EXPORT_AGILITY_SDK


namespace sgl::testing {

// Helpers to get current test suite and case name.
// See https://github.com/doctest/doctest/issues/345.
// Has to be defined in the same file as DOCTEST_CONFIG_IMPLEMENT
std::string get_current_test_suite_name()
{
    return doctest::detail::g_cs->currentTest->m_test_suite;
}
std::string get_current_test_case_name()
{
    return doctest::detail::g_cs->currentTest->m_name;
}

} // namespace sgl::testing

int main(int argc, char** argv)
{
    sgl::static_init();

    sgl::Logger::get().remove_all_outputs();
    sgl::Logger::get().add_debug_console_output();
    sgl::Logger::get().add_file_output("sgl_tests.log");

    /// Do not break debugger on exceptions when running tests.
    sgl::set_exception_diagnostics(sgl::ExceptionDiagnosticFlags::none);

    int result = 1;
    {
        sgl::testing::static_init();

        doctest::Context context(argc, argv);

        // Select specific test suite to run
        // context.setOption("-ts", "formats");
        // Report successful tests
        // context.setOption("success", true);

        result = context.run();

        sgl::testing::static_shutdown();
    }

    sgl::Device::close_all_devices();

    sgl::static_shutdown();

    return result;
}
