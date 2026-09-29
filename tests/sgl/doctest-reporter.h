// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "testing.h"

#include <doctest/doctest.h>

namespace doctest {
namespace {

    /// Console reporter with support for the runtime skips recorded by the SGL test framework.
    struct SglReporter : public IReporter {
        std::ostream& stream;
        const ContextOptions& options;
        ConsoleReporter console_reporter;
        const TestCaseData* test_case{nullptr};

        explicit SglReporter(const ContextOptions& options)
            : stream(*options.cout)
            , options(options)
            , console_reporter(options)
        {
        }

        void report_query(const QueryData& data) override
        {
            console_reporter.report_query(data);
            if (options.help) {
                stream << "\n[doctest] Additional SGL-specific options:\n\n";
                stream << " -skip-device-tests                   skip tests that require a device\n";
            }
        }

        void test_run_start() override { console_reporter.test_run_start(); }
        void test_run_end(const TestRunStats& stats) override { console_reporter.test_run_end(stats); }

        void test_case_start(const TestCaseData& data) override
        {
            test_case = &data;
            console_reporter.test_case_start(data);
        }

        void test_case_reenter(const TestCaseData& data) override { console_reporter.test_case_reenter(data); }

        void test_case_end(const CurrentTestCaseStats& stats) override
        {
            const char* skip_message = sgl::testing::get_skip_message(test_case);
            if (!skip_message || stats.failure_flags) {
                console_reporter.test_case_end(stats);
                return;
            }

            if (!options.quiet && !test_case->m_no_output)
                stream << "[doctest] SKIPPED: " << test_case->m_name << " (" << skip_message << ")\n";
        }

        void test_case_exception(const TestCaseException& exception) override
        {
            console_reporter.test_case_exception(exception);
        }

        void subcase_start(const SubcaseSignature& signature) override { console_reporter.subcase_start(signature); }
        void subcase_end() override { console_reporter.subcase_end(); }
        void log_assert(const AssertData& data) override { console_reporter.log_assert(data); }
        void log_message(const MessageData& data) override { console_reporter.log_message(data); }
        void test_case_skipped(const TestCaseData& data) override { console_reporter.test_case_skipped(data); }
    };

    REGISTER_REPORTER("sgl", 1, SglReporter);

} // namespace
} // namespace doctest
