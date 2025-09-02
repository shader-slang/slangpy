# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import inspect
from typing import Any
from slangpy.testing import helpers

# Importing these here means we don't have to explicitly import them into every benchmark
# file, even if the benchmark file doesn't directly reference them.
from slangpy.testing.benchmark import benchmark_slang_function, benchmark_python_function, benchmark_compute_kernel, report  # type: ignore

pytest_plugins = ["slangpy.testing.plugin", "slangpy.testing.benchmark.plugin"]


def pytest_runtest_setup(item: Any) -> None:
    """
    Automatically skip tests based on device type when in benchmark mode.
    This hook runs before each test and can skip tests that don't match
    the target benchmark device type.
    """
    if not helpers.is_benchmark_mode():
        return

    # Check if the test function has a device_type parameter
    if hasattr(item, "function"):
        sig = inspect.signature(item.function)
        if "device_type" in sig.parameters:
            # This is a device-based test
            if helpers.is_benchmark_nodevice_mode():
                pytest.skip("Skipping device test in nodevice benchmark mode")

            # Get the device_type value for this specific test instance
            if hasattr(item, "callspec") and "device_type" in item.callspec.params:
                test_device_type = item.callspec.params["device_type"]
                if helpers.should_skip_test_for_device(test_device_type):
                    benchmark_device = helpers.get_benchmark_device_type()
                    benchmark_device_name = benchmark_device.name if benchmark_device else "unknown"
                    pytest.skip(
                        f"Skipping test for device type {test_device_type.name} (benchmark mode: {benchmark_device_name})"
                    )
        else:
            # This is a non-device test, skip it unless in nodevice mode
            if helpers.should_skip_non_device_test():
                benchmark_device = helpers.get_benchmark_device_type()
                benchmark_device_name = benchmark_device.name if benchmark_device else "unknown"
                pytest.skip(
                    f"Skipping non-device test in device benchmark mode (targeting: {benchmark_device_name})"
                )
