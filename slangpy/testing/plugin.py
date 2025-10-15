# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import inspect
from typing import Any

import slangpy as spy
from .helpers import (
    close_all_devices,
    close_leaked_devices,
    set_device_types,
    should_skip_test_for_device,
    should_skip_non_device_test,
    SELECTED_DEVICE_TYPES,
)


def pytest_addoption(parser: pytest.Parser):
    """Add command line options for testing specific device types."""
    group = parser.getgroup("slangpy", "SlangPy test configuration")
    group.addoption(
        "--device-types",
        action="store",
        default=None,
        help="Run tests only for the specified device types (comma-separated). "
        "Valid types: d3d12, vulkan, cuda, metal. Use 'nodevice' for tests that don't require a device.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session):
    # pytest's stdout/stderr capturing sometimes leads to bad file descriptor exceptions
    # when logging in sgl. By setting IGNORE_PRINT_EXCEPTION, we ignore those exceptions.
    spy.ConsoleLoggerOutput.IGNORE_PRINT_EXCEPTION = True

    # Set the global device types based on the command line option
    device_types_option = session.config.getoption("--device-types")
    set_device_types(device_types_option)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    close_all_devices()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: Any, nextitem: Any):
    close_leaked_devices()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Any) -> None:
    """
    Automatically skip tests based on device type when testing specific devices.
    This hook runs before each test and can skip tests that don't match
    the target device types.
    """
    # Check if the test function has a device_type parameter
    if hasattr(item, "function"):
        sig = inspect.signature(item.function)
        if "device_type" in sig.parameters:
            # This is a device-based test
            # Get the device_type value for this specific test instance
            if hasattr(item, "callspec") and "device_type" in item.callspec.params:
                test_device_type = item.callspec.params["device_type"]
                if should_skip_test_for_device(test_device_type):
                    target_device_names = (
                        [dt.name for dt in SELECTED_DEVICE_TYPES]
                        if SELECTED_DEVICE_TYPES
                        else ["nodevice"]
                    )
                    pytest.skip(
                        f"Skipping test for device type {test_device_type.name} (target devices: {', '.join(target_device_names)})"
                    )
        else:
            # This is a non-device test, skip it unless in nodevice mode
            if should_skip_non_device_test():
                target_device_names = (
                    [dt.name for dt in SELECTED_DEVICE_TYPES]
                    if SELECTED_DEVICE_TYPES
                    else ["nodevice"]
                )
                pytest.skip(
                    f"Skipping non-device test (target devices: {', '.join(target_device_names)})"
                )
