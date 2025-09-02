# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import inspect
from typing import Any

import slangpy as spy
from .helpers import close_all_devices, close_leaked_devices, set_device_type


def pytest_addoption(parser: pytest.Parser):
    """Add command line options for testing single device type."""
    group = parser.getgroup("slangpy", "SlangPy test configuration")
    group.addoption(
        "--device-type",
        action="store",
        default=None,
        choices=["d3d12", "vulkan", "cuda", "metal", "nodevice"],
        help="Run tests only for the specified device type. Use 'nodevice' for tests that don't require a device.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session):
    # pytest's stdout/stderr capturing sometimes leads to bad file descriptor exceptions
    # when logging in sgl. By setting IGNORE_PRINT_EXCEPTION, we ignore those exceptions.
    spy.ConsoleLoggerOutput.IGNORE_PRINT_EXCEPTION = True

    # Set the global device type based on the command line option
    device_type_option = session.config.getoption("--device-type")
    set_device_type(device_type_option)


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
    the target device type.
    """
    # Import here to avoid circular imports
    from .helpers import (
        should_skip_test_for_device,
        should_skip_non_device_test,
        TEST_SINGLE_DEVICE_TYPE,
        TEST_DEVICE_TYPE,
    )

    if not TEST_SINGLE_DEVICE_TYPE:
        return

    # Check if the test function has a device_type parameter
    if hasattr(item, "function"):
        sig = inspect.signature(item.function)
        if "device_type" in sig.parameters:
            # This is a device-based test
            if TEST_DEVICE_TYPE is None:  # nodevice mode
                pytest.skip("Skipping device test in nodevice mode")

            # Get the device_type value for this specific test instance
            if hasattr(item, "callspec") and "device_type" in item.callspec.params:
                test_device_type = item.callspec.params["device_type"]
                if should_skip_test_for_device(test_device_type):
                    target_device_name = TEST_DEVICE_TYPE.name if TEST_DEVICE_TYPE else "nodevice"
                    pytest.skip(
                        f"Skipping test for device type {test_device_type.name} (target device: {target_device_name})"
                    )
        else:
            # This is a non-device test, skip it unless in nodevice mode
            if should_skip_non_device_test():
                target_device_name = TEST_DEVICE_TYPE.name if TEST_DEVICE_TYPE else "nodevice"
                pytest.skip(f"Skipping non-device test (target device: {target_device_name})")
