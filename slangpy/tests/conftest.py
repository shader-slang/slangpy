# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
import slangpy as spy


# Called after every test to ensure any devices that aren't part of the
# device cache are cleaned up.
def pytest_runtest_teardown(item: Any, nextitem: Any):
    for device in spy.Device.get_created_devices():
        if device.desc.label.startswith("cached-"):
            continue
        print(f"Closing leaked device {device.desc.label}")
        device.close()


def pytest_sessionstart(session: Any):
    # pytest's stdout/stderr capturing sometimes leads to bad file descriptor exceptions
    # when logging in sgl. By setting IGNORE_PRINT_EXCEPTION, we ignore those exceptions.
    spy.ConsoleLoggerOutput.IGNORE_PRINT_EXCEPTION = True


# After all tests finished, close remaining devices. This ensures they're
# cleaned up before pytorch, avoiding crashes for devices that share context.
def pytest_sessionfinish(session: Any, exitstatus: Any):

    # If torch enabled, sync all devices to ensure all operations are finished.
    try:
        import torch

        torch.cuda.synchronize()
    except ImportError:  # @IgnoreException
        pass

    # Close all devices that were created during the tests.
    for device in spy.Device.get_created_devices():
        print(f"Closing device on shutdown {device.desc.label}")
        device.close()
