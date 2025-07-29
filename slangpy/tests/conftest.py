# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
import pytest
import slangpy as spy


@pytest.fixture(autouse=True)
def clean_up():
    yield

    for device in spy.Device.get_created_devices():
        if device.desc.label.startswith("cached-"):
            continue
        print(f"Closing device {device.desc.label}")
        device.close()


def pytest_sessionfinish(session: Any, exitstatus: Any):
    print("\n[SHUTDOWN] All tests are complete.")
    for device in spy.Device.get_created_devices():
        print(f"  Closing device {device.desc.label}")
        device.close()
