# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from .helpers import close_all_devices, close_leaked_devices

from typing import Any


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session):
    # pytest's stdout/stderr capturing sometimes leads to bad file descriptor exceptions
    # when logging in sgl. By setting IGNORE_PRINT_EXCEPTION, we ignore those exceptions.
    spy.ConsoleLoggerOutput.IGNORE_PRINT_EXCEPTION = True


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    close_all_devices()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: Any, nextitem: Any):
    close_leaked_devices()
