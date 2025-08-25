# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from typing import Any
from pathlib import Path
from slangpy.testing import helpers

SHADER_DIR = Path(__file__).parent / "slangpy_tests"


def pytest_sessionstart(session: pytest.Session):
    helpers.start_session(shader_include_paths=[SHADER_DIR])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    helpers.finish_session()


def pytest_runtest_setup(item: Any):
    helpers.setup_test()


def pytest_runtest_teardown(item: Any, nextitem: Any):
    helpers.teardown_test()
