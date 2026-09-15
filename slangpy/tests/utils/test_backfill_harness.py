# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The backfill accommodations in the shared harness stay shut in ordinary CI.

The backfill itself is a one-shot sweep whose mistakes surface on the first run.
These two gates are different: they live in ``slangpy/testing/helpers.py``, which
every test run imports, and if either softened, ordinary CI would quietly skip
what it should be failing on.
"""

from types import SimpleNamespace
from typing import Any

import pytest

import slangpy as spy
from slangpy.testing import helpers


def test_a_missing_api_is_fatal_outside_a_backfill(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(__name__="fake", present=1)
    helpers.require_apis(module, "present")

    with pytest.raises(ImportError, match="fake.absent"):
        helpers.require_apis(module, "absent")

    monkeypatch.setattr(helpers, "BACKFILL_TARGET_SHA", "0123456789abcdef")
    with pytest.raises(pytest.skip.Exception, match="fake.absent"):
        helpers.require_apis(module, "absent")


def test_a_required_device_failure_is_never_absorbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only CUDA is an era-dependent capability gap; d3d12 failing is a regression."""

    def refuse(**kwargs: Any) -> Any:
        raise RuntimeError("device creation failed")

    monkeypatch.setattr(helpers, "Device", refuse)
    monkeypatch.setattr(helpers, "BACKFILL_TARGET_SHA", "0123456789abcdef")
    monkeypatch.setattr(helpers, "BACKFILL_UNAVAILABLE_DEVICES", {})
    monkeypatch.setattr(helpers, "DEVICE_CACHE", {})
    monkeypatch.setattr(helpers, "SELECTED_DEVICE_TYPES", None)

    with pytest.raises(pytest.skip.Exception):
        helpers.get_device(spy.DeviceType.cuda)

    # Caught explicitly: a skip escaping here would mark this test skipped and hide
    # the very regression it exists to catch.
    try:
        helpers.get_device(spy.DeviceType.d3d12)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"required device type was skipped instead of reported: {skipped}")
    except RuntimeError as error:
        assert "device creation failed" in str(error)
    else:
        pytest.fail("device creation was expected to fail")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
