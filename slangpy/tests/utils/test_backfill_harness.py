# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Coverage for the backfill accommodations in the shared test harness.

The backfill runs today's benchmarks against year-old builds, so the harness has to
tolerate APIs and device types those builds do not have. Every accommodation is gated
on BACKFILL_TARGET_SHA, and these tests pin both sides of that gate: absorbed during a
backfill, fatal everywhere else.
"""

from types import SimpleNamespace
from typing import Any

import pytest

import slangpy as spy
from slangpy.testing import helpers


def test_a_missing_api_is_fatal_outside_a_backfill_and_skips_inside_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both sides of the gate matter: softening ordinary CI would hide a regression."""

    module = SimpleNamespace(__name__="fake", present=1)
    helpers.require_apis(module, "present")

    with pytest.raises(ImportError, match="fake.absent"):
        helpers.require_apis(module, "absent")

    monkeypatch.setattr(helpers, "BACKFILL_TARGET_SHA", "0123456789abcdef")
    with pytest.raises(pytest.skip.Exception, match="fake.absent"):
        helpers.require_apis(module, "absent")


def test_backfill_reports_required_device_failures_instead_of_skipping_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only optional device types are absorbed as backfill capability gaps.

    CUDA cannot be created on Windows before roughly 2026-02, so skipping it keeps the
    rest of a historical commit's benchmarks reportable. d3d12 exists on the perf
    runners across the whole supported range, so a build that cannot create one has
    regressed and must fail rather than quietly lose its coverage.
    """

    def refuse(**kwargs: Any) -> Any:
        raise RuntimeError("device creation failed")

    monkeypatch.setattr(helpers, "Device", refuse)
    monkeypatch.setattr(helpers, "BACKFILL_TARGET_SHA", "0123456789abcdef")
    monkeypatch.setattr(helpers, "BACKFILL_UNAVAILABLE_DEVICES", {})
    monkeypatch.setattr(helpers, "DEVICE_CACHE", {})
    monkeypatch.setattr(helpers, "SELECTED_DEVICE_TYPES", None)

    with pytest.raises(pytest.skip.Exception):
        helpers.get_device(spy.DeviceType.cuda)
    assert len(helpers.BACKFILL_UNAVAILABLE_DEVICES) == 1

    # Catching the skip explicitly rather than using pytest.raises: a skip raised here
    # would otherwise propagate and mark this test skipped, hiding the regression it
    # exists to catch.
    try:
        helpers.get_device(spy.DeviceType.d3d12)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"required device type was skipped instead of reported: {skipped}")
    except RuntimeError as error:
        assert "device creation failed" in str(error)
    else:
        pytest.fail("device creation was expected to fail")

    # The failure is reported, so it must not be recorded as an unavailable device
    # and suppress the same configuration for the rest of the run.
    assert len(helpers.BACKFILL_UNAVAILABLE_DEVICES) == 1
