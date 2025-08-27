# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_and_destroy_device_via_del(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None
    del device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_and_destroy_device_via_none(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None
    device = None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_and_cleanup_in_order(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None

    module = device.load_module_from_source(
        module_name="module_from_source",
        source=r"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() {
        }
    """,
    )

    module = None
    device = None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_and_cleanup_in_reverse_order(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None

    module = device.load_module_from_source(
        module_name="module_from_source",
        source=r"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() {
        }
    """,
    )

    device = None
    module = None


def asserting_creation(device_type: spy.DeviceType):
    device = helpers.get_device(device_type, use_cache=False)
    assert device is not None

    module = device.load_module_from_source(
        module_name="module_from_source",
        source=r"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() {
        }
    """,
    )
    raise Exception("Test failed")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_and_cleanup_through_assert(device_type: spy.DeviceType):
    with pytest.raises(Exception):
        asserting_creation(device_type)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
