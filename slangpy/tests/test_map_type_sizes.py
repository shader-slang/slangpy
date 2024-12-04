# SPDX-License-Identifier: Apache-2.0
import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType


@pytest.mark.skip(reason="Not implemented yet")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_explicit_add_int64s(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int64_t add_numbers(int64_t a, int64_t b) {
    return a+b;
}
""",
    )

    res = function.map('int64_t', 'int64_t')(5000000000, 10000000000)

    assert res == 15000000000


@pytest.mark.skip(reason="Not implemented yet")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_add_int64s(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int64_t add_numbers(int64_t a, int64_t b) {
    return a+b;
}
""",
    )

    res = function(5000000000, 10000000000)

    assert res == 15000000000


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_add_uint32s(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
uint32_t add_numbers(uint32_t a, uint32_t b) {
    return a+b;
}
""",
    )

    res = function(0x8000000, 1)

    assert res == 0x8000001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
