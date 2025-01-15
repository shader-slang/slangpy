# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType
from slangpy.types.buffer import NDBuffer

BASE_MODULE = r"""
import "slangpy";
float foo(float a, float b) { 
    return a+b; 
}
"""


def load_test_module(device_type: DeviceType, link: list[Any] = []):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, BASE_MODULE, link=link)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_types(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = load_test_module(device_type)
    assert m is not None

    func = m.foo.as_func()

    scalar_float_float_cd = func.debug_build_call_data(1.0, 2.0)
    scalar_float_float_cd_2 = func.debug_build_call_data(1.0, 2.0)
    scalar_int_int_cd = func.debug_build_call_data(1, 2)
    scalar_int_int_cd_2 = func.debug_build_call_data(1, 2)
    scalar_int_float_cd = func.debug_build_call_data(1, 2.0)

    assert scalar_float_float_cd == scalar_float_float_cd_2
    assert scalar_int_int_cd == scalar_int_int_cd_2
    assert scalar_float_float_cd != scalar_int_int_cd
    assert scalar_int_float_cd != scalar_int_int_cd


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_share_kernel_with_same_mapping(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = load_test_module(device_type)
    assert m is not None

    func = m.foo.as_func()

    b0 = NDBuffer(device, program_layout=m.layout, dtype=float, shape=(100,))
    b1 = NDBuffer(device, program_layout=m.layout, dtype=float, shape=(100,))

    float_float_cd = func.debug_build_call_data(b0, b1)
    mapped_float_float_cd = func.map((0,), (0,)).debug_build_call_data(b0, b1)
    assert float_float_cd != mapped_float_float_cd
    assert float_float_cd.kernel == mapped_float_float_cd.kernel


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
