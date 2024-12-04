# SPDX-License-Identifier: Apache-2.0
import pytest
from sgl import TypeConformance

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType

CONFORMING_MODULE = r"""
import "slangpy";

interface IVal {
    int get();
}

struct Get10 : IVal {
    int get() {
        return 10;
    }
}

struct Get20 : IVal {
    int get() {
        return 20;
    }
}


int getval() {
    IVal val;
    return val.get();
}   
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_conformance(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "getval", CONFORMING_MODULE)

    res = function.type_conformances([TypeConformance("IVal", "Get10")])()
    assert res == 10

    res = function.type_conformances([TypeConformance("IVal", "Get20")])()
    assert res == 20


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_conformance_fail(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "getval", CONFORMING_MODULE)

    with pytest.raises(ValueError, match="Failed to create shader program"):
        res = function()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
