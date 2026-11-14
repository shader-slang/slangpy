# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from slangpy import DeviceType, TypeConformance
from slangpy.testing import helpers

CONFORMING_MODULE = r"""
import "slangpy";

public interface IVal {
    int get();
}

public struct Get10 : IVal {
    public int get() {
        return 10;
    }
}

public struct Get20 : IVal {
    public int get() {
        return 20;
    }
}


public int getval() {
    IVal val = createDynamicObject<IVal>(0, 0);
    return val.get();
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_conformance(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "getval", CONFORMING_MODULE)

    res = function.type_conformances([TypeConformance("IVal", "Get10")])()
    assert res == 10

    res = function.type_conformances([TypeConformance("IVal", "Get20")])()
    assert res == 20


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_conformance_fail(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "getval", CONFORMING_MODULE)

    with pytest.raises(RuntimeError):
        res = function()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
