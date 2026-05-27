# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from slangpy import DeviceType
from slangpy.testing import helpers

from typing import Any

BASE_MODULE = r"""
import "slangpy";
extern static const bool BOOL;
extern static const int INT;
extern static const float FLOAT;
float foo() { return BOOL ? float(INT) * FLOAT : 0.0; }
"""

IMPORT_MODULE = r"""
export static const bool BOOL = true;
export static const int INT = 10;
export static const float FLOAT = 42.0;
"""

LINK_TYPE_BASE_MODULE = r"""
interface IVal
{
    static int get();
}

extern struct VAL : IVal;

int get_value()
{
    return VAL::get();
}
"""


def load_test_module(device_type: DeviceType, link: list[Any] = []):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, BASE_MODULE, link=link)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_import_const(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = load_test_module(
        device_type,
        link=[device.load_module_from_source("importmodule", IMPORT_MODULE)],
    )
    assert m is not None

    res = m.foo()
    assert res == 420.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_define_const(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = load_test_module(device_type)
    assert m is not None

    res = m.foo.constants({"BOOL": True, "INT": 20, "FLOAT": 15.0})()
    assert res == 300.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_prelude_defines_link_time_type(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = helpers.create_module(device, LINK_TYPE_BASE_MODULE)

    val_10_prelude = r"""
struct Val10 : IVal
{
    static int get() { return 10; }
}

export struct VAL : IVal = Val10;
"""

    val_20_prelude = r"""
struct Val20 : IVal
{
    static int get() { return 20; }
}

export struct VAL : IVal = Val20;
"""

    assert m.get_value.prelude(val_10_prelude)() == 10
    assert m.get_value.prelude(val_20_prelude)() == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
