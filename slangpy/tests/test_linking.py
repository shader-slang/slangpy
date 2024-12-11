# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType

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


def load_test_module(device_type: DeviceType, link: list[Any] = []):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, BASE_MODULE, link=link)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_import_const(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = load_test_module(device_type, link=[
                         device.load_module_from_source("importmodule", IMPORT_MODULE)])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
