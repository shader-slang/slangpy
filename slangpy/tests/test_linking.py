from typing import Any
import pytest
from kernelfunctions.backend import DeviceType
import kernelfunctions.tests.helpers as helpers

BASE_MODULE = r"""
import "slangpy";
extern static const float VAL;
float foo() { return VAL; }
"""

IMPORT_MODULE = r"""
export static const float VAL = 42.0;
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
    assert res == 42.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_define_const(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = load_test_module(device_type)
    assert m is not None

    res = m.foo.constants({"VAL": 15.0})()
    assert res == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
