# SPDX-License-Identifier: Apache-2.0
import pytest

import slangpy as kf
import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType

SIMPLE_FUNCTION_RETURN_VALUE = r"""
int add_numbers(int a, int b) {
    return a + b;
}
"""

SIMPLE_FUNCTION_IN_TYPE_RETURN_VALUE = r"""
struct MyStruct {
    int add_numbers(int a, int b) {
        return a + b;
    }
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_function(device_type: DeviceType):

    device = helpers.get_device(device_type)

    slang_module = device.load_module_from_source(
        "simple_function_return_value", SIMPLE_FUNCTION_RETURN_VALUE
    )
    module = kf.Module(slang_module)
    function = module.find_function("add_numbers")
    assert function is not None
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_in_type(device_type: DeviceType):

    device = helpers.get_device(device_type)

    slang_module = device.load_module_from_source(
        "simple_function_create_in_type", SIMPLE_FUNCTION_IN_TYPE_RETURN_VALUE
    )
    module = kf.Module(slang_module)
    function = module.find_function_in_struct("MyStruct", "add_numbers")
    assert function is not None
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_function_helper(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "add_numbers", SIMPLE_FUNCTION_RETURN_VALUE
    )
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_in_type_helper(device_type: DeviceType):

    device = helpers.get_device(device_type)

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "MyStruct.add_numbers", SIMPLE_FUNCTION_IN_TYPE_RETURN_VALUE
    )
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_missing_function(device_type: DeviceType):

    device = helpers.get_device(device_type)
    with pytest.raises(ValueError):
        function = helpers.create_function_from_module(
            device, "add_numbers_bla", SIMPLE_FUNCTION_RETURN_VALUE
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
