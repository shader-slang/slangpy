import pytest
import sgl
import kernelfunctions as kf
import kernelfunctions.tests.helpers as helpers

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
def test_create_function(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    module = device.load_module_from_source(
        "simple_function_return_value", SIMPLE_FUNCTION_RETURN_VALUE
    )
    function = kf.Function(module, "add_numbers")
    assert function.module == module
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_in_type(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    module = device.load_module_from_source(
        "simple_function_return_value", SIMPLE_FUNCTION_IN_TYPE_RETURN_VALUE
    )
    function = kf.Function(
        module,
        "add_numbers",
        module.module_decl.find_first_child_of_kind(
            sgl.DeclReflection.Kind.struct, "MyStruct"
        ),
    )
    assert function.module == module
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_function_helper(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "add_numbers", SIMPLE_FUNCTION_RETURN_VALUE
    )
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_in_type_helper(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "MyStruct.add_numbers", SIMPLE_FUNCTION_IN_TYPE_RETURN_VALUE
    )
    assert function.name == "add_numbers"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_missing_function(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    with pytest.raises(ValueError):
        function = helpers.create_function_from_module(
            device, "add_numbers_bla", SIMPLE_FUNCTION_RETURN_VALUE
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
