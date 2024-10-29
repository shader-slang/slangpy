from typing import Any
import pytest
from sgl import float3, int3
from kernelfunctions.backend import DeviceType, TypeReflection
from kernelfunctions import Module
from kernelfunctions.tests import helpers
from kernelfunctions.types.buffer import NDBuffer

SIMPLE_FUNC = """
import "slangpy";
float foo(float a) { return a; }
float intfoo(int a) { return a; }
T genericfoo<T>(T a) { return a; }
T genericconstrainedfoo<T: IFloat>(T a) { return a; }
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_explicit_no_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", SIMPLE_FUNC)

    call_data = function.map(()).debug_build_call_data(10.0)
    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == ()
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_explicit_type_no_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", SIMPLE_FUNC)

    call_data = function.map("float").debug_build_call_data(10.0)
    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == ()
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_no_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", SIMPLE_FUNC)

    call_data = function.debug_build_call_data(10.0)
    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == ()
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_cast_no_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "intfoo", SIMPLE_FUNC)

    call_data = function.debug_build_call_data(10.0)
    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == ()
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_no_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "genericfoo", SIMPLE_FUNC)

    call_data = function.debug_build_call_data(10.0)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == ()
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_constrained_no_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "genericconstrainedfoo", SIMPLE_FUNC)

    call_data = function.debug_build_call_data(10.0)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == ()
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_constrained_fail_no_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "genericconstrainedfoo", SIMPLE_FUNC)

    call_data = function.debug_build_call_data(int3(1, 1, 1))

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == ()
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_1d_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, element_type=float, shape=(10,))

    call_data = function.map((0,)).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_cast_1d_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "intfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, element_type=float, shape=(10,))

    call_data = function.map((0,)).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_1d_explicit_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "genericfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, element_type=float, shape=(10,))

    call_data = function.map((0,)).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_genericconstrained_1d_explicit_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "genericconstrainedfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, element_type=float, shape=(10,))

    call_data = function.map((0,)).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_genericconstrained_1d_explicit_typed_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "genericconstrainedfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, element_type=float, shape=(10,))

    call_data = function.map("float").debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0,)
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_genericconstrained_1d_fail_implicit_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "genericconstrainedfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, element_type=float, shape=(10,))

    call_data = function.debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == ()
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
