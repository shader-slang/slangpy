# SPDX-License-Identifier: Apache-2.0
import pytest

from slangpy.backend import DeviceType, int3, float3
from slangpy.tests import helpers
from slangpy.types.buffer import NDBuffer
import numpy as np

SIMPLE_FUNC = """
import "slangpy";
float foo(float a) { return a; }
float intfoo(int a) { return a; }
T genericfoo<T>(T a) { return a; }
T genericconstrainedfoo<T: IFloat>(T a) { return a; }
float3 foo3(float3 a) { return a; }
float add(float a, float b) { return a+b; }
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

    with pytest.raises(ValueError):
        call_data = function.debug_build_call_data(int3(1, 1, 1))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_1d_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    call_data = function.debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0,)
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_disabled_implicit_1d_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", SIMPLE_FUNC, options={
        'implicit_element_casts': False
    })

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    with pytest.raises(ValueError):
        call_data = function.debug_build_call_data(buffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_implicit_float_to_int_1d_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "intfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    with pytest.raises(ValueError):
        call_data = function.debug_build_call_data(buffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_explicit_map_float_to_int_1d_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "intfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    call_data = function.map((0,)).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0,)
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_explicit_cast_float_to_int_1d_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "intfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    call_data = function.map(float).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0,)
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_1d_explicit_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "genericfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    call_data = function.map((0,)).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0,)
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_genericconstrained_1d_explicit_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "genericconstrainedfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    call_data = function.map((0,)).debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0,)
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_genericconstrained_1d_explicit_typed_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "genericconstrainedfoo", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

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

    buffer = NDBuffer(device=device, dtype=float, shape=(10,))

    with pytest.raises(ValueError):
        call_data = function.debug_build_call_data(buffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_tensor_to_vector(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo3", SIMPLE_FUNC)

    buffer = NDBuffer(device=device, dtype=float, shape=(10, 3))

    call_data = function.debug_build_call_data(buffer)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0,)
    assert binding.vector_type is not None
    assert binding.vector_type.full_name == "vector<float,3>"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0,)
    assert result.vector_type is not None
    assert result.vector_type.full_name == "vector<float,3>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_disabled_implicit_tensor_to_vector(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo3", SIMPLE_FUNC, options={
        'implicit_tensor_casts': False
    })

    buffer = NDBuffer(device=device, dtype=float, shape=(10, 3))

    with pytest.raises(ValueError):
        call_data = function.debug_build_call_data(buffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_implicit_dimension_adding_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "add", SIMPLE_FUNC)

    a = NDBuffer(device=device, dtype=float, shape=(10, 10))
    b = NDBuffer(device=device, dtype=float, shape=(10,))

    with pytest.raises(Exception):
        call_data = function.debug_build_call_data(a, b)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_none_strict_implicit_dimension_adding_vectorization(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "add", SIMPLE_FUNC, options={
        'strict_broadcasting': False
    })

    a = NDBuffer(device=device, dtype=float, shape=(10, 10))
    b = NDBuffer(device=device, dtype=float, shape=(10,))

    call_data = function.debug_build_call_data(a, b)

    binding = call_data.debug_only_bindings.args[0]
    assert binding.vector_mapping.as_tuple() == (0, 1)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    binding = call_data.debug_only_bindings.args[1]
    assert binding.vector_mapping.as_tuple() == (1,)
    assert binding.vector_type is not None
    assert binding.vector_type.name == "float"

    result = call_data.debug_only_bindings.kwargs["_result"]
    assert result.vector_mapping.as_tuple() == (0, 1)
    assert result.vector_type is not None
    assert result.vector_type.name == "float"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_broadcast_vector(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "add", SIMPLE_FUNC)

    res_buffer = NDBuffer(device=device, dtype=float, shape=(3,))
    function(float3(1, 2, 3), float3(4, 5, 6), _result=res_buffer)
    assert np.allclose(res_buffer.to_numpy().view(dtype=np.float32), [5, 7, 9])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
