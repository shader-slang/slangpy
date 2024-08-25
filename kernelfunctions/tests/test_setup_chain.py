from typing import Any
import pytest
import kernelfunctions as kf
import sgl
import kernelfunctions.tests.helpers as helpers


SIMPLE_FUNCTION_RETURN_VALUE = r"""
int add_numbers(int a, int b) {
    return a + b;
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_build_chain_function_only(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "add_numbers", SIMPLE_FUNCTION_RETURN_VALUE
    )

    chain = function._build_call_data(False, 1, 1).chain
    assert len(chain) == 1
    assert isinstance(chain[0], kf.Function)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_build_chain_with_sets(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "add_numbers", SIMPLE_FUNCTION_RETURN_VALUE
    )
    s1 = function.set(a=1, b=2)
    s2 = s1.set(a=3)

    # Verify the function only has itself in the chain.
    func_call = function._build_call_data(False, 1, 1)
    func_chain = func_call.chain
    assert len(func_chain) == 1
    assert isinstance(func_chain[0], kf.Function)

    # Verify the chain for the function and s1.
    s1_call = s1._build_call_data(False, 1, 1)
    s1_chain = s1_call.chain
    assert len(s1_chain) == 2
    assert isinstance(s1_chain[0], kf.Function)
    assert isinstance(s1_chain[1], kf.function.FunctionChainSet)
    assert s1_chain[1].props == {"a": 1, "b": 2}

    # Verify the chain for the function, s1, and s2.
    s2_call = s2._build_call_data(False, 1, 1)
    s2_chain = s2_call.chain
    assert len(s2_chain) == 3
    assert isinstance(s2_chain[0], kf.Function)
    assert isinstance(s2_chain[1], kf.function.FunctionChainSet)
    assert s2_chain[1].props == {"a": 1, "b": 2}
    assert isinstance(s2_chain[2], kf.function.FunctionChainSet)
    assert s2_chain[2].props == {"a": 3}

    # Verify the function alone sets no globals
    sets = func_call.sets
    assert sets == {}

    # Verify the function and s1 sets the correct globals
    sets = s1_call.sets
    assert sets == {"a": 1, "b": 2}

    # Verify the function, s1, and s2 sets the correct globals
    sets = s2_call.sets
    assert sets == {"a": 3, "b": 2}


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_set_arguments(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "add_numbers", SIMPLE_FUNCTION_RETURN_VALUE
    )

    function.set(a=1, b=2)
    function.set(a=3)
    function.set({"a": 1, "b": 2})
    function.set(lambda: {"a": 1, "b": 2})

    with pytest.raises(ValueError):
        function.set(0)
    with pytest.raises(ValueError):
        function.set(0, 1)
    with pytest.raises(ValueError):
        function.set([10, 20])
    with pytest.raises(ValueError):
        function.set({}, {})
    with pytest.raises(ValueError):
        function.set(lambda: 0, {})


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_set_lambda_callback(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "add_numbers", SIMPLE_FUNCTION_RETURN_VALUE
    )

    chain = function.set(a=1, b=2).set(lambda chain: {"a": 10}).set(b=3)

    call_data = chain._build_call_data(False, 1, 1)
    assert call_data.sets == {"a": 10, "b": 3}


class InstanceTest:
    def __init__(self, a: Any, b: Any):
        super().__init__()
        self.a = a
        self.b = b

    def get_values(self, calldata: kf.CallData):
        assert calldata.function.name == "add_numbers"
        return {"a": self.a, "b": self.b}


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_set_instance_callback(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "add_numbers", SIMPLE_FUNCTION_RETURN_VALUE
    )

    instance = InstanceTest(10, 20)

    chain = function.set(a=1, b=2).set(instance.get_values).set(b=3)

    call_data = chain._build_call_data(False, 1, 1)
    assert call_data.sets == {"a": 10, "b": 3}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
