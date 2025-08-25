# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from slangpy import float4
from slangpy.bindings.boundvariable import BoundVariableException
from slangpy import DeviceType
from slangpy.types.buffer import NDBuffer
from slangpy.testing import helpers

MODULE = """
import "slangpy";
float foo(float a) { return a; }
float foo2(float a, float b) { return a+b; }
float foo_v3(float3 a) { return a.x; }
float foo_ol(float a) { return a; }
float foo_ol(float a, float b) { return a+b; }
float foo_generic<T>(T a) { return 0; }

"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_no_matching_arg_count(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", MODULE)

    with pytest.raises(Exception, match=r"Too many positional arguments"):
        function.call(1.0, 2.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_no_matching_arg_name(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", MODULE)

    with pytest.raises(Exception, match=r"No parameter named"):
        function.call(b=10.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_not_enough_args(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", MODULE)

    # note: due to no implicit args, falls straight through to slang resolution which provides
    # no special error info yet
    with pytest.raises(Exception, match=r"No Slang overload found"):
        function.call()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_not_enough_args_2(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)

    with pytest.raises(Exception, match=r"all parameters must be specified"):
        function.call(10.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specify_twice(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)

    with pytest.raises(Exception, match=r"already specified"):
        function.call(10.0, a=20.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_implicit_overload(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo_ol", MODULE)

    with pytest.raises(Exception, match=r"overloaded function with named or implicit arguments"):
        function.call(10.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_none(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo_ol", MODULE)

    # Currently hits error as 'None' does not implement reduce
    with pytest.raises(
        ValueError,
        match=r"Explicit vectorization raised exception\: NotImplementedError",
    ):
        function.map(()).call(None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_string(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo", MODULE)

    # Currently hits error as 'None' does not implement reduce
    with pytest.raises(BoundVariableException, match=r"Unsupported type"):
        function.call("hello")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bad_implicit_buffer_cast(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo_v3", MODULE)

    buffer = NDBuffer(device, dtype=float4, shape=(10,))

    # fail to specialize a float3 against a float
    with pytest.raises(ValueError, match=r"After implicit casting.*"):
        function(buffer)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_invalid_broadcast(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "foo2", MODULE, options={"strict_broadcasting": True}
    )

    buffer = NDBuffer(device, dtype=float, shape=(10,))
    buffer2 = NDBuffer(device, dtype=float, shape=(10, 10))

    # fail to specialize a float3 against a float
    with pytest.raises(ValueError, match=r"Strict broadcasting is enabled"):
        function(buffer, buffer2)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_invalid_broadcast_during_dispatch(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)

    buffer = NDBuffer(device, dtype=float, shape=(10, 5))
    buffer2 = NDBuffer(device, dtype=float, shape=(10, 10))

    # fail to specialize a float3 against a float
    with pytest.raises(ValueError, match=r"Shape mismatch"):
        function(buffer, buffer2)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_missing_function(device_type: DeviceType):

    device = helpers.get_device(device_type)
    module = helpers.create_module(device, """void hello() {}""")

    with pytest.raises(AttributeError, match=r"has no function or type named"):
        func = module.foo


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_missing_child_function(device_type: DeviceType):

    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device,
        """
                                   struct Foo {}
                                   void hello() {}""",
    )

    with pytest.raises(AttributeError, match=r"has no method or sub-type named"):
        func = module.Foo.foo  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
