# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
import pytest

import slangpy.tests.helpers as helpers
from slangpy import Module
from slangpy.backend import DeviceType
from slangpy.types.buffer import NDBuffer
from slangpy.bindings import CallContext
from slangpy.core.function import Function

TEST_MODULE = r"""
import "slangpy";

struct Params {
    float k;
}
ParameterBlock<Params> params;

float add_k(float val) {
    return val + params.k.x;
}

"""


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module_from_source("test_sets_and_hooks.py", TEST_MODULE))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_set(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    add_k = m.add_k.as_func()

    val = NDBuffer(m.device, float, 10)
    val_data = np.zeros(10, dtype=np.float32)  # np.random.rand(10).astype(np.float32)
    val.from_numpy(val_data)

    add_k = add_k.set({'params': {
        'k': 10
    }})

    res = add_k(val)

    res_data = res.to_numpy().view(dtype=np.float32)
    assert np.allclose(res_data, val_data + 10)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_set_with_callback(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    add_k = m.add_k.as_func()

    val = NDBuffer(m.device, float, 10)
    val_data = np.random.rand(10).astype(np.float32)
    val.from_numpy(val_data)

    add_k = add_k.set(lambda x: {'params': {
        'k': 10
    }})

    res = add_k(val)

    res_data = res.to_numpy().view(dtype=np.float32)
    assert np.allclose(res_data, val_data + 10)


@pytest.mark.skip("Removed hooks")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hook(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    add_k = m.add_k.as_func()

    val = NDBuffer(m.device, float, 10)
    val_data = np.random.rand(10).astype(np.float32)
    val.from_numpy(val_data)

    hooks_called = 0

    def check_result(res: NDBuffer):
        res_data = res.to_numpy().view(dtype=np.float32)
        assert np.allclose(res_data, val_data + 10)

    def before_call(func: Function):
        nonlocal hooks_called
        assert func is add_k
        assert hooks_called == 0
        hooks_called += 1

    def before_write_call_data(ctx: CallContext, unpacked_args: tuple[Any], unpacked_kwargs: dict[str, Any]):
        nonlocal hooks_called
        assert len(unpacked_args) == 1
        assert len(unpacked_kwargs) == 1
        assert unpacked_args[0] is val
        assert '_result' in unpacked_kwargs
        assert hooks_called == 1
        hooks_called += 1

    def before_dispatch(args: dict[str, Any]):
        nonlocal hooks_called
        args['params'] = {
            'k': 10
        }
        assert hooks_called == 2
        hooks_called += 1

    def after_dispatch(args: dict[str, Any]):
        nonlocal hooks_called
        assert args['params']['k'] == 10
        assert hooks_called == 3
        hooks_called += 1

    def after_read_call_data(ctx: CallContext, unpacked_args: tuple[Any], unpacked_kwargs: dict[str, Any]):
        nonlocal hooks_called
        assert '_result' in unpacked_kwargs
        assert isinstance(unpacked_kwargs['_result'], NDBuffer)
        check_result(unpacked_kwargs['_result'])
        assert hooks_called == 4
        hooks_called += 1

    def after_call(func: Function):
        nonlocal hooks_called
        assert func is add_k
        assert hooks_called == 5
        hooks_called += 1

    add_k = add_k._internal_hook(before_dispatch=before_dispatch, after_dispatch=after_dispatch,
                                 before_write_call_data=before_write_call_data,
                                 after_read_call_data=after_read_call_data,
                                 before_call=before_call, after_call=after_call)

    res = add_k(val)

    check_result(res)
    assert hooks_called == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
