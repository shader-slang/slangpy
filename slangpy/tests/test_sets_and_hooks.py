from typing import Any
import pytest
from kernelfunctions.backend import DeviceType
from kernelfunctions.module import Module
import kernelfunctions.tests.helpers as helpers
from kernelfunctions.types.buffer import NDBuffer
import numpy as np

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
    val_data = np.random.rand(10).astype(np.float32)
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hook(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    add_k = m.add_k.as_func()

    val = NDBuffer(m.device, float, 10)
    val_data = np.random.rand(10).astype(np.float32)
    val.from_numpy(val_data)

    hooks_called = 0

    def before_call(args: dict[str, Any]):
        nonlocal hooks_called
        args['params'] = {
            'k': 10
        }
        hooks_called += 1

    def after_call(args: dict[str, Any]):
        nonlocal hooks_called
        assert args['params']['k'] == 10
        hooks_called += 1

    add_k = add_k.hook(before_dispatch=before_call, after_dispatch=after_call)

    res = add_k(val)

    res_data = res.to_numpy().view(dtype=np.float32)
    assert np.allclose(res_data, val_data + 10)
    assert hooks_called == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
