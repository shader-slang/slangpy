# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType
from slangpy.types.buffer import NDBuffer

import numpy as np

from slangpy.types.tensor import Tensor


def do_generic_test(device_type: DeviceType, container_type: str, slang_type_name: str, generic_args: str, buffer_type_name: str):
    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, f"""
{slang_type_name} get{generic_args}({slang_type_name} input) {{
    return input;
}}
"""
    )

    shape = (1024,)
    buffertype = module.layout.find_type_by_name(buffer_type_name)

    if container_type == 'buffer':
        buffer = NDBuffer(device, dtype=buffertype, shape=shape)
        buffer.copy_from_numpy(np.random.random(int(buffer.storage.size / 4)).astype(np.float32))
        results = module.get(buffer)
        assert results.dtype == buffer.dtype
        assert np.all(buffer.to_numpy() == results.to_numpy())
    elif container_type == 'tensor':
        tensor = Tensor.empty(device, dtype=buffertype, shape=shape)
        results = module.get(tensor, _result='tensor')
        assert results.dtype == tensor.dtype
        assert np.all(tensor.to_numpy() == results.to_numpy())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ['buffer', 'tensor'])
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_generic_vector(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(device_type, container_type, 'vector<float,N>', "<let N: int>", f'float{dim}')


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ['buffer', 'tensor'])
@pytest.mark.parametrize("dim", [2])
def test_generic_array(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(device_type, container_type, 'float[N]', "<let N: int>", f'float[{dim}]')


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ['buffer', 'tensor'])
@pytest.mark.parametrize("dim", [2])
def test_generic_2d_array(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(device_type, container_type,
                    'float[N][N]', "<let N: int>", f'float[{dim}][{dim}]')


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("container_type", ['buffer', 'tensor'])
@pytest.mark.parametrize("dim", [2])
def test_generic_matrix(device_type: DeviceType, container_type: str, dim: int):
    do_generic_test(device_type, container_type, 'matrix<float,N,N>',
                    "<let N: int>", f'float{dim}x{dim}')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
