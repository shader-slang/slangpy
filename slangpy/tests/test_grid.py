# SPDX-License-Identifier: Apache-2.0
import random
import numpy as np
import pytest

from slangpy.backend import DeviceType, float3, int3, uint3
from slangpy.core.module import Module
from slangpy.experimental.gridarg import grid
from slangpy.tests import helpers
from slangpy.types.buffer import NDBuffer


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return


def grid_test(device_type: DeviceType, dims: int = 2, datatype: str = 'array', stride: int = 1, offset: int = 0, fixed_shape: bool = True):
    # Generate random shape and the arguments for numpy transpose
    random.seed(42)
    shape = tuple([random.randint(5, 15) for _ in range(dims)])
    transpose = tuple([i+1 for i in range(dims)]) + (0,)

    if datatype == 'vector':
        typename = f"int{dims}"
    else:
        typename = f"int[{dims}]"

    # Create function that just dumps input to output for correct sized int
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, f"""
{typename} get({typename} input) {{
    return input;
}}
"""
    )

    # Buffer for vector results
    res = NDBuffer(device, shape=shape, dtype=module.layout.find_type_by_name(typename))

    # Offset per dimension
    offsets = tuple([offset for s in shape])

    # Call function with grid as input argument
    if fixed_shape:
        if stride == 1:
            module.get(grid(shape, offset=offsets), _result=res)
        else:
            full_shape = tuple([s*stride for s in shape])
            strides = tuple([stride for s in shape])
            module.get(grid(full_shape, stride=strides, offset=offsets), _result=res)
    else:
        if stride == 1:
            module.get(grid(len(shape), offset=offsets), _result=res)
        else:
            strides = tuple([stride for s in shape])
            module.get(grid(len(shape), stride=strides, offset=offsets), _result=res)

    # Should get random numbers
    resdata = res.to_numpy().view(np.int32).reshape(shape + (dims,))
    expected = np.indices(shape).transpose(*transpose) * stride + offset

    assert np.all(resdata == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dims", [1, 2, 3, 4])
@pytest.mark.parametrize("stride", [1, 3])
def test_grid_vectors(device_type: DeviceType, dims: int, stride: int):
    if dims > 4:
        pytest.skip("Vector types only supported up to 4 dimensions")
    grid_test(device_type, dims=dims, datatype='vector', stride=stride)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dims", [1, 3, 6])
def test_grid_arrays(device_type: DeviceType, dims: int):
    grid_test(device_type, dims=dims, datatype='array')


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("offsets", [-100, 50])
def test_grid_offsets(device_type: DeviceType, offsets: int):
    grid_test(device_type, offset=offsets)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("stride", [1, 3])
def test_grid_implicit_shape(device_type: DeviceType, stride: int):
    grid_test(device_type, fixed_shape=False, stride=stride)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("stride", [1, 3])
def test_grid_error_pass_to_bad_vector_size(device_type: DeviceType, stride: int):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, f"""
int3 get(int3 input) {{
    return input;
}}
"""
                                   )
    with pytest.raises(ValueError, match="After implicit casting"):
        module.get(grid(shape=(2, 2)), _result='numpy')


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("stride", [1, 3])
def test_grid_error_pass_to_bad_array_size(device_type: DeviceType, stride: int):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, f"""
int3 get(int[3] input) {{
    return 0;
}}
"""
                                   )
    with pytest.raises(ValueError, match="After implicit casting"):
        module.get(grid(shape=(2, 2)), _result='numpy')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
