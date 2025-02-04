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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dims", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("datatype", ['vector', 'array'])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("offset", [0, -100])
def test_grid_vector(device_type: DeviceType, dims: int, datatype: str, stride: int, offset: int):

    if datatype == 'vector' and dims > 4:
        pytest.skip("Vector types only supported up to 4 dimensions")

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
    if stride == 1:
        module.get(grid(shape, offset=offsets), _result=res)
    else:
        full_shape = tuple([s*stride for s in shape])
        strides = tuple([stride for s in shape])
        module.get(grid(full_shape, stride=strides, offset=offsets), _result=res)

    # Should get random numbers
    resdata = res.to_numpy().view(np.int32).reshape(shape + (dims,))
    expected = np.indices(shape).transpose(*transpose) * stride + offset

    assert np.all(resdata == expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
