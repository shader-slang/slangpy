# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import random
import numpy as np

from slangpy import DeviceType
from slangpy.experimental.gridarg import grid
from slangpy.types.buffer import NDBuffer
from slangpy.testing import helpers


def grid_test(
    device_type: DeviceType,
    dims: int = 2,
    datatype: str = "array",
    stride: int = 1,
    offset: int = 0,
    fixed_shape: bool = True,
):
    # Generate random shape and the arguments for numpy transpose
    random.seed(42)
    shape = tuple([random.randint(5, 15) for _ in range(dims)])
    transpose = tuple([i + 1 for i in range(dims)]) + (0,)
    gen_args = ""

    if datatype == "vector":
        buffertypename = f"int{dims}"
        slangtypename = buffertypename
    elif datatype == "array":
        buffertypename = f"int[{dims}]"
        slangtypename = buffertypename
    elif datatype == "genvector":
        buffertypename = f"int{dims}"
        slangtypename = f"vector<int, N>"
        gen_args = "<let N: int>"
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    # Create function that just dumps input to output for correct sized int
    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device,
        f"""
{slangtypename} get{gen_args}({slangtypename} input) {{
    return input;
}}
""",
    )

    # Buffer for vector results
    res = NDBuffer(device, shape=shape, dtype=module.layout.find_type_by_name(buffertypename))

    # Offset per dimension
    offsets = tuple([offset for s in shape])

    # Call function with grid as input argument
    if fixed_shape:
        if stride == 1:
            module.get(grid(shape, offset=offsets), _result=res)
        else:
            strides = tuple([stride for s in shape])
            module.get(grid(shape, stride=strides, offset=offsets), _result=res)
    else:
        if stride == 1:
            module.get(grid(len(shape), offset=offsets), _result=res)
        else:
            strides = tuple([stride for s in shape])
            module.get(grid(len(shape), stride=strides, offset=offsets), _result=res)

    # Should get random numbers
    resdata = res.to_numpy().view(np.int32).reshape(shape + (dims,))
    expected = np.indices(shape).transpose(*transpose) * stride + offset

    if datatype == "vector":
        expected = np.flip(expected, axis=-1)

    assert np.all(resdata == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dims", [1, 2, 3, 4])
@pytest.mark.parametrize("stride", [1, 3])
def test_grid_vectors(device_type: DeviceType, dims: int, stride: int):
    grid_test(device_type, dims=dims, datatype="vector", stride=stride)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_grid_generic_vectors(device_type: DeviceType):
    pytest.skip("Doesn't currently work due to handling of generic arguments in specialize")
    grid_test(device_type, dims=3, datatype="genvector")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dims", [1, 3, 6])
def test_grid_arrays(device_type: DeviceType, dims: int):
    grid_test(device_type, dims=dims, datatype="array")


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
    module = helpers.create_module(
        device,
        f"""
int3 get(int3 input) {{
    return input;
}}
""",
    )
    with pytest.raises(ValueError, match="Could not find suitable"):
        module.get(grid(shape=(2, 2)), _result="numpy")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("stride", [1, 3])
def test_grid_error_pass_to_bad_array_size(device_type: DeviceType, stride: int):
    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device,
        f"""
int3 get(int[3] input) {{
    return 0;
}}
""",
    )
    with pytest.raises(ValueError, match="Could not find suitable"):
        module.get(grid(shape=(2, 2)), _result="numpy")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
