from typing import Any, Optional
import pytest
from sgl import float4
from kernelfunctions.backend import DeviceType, float1, float3
from kernelfunctions.callsignature import CallMode, BoundVariable, bind, build_signature, calculate_call_shape, match_signatures
from kernelfunctions.shapes import TLooseShape
import deepdiff

from kernelfunctions.tests import helpers
from kernelfunctions.types import floatRef
from kernelfunctions.types.buffer import NDBuffer
from kernelfunctions.types.valueref import ValueRef
from kernelfunctions.tests.helpers import FakeBuffer

# First set of tests emulate the shape of the following slang function
# float test(float3 a, float3 b) { return dot(a,b); }
# Note that the return value is simply treated as a final 'out' parameter


def make_int_buffer(device_type: DeviceType, shape: tuple[int, ...]):
    return NDBuffer(device=helpers.get_device(device_type), shape=shape, element_type=int)


def make_float_buffer(device_type: DeviceType, shape: tuple[int, ...]):
    return NDBuffer(device=helpers.get_device(device_type), shape=shape, element_type=float)


def make_vec4_buffer(device_type: DeviceType, shape: tuple[int, ...]):
    return NDBuffer(device=helpers.get_device(device_type), shape=shape, element_type=float4)


def make_vec4_raw_buffer(device_type: DeviceType, count: int):
    nd = make_vec4_buffer(device_type, (count,))
    return nd.buffer


def dot_product(device_type: DeviceType, a: Any, b: Any, result: Any,
                transforms: Optional[dict[str, tuple[int, ...]]] = None,
                ) -> Any:
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"float add_numbers(float3 a, float3 b) { return dot(a,b);}",
    )

    if transforms is not None:
        function = function.transform_output(transforms)

    call_data = function._build_call_data(False, a=a, b=b, _result=result)
    call_data.call(a=a, b=b, _result=result)

    nodes: list[BoundVariable] = []
    for node in call_data.bindings.kwargs.values():
        node.get_input_list(nodes)
    return {
        "call_shape": call_data.call_shape,
        "node_call_dims": [x.call_dimensionality for x in nodes],
        "node_transforms": [x.transform for x in nodes],
        "python_dims": [x.python.dimensionality for x in nodes]
    }

# Second set of tests emulate the shape of the following slang function,
# which has a 2nd parameter with with undefined dimension sizes
# float4 read(int2 index, Slice<2,float4> array) { return array[index];}


def read_slice(device_type: DeviceType, index: Any, texture: Any, result: Any,
               transforms: Optional[dict[str, tuple[int, ...]]] = None,
               ) -> Any:
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "read_slice",
        r"""import "slangpy"; float read_slice(int2 index, NDBuffer<float,2> texture) { return texture[{index.x,index.y}]; }""",
    )

    if transforms is not None:
        function = function.transform_output(transforms)

    call_data = function._build_call_data(
        False, index=index, texture=texture, _result=result)
    call_data.call(index=index, texture=texture, _result=result)

    nodes: list[BoundVariable] = []
    for node in call_data.bindings.kwargs.values():
        node.get_input_list(nodes)
    return {
        "call_shape": call_data.call_shape,
        "node_call_dims": [x.call_dimensionality for x in nodes],
        "node_transforms": [x.transform for x in nodes],
        "python_dims": [x.python.dimensionality for x in nodes]
    }


# Copy function designed to replicate situations in which we'd ideally
# be able to infer a buffer size but can't due to absence of generics
# void copy(int index, Slice<1,float4> from, Slice<1,float4> to) { to[index] = from[index];}
COPY_AT_INDEX_SIGNATURE: list[TLooseShape] = [(1,), (None, 4), (None, 4)]


def copy_at_index(device_type: DeviceType, index: Any, frombuffer: Any, tobuffer: Any,
                  transforms: Optional[dict[str, tuple[int, ...]]] = None
                  ) -> Any:
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "copy_at_index",
        r"void copy_at_index(int index, StructuredBuffer<float4> fr, inout RWStructuredBuffer<float4> to) { to[index] = fr[index]; }",
    )

    if transforms is not None:
        function = function.transform_output(transforms)

    call_data = function._build_call_data(
        False, index=index, fr=frombuffer, to=tobuffer)
    call_data.call(index=index, fr=frombuffer, to=tobuffer)

    nodes: list[BoundVariable] = []
    for node in call_data.bindings.kwargs.values():
        node.get_input_list(nodes)
    return {
        "call_shape": call_data.call_shape,
        "node_call_dims": [x.call_dimensionality for x in nodes],
        "node_transforms": [x.transform for x in nodes],
        "python_dims": [x.python.dimensionality for x in nodes]
    }

    sig = build_signature(index=index, fr=frombuffer, to=tobuffer)
    match = match_signatures(
        sig, function.overloads[0], CallMode.prim)
    assert match is not None
    tree = bind(sig, match, CallMode.prim, input_transforms, ouput_transforms)
    call_shape = calculate_and_apply_call_shape(tree)

    nodes: list[BoundVariable] = []
    for node in tree.values():
        node.get_input_list(nodes)
    return {
        "call_shape": call_shape,
        "type_shapes": [x.type_shape for x in nodes],
        "arg_shapes": [x.argument_shape for x in nodes],
    }


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar(device_type: DeviceType):

    # really simple test case emulating slang function that takes
    # 2 x float3 and returns a float. Expecting a scalar call
    shapes = dot_product(device_type, float3(), float3(), None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [],
            "node_call_dims": [0, 0, 0],
            "node_transforms": [[0], [0], []],
            "python_dims": [1, 1, 0],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar_floatref(device_type: DeviceType):

    # exactly the same but explicitly specifying a float ref for output
    shapes = dot_product(device_type, float3(), float3(), floatRef())
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [],
            "node_call_dims": [0, 0, 0],
            "node_transforms": [[0], [0], []],
            "python_dims": [1, 1, 0],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_a(device_type: DeviceType):

    # emulates the same case but being passed a buffer for b
    shapes = dot_product(device_type, float3(),
                         make_float_buffer(device_type, (100, 3)), None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [100],
            "node_call_dims": [0, 1, 1],
            "node_transforms": [[1], [0, 1], [0]],
            "python_dims": [1, 2, 1],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_b(device_type: DeviceType):

    # emulates the same case but being passed a buffer for a
    shapes = dot_product(device_type, make_float_buffer(
        device_type, (100, 3)), float3(), None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [100],
            "node_call_dims": [1, 0, 1],
            "node_transforms": [[0, 1], [1], [0]],
            "python_dims": [2, 1, 1],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_b_from_buffer(device_type: DeviceType):

    # similar, but broadcasting b out of a 1D buffer instead
    shapes = dot_product(device_type, make_float_buffer(
        device_type, (100, 3)), make_float_buffer(device_type, (1, 3)), None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [100],
            "node_call_dims": [1, 1, 1],
            "node_transforms": [[0, 1], [0, 1], [0]],
            "python_dims": [2, 2, 1],
        }
    )
    assert not diff


@pytest.mark.skip("TODO: Catch this error")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_shape_error(device_type: DeviceType):

    # attempt to pass a buffer of float4s for a, causes shape error
    with pytest.raises(ValueError):
        dot_product(device_type, make_float_buffer(device_type, (100, 4)),
                    make_float_buffer(device_type, (3,)), None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_error(device_type: DeviceType):

    # attempt to pass missmatching buffer sizes for a and b
    with pytest.raises(ValueError):
        dot_product(device_type, make_float_buffer(device_type, (100, 3)),
                    make_float_buffer(device_type, (1000, 3)), None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_result(device_type: DeviceType):

    # pass an output, which is also broadcast so would in practice be a race condition
    shapes = dot_product(device_type, make_float_buffer(device_type,
                                                        (100, 3)), make_float_buffer(device_type, (3,)), ValueRef(float1()))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [100],
            "node_call_dims": [1, 0, 0],
            "node_transforms": [[0, 1], [1], []],
            "python_dims": [2, 1, 0],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_invalid_result(device_type: DeviceType):

    # pass an output of the wrong shape resulting in error
    with pytest.raises(ValueError):
        shapes = dot_product(device_type, make_float_buffer(device_type, (100, 3)),
                             make_float_buffer(device_type, (3,)), make_float_buffer(device_type, (3,)))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_big_tensors(device_type: DeviceType):

    # Test some high dimensional tensors with some broadcasting
    shapes = dot_product(device_type, make_float_buffer(device_type, (8, 1, 2, 3)),
                         make_float_buffer(device_type, (8, 4, 2, 3)), make_float_buffer(device_type, (8, 4, 2)))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [8, 4, 2],
            "node_call_dims": [3, 3, 3],
            "node_transforms": [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2]],
            "python_dims": [4, 4, 3],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_input_transform(device_type: DeviceType):

    # Remapping inputs from big buffers
    shapes = dot_product(device_type,
                         make_float_buffer(device_type, (8, 1, 2, 3)),
                         make_float_buffer(device_type, (4, 8, 2, 3)),
                         None,
                         transforms={"b": (1, 0, 2, 3)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [8, 4, 2],
            "node_call_dims": [3, 3, 3],
            "node_transforms": [[0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 2]],
            "python_dims": [4, 4, 3],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_output_transform(device_type: DeviceType):

    # Remapping outputs so buffers of length [10] and [5] can output [10,5]
    shapes = dot_product(device_type,
                         make_float_buffer(device_type, (10, 3)),
                         make_float_buffer(device_type, (5, 3)),
                         None,
                         transforms={
                             "a": (0, 2),
                             "b": (1, 2)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [10, 5],
            "node_call_dims": [2, 2, 2],
            "node_transforms": [[0, 2], [1, 2], [0, 1]],
            "python_dims": [2, 2, 2],
        }
    )
    assert not diff


@pytest.mark.skip(reason="Awaiting slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_scalar(device_type: DeviceType):

    # Scalar call to the read slice function, with a single index
    # and a single slice, and the result undefined.
    shapes = read_slice(device_type,
                        make_int_buffer(device_type, (2, )),
                        make_float_buffer(device_type, (256, 128, 4)),
                        None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [10, 5],
            "node_call_dims": [2, 2, None],
            "node_transforms": [[0, 2], [1, 2], [0, 1]],
            "python_dims": [2, 2, 2],
        }
    )
    assert not diff


@pytest.mark.skip(reason="Awaiting slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_broadcast_slice(device_type: DeviceType):

    # Provide a buffer of 50 indices to sample against the 1 slice
    shapes = read_slice(device_type,
                        make_float_buffer(device_type, (50, 2)),
                        make_float_buffer(device_type, (256, 128, 4)),
                        None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[50], [], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


@pytest.mark.skip(reason="Awaiting slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_broadcast_index(device_type: DeviceType):

    # Test the same index against 50 slices
    shapes = read_slice(device_type,
                        make_float_buffer(device_type, (2, )),
                        make_float_buffer(device_type, (50, 256, 128, 4)),
                        None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[], [50], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


@pytest.mark.skip(reason="Awaiting slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_vectorcall(device_type: DeviceType):

    # Test the 50 indices against 50 slices
    shapes = read_slice(device_type,
                        make_float_buffer(device_type, (50, 2)),
                        make_float_buffer(device_type, (50, 256, 128, 4)),
                        None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[50], [50], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_invalid_shape(device_type: DeviceType):

    # Fail trying to pass a float3 buffer into the float4 slice
    with pytest.raises(ValueError):
        shapes = read_slice(device_type,
                            make_float_buffer(device_type, (50, 2)),
                            make_float_buffer(device_type, (50, 256, 128, 3)),
                            None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_invalid_broadcast(device_type: DeviceType):

    # Fail trying to pass mismatched broadcast dimensions
    with pytest.raises(ValueError):
        shapes = read_slice(device_type,
                            make_float_buffer(device_type, (50, 2)),
                            make_float_buffer(device_type, (75, 256, 128, 4)),
                            None)


@pytest.mark.skip(reason="Awaiting slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_argument_map(device_type: DeviceType):

    # Use argument mapping to allow 50 (4,256,128) buffers to be
    # passed as 50 (256,128,4) slices
    shapes = read_slice(device_type,
                        make_float_buffer(device_type, (50, 2)),
                        make_float_buffer(device_type, (50, 4, 256, 128)),
                        None, input_transforms={"texture": (0, 2, 3, 1)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[50], [50], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


@pytest.mark.skip(reason="Awaiting slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_function_map(device_type: DeviceType):

    # Use remapping to allow 1000 indices to be batch tested
    # against 50*(4,256,128), resulting in output of 50*(1000)
    shapes = read_slice(device_type,
                        make_float_buffer(device_type, (1000, 2)),
                        make_float_buffer(device_type, (50, 256, 128, 4)),
                        None, ouput_transforms={"index": (1,), "texture": (0,)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[1000], [50], [50, 1000]],
            "call_shape": [50, 1000],
        },
    )
    assert not diff


@pytest.mark.skip(reason="Awaiting slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_both_map(device_type: DeviceType):

    # Use remapping to allow 1000 indices to be batch tested
    # against 50*(4,256,128), resulting in output of 50*(1000)
    shapes = read_slice(device_type,
                        make_float_buffer(device_type, (2, 1000)),
                        make_float_buffer(device_type, (50, 4, 256, 128)),
                        None,
                        input_transforms={"index": (1, 0), "texture": (0, 2, 3, 1)},
                        ouput_transforms={"index": (1,), "texture": (0,)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[1000], [50], [50, 1000]],
            "call_shape": [50, 1000],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copyatindex_both_buffers_defined(device_type: DeviceType):

    # Call copy-at-index passing 2 fully defined buffers
    shapes = copy_at_index(device_type,
                           make_int_buffer(device_type, (50,)),
                           make_vec4_raw_buffer(device_type, 100),
                           make_vec4_raw_buffer(device_type, 100))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [50],
            "node_call_dims": [1, 0, 0],
            "node_transforms": [[0], [1, 2], [1, 2]],
            "python_dims": [1, 2, 2],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copyatindex_undersized_output(device_type: DeviceType):

    # Situation we'd ideally detect in which output
    # buffer will overrun as its too small, but we
    # need generics/IBuffer to do so.
    shapes = copy_at_index(device_type,
                           make_int_buffer(device_type, (50,)),
                           make_vec4_raw_buffer(device_type, 100),
                           make_vec4_raw_buffer(device_type, 10))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "call_shape": [50],
            "node_call_dims": [1, 0, 0],
            "node_transforms": [[0], [1, 2], [1, 2]],
            "python_dims": [1, 2, 2],
        }
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copyatindex_undefined_output_size(device_type: DeviceType):

    # Output buffer size is undefined and can't be inferred.
    # This would ideally be solved with generics / IBuffer interface
    with pytest.raises(ValueError):
        shapes = copy_at_index(device_type,
                               make_int_buffer(device_type, (50,)),
                               make_vec4_raw_buffer(device_type, 100),
                               None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])