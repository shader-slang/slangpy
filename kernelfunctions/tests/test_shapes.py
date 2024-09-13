import re
from types import NoneType
from typing import Any, Optional
import pytest
from kernelfunctions.backend import DeviceType, float1, float3
from kernelfunctions.callsignature import CallMode, SignatureNode, apply_signature, build_signature, calculate_and_apply_call_shape, match_signature
from kernelfunctions.shapes import TLooseShape
import deepdiff

from kernelfunctions.tests import helpers
from kernelfunctions.typeregistry import PYTHON_TYPES
from kernelfunctions.types import floatRef
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.valueref import ValueRef

# Dummy class that fakes a buffer of a given shape for testing


class FakeBuffer:
    def __init__(self, shape: tuple[Optional[int], ...]):
        super().__init__()
        self.shape = shape


class FakeBufferType(BaseTypeImpl):
    def __init__(self):
        super().__init__()

    def has_derivative(self, value: Any = None) -> bool:
        return False

    def is_writable(self, value: Any) -> bool:
        return True

    def container_shape(self, value: FakeBuffer):
        return value.shape

    def shape(self, value: Any = None):
        return value.shape

    def element_type(self, value: Any):
        return PYTHON_TYPES[NoneType]


PYTHON_TYPES[FakeBuffer] = FakeBufferType()


# First set of tests emulate the shape of the following slang function
# float test(float3 a, float3 b) { return dot(a,b); }
# Note that the return value is simply treated as a final 'out' parameter
def dot_product(device_type: DeviceType, a: Any, b: Any, result: Any,
                input_transforms: Optional[dict[str, tuple[int, ...]]] = None,
                ouput_transforms: Optional[dict[str, tuple[int, ...]]] = None,
                ) -> Any:
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"float add_numbers(float3 a, float3 b) { return dot(a,b);}",
    )

    sig = build_signature(a=a, b=b, _result=result)
    match = match_signature(
        sig, function.overloads[0], CallMode.prim)
    assert match is not None
    tree = apply_signature(sig, match, CallMode.prim, input_transforms, ouput_transforms)
    call_shape = calculate_and_apply_call_shape(tree)

    nodes: list[SignatureNode] = []
    for node in tree.values():
        node.get_input_list(nodes)
    return {
        "call_shape": call_shape,
        "type_shapes": [x.type_shape for x in nodes],
        "arg_shapes": [x.argument_shape for x in nodes],
    }

# Second set of tests emulate the shape of the following slang function,
# which has a 2nd parameter with with undefined dimension sizes
# float4 read(int2 index, Slice<2,float4> array) { return array[index];}


def read_slice(device_type: DeviceType, index: Any, texture: Any, result: Any,
               input_transforms: Optional[dict[str, tuple[int, ...]]] = None,
               ouput_transforms: Optional[dict[str, tuple[int, ...]]] = None,
               ) -> Any:
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "read_slice",
        r"float4 read_slice(int2 index, Texture2D<float4> texture) { return texture[index]; }",
    )

    sig = build_signature(index=index, texture=texture, _result=result)
    match = match_signature(
        sig, function.overloads[0], CallMode.prim)
    assert match is not None
    tree = apply_signature(sig, match, CallMode.prim, input_transforms, ouput_transforms)
    call_shape = calculate_and_apply_call_shape(tree)

    nodes: list[SignatureNode] = []
    for node in tree.values():
        node.get_input_list(nodes)
    return {
        "call_shape": call_shape,
        "type_shapes": [x.type_shape for x in nodes],
        "arg_shapes": [x.argument_shape for x in nodes],
    }


# Copy function designed to replicate situations in which we'd ideally
# be able to infer a buffer size but can't due to absence of generics
# void copy(int index, Slice<1,float4> from, Slice<1,float4> to) { to[index] = from[index];}
COPY_AT_INDEX_SIGNATURE: list[TLooseShape] = [(1,), (None, 4), (None, 4)]


def copy_at_index(device_type: DeviceType, index: Any, frombuffer: Any, tobuffer: Any,
                  input_transforms: Optional[dict[str, tuple[int, ...]]] = None,
                  ouput_transforms: Optional[dict[str, tuple[int, ...]]] = None,
                  ) -> Any:
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "copy_at_index",
        r"void copy_at_index(int index, StructuredBuffer<float4> fr, RWStructuredBuffer<float4> to) { to[index] = fr[index]; }",
    )

    sig = build_signature(index=index, fr=frombuffer, to=tobuffer)
    match = match_signature(
        sig, function.overloads[0], CallMode.prim)
    assert match is not None
    tree = apply_signature(sig, match, CallMode.prim, input_transforms, ouput_transforms)
    call_shape = calculate_and_apply_call_shape(tree)

    nodes: list[SignatureNode] = []
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
        {"type_shapes": [[3], [3], [1]], "arg_shapes": [[], [], []], "call_shape": []},
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_scalar_floatref(device_type: DeviceType):

    # exactly the same but explicitly specifying a float ref for output
    shapes = dot_product(device_type, float3(), float3(), floatRef())
    diff = deepdiff.DeepDiff(
        shapes,
        {"type_shapes": [[3], [3], [1]], "arg_shapes": [[], [], []], "call_shape": []},
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_a(device_type: DeviceType):

    # emulates the same case but being passed a buffer for b
    shapes = dot_product(device_type, float3(), FakeBuffer((100, 3)), None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[], [100], [100]],
            "call_shape": [100],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_b(device_type: DeviceType):

    # emulates the same case but being passed a buffer for a
    shapes = dot_product(device_type, FakeBuffer((100, 3)), float3(), None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[100], [], [100]],
            "call_shape": [100],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_b_from_buffer(device_type: DeviceType):

    # similar, but broadcasting b out of a 1D buffer instead
    shapes = dot_product(device_type, FakeBuffer((100, 3)), FakeBuffer((1, 3)), None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[100], [1], [100]],
            "call_shape": [100],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_shape_error(device_type: DeviceType):

    # attempt to pass a buffer of float4s for a, causes shape error
    with pytest.raises(ValueError, match=re.escape("Arg 0, PS[0] != IS[1], 3 != 4")):
        dot_product(device_type, FakeBuffer((100, 4)), FakeBuffer((3,)), None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_error(device_type: DeviceType):

    # attempt to pass missmatching buffer sizes for a and b
    with pytest.raises(
        ValueError, match=re.escape("Arg 1, CS[0] != AS[0], 100 != 1000")
    ):
        dot_product(device_type, FakeBuffer((100, 3)), FakeBuffer((1000, 3)), None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_result(device_type: DeviceType):

    # pass an output, which is also broadcast so would in practice be a race condition
    shapes = dot_product(device_type, FakeBuffer(
        (100, 3)), FakeBuffer((3,)), ValueRef(float1()))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[100], [], []],
            "call_shape": [100],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_broadcast_invalid_result(device_type: DeviceType):

    # pass an output of the wrong shape resulting in error
    with pytest.raises(ValueError, match=re.escape("Arg -1, PS[0] != IS[0], 1 != 3")):
        shapes = dot_product(device_type, FakeBuffer((100, 3)),
                             FakeBuffer((3,)), FakeBuffer((3,)))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_ambiguous_call_shape(device_type: DeviceType):

    # Passing buffer for result with undefined size. In principle
    # this would broadcast to each entry of the buffer, but because
    # the size is undefined it will raise an error
    with pytest.raises(ValueError, match=re.escape("Call shape is ambiguous: [None]")):
        dot_product(device_type, FakeBuffer((3,)),
                    FakeBuffer((3,)), FakeBuffer((None, 1)))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_infer_buffer_size(device_type: DeviceType):

    # Passing buffer for result with undefined size. Because we
    # also pass a fixed size buffer for b, we can infer the call
    # shape, and thus the result buffer size
    shapes = dot_product(device_type, FakeBuffer(
        (3,)), FakeBuffer((100, 3)), FakeBuffer((None, 1)))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[], [100], [100]],
            "call_shape": [100],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_big_tensors(device_type: DeviceType):

    # Test some high dimensional tensors with some broadcasting
    shapes = dot_product(device_type, FakeBuffer((8, 1, 2, 3)),
                         FakeBuffer((8, 4, 2, 3)), FakeBuffer((8, 4, 2, 1)))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[8, 1, 2], [8, 4, 2], [8, 4, 2]],
            "call_shape": [8, 4, 2],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_input_transform(device_type: DeviceType):

    # Remapping inputs from big buffers
    shapes = dot_product(device_type,
                         FakeBuffer((8, 1, 2, 3)),
                         FakeBuffer((4, 8, 2, 3)),
                         None,
                         input_transforms={"b": (1, 0, 2, 3)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[8, 1, 2], [8, 4, 2], [8, 4, 2]],
            "call_shape": [8, 4, 2],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_output_transform(device_type: DeviceType):

    # Remapping outputs so buffers of length [10] and [5] can output [10,5]
    shapes = dot_product(device_type,
                         FakeBuffer((10, 3)),
                         FakeBuffer((5, 3)),
                         None,
                         ouput_transforms={
                             "a": (0,),
                             "b": (1,)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[10], [5], [10, 5]],
            "call_shape": [10, 5],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dotproduct_both_transform(device_type: DeviceType):

    # Combine simple input and output transforms
    shapes = dot_product(device_type,
                         FakeBuffer((3, 10)),
                         FakeBuffer((5, 3)),
                         None,
                         input_transforms={
                             "a": (1, 0)
                         },
                         ouput_transforms={
                             "a": (0,),
                             "b": (1,)})
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[10], [5], [10, 5]],
            "call_shape": [10, 5],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_scalar(device_type: DeviceType):

    # Scalar call to the read slice function, with a single index
    # and a single slice, and the result undefined.
    shapes = read_slice(device_type,
                        FakeBuffer((2, )),
                        FakeBuffer((256, 128, 4)),
                        None)
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[], [], []],
            "call_shape": [],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_broadcast_slice(device_type: DeviceType):

    # Provide a buffer of 50 indices to sample against the 1 slice
    shapes = read_slice(device_type,
                        FakeBuffer((50, 2)),
                        FakeBuffer((256, 128, 4)),
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_broadcast_index(device_type: DeviceType):

    # Test the same index against 50 slices
    shapes = read_slice(device_type,
                        FakeBuffer((2, )),
                        FakeBuffer((50, 256, 128, 4)),
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_vectorcall(device_type: DeviceType):

    # Test the 50 indices against 50 slices
    shapes = read_slice(device_type,
                        FakeBuffer((50, 2)),
                        FakeBuffer((50, 256, 128, 4)),
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
    with pytest.raises(ValueError, match=re.escape("Arg 1, PS[2] != IS[3], 4 != 3")):
        shapes = read_slice(device_type,
                            FakeBuffer((50, 2)),
                            FakeBuffer((50, 256, 128, 3)),
                            None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_invalid_broadcast(device_type: DeviceType):

    # Fail trying to pass mismatched broadcast dimensions
    with pytest.raises(ValueError, match=re.escape("Arg 1, CS[0] != AS[0], 50 != 75")):
        shapes = read_slice(device_type,
                            FakeBuffer((50, 2)),
                            FakeBuffer((75, 256, 128, 4)),
                            None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_argument_map(device_type: DeviceType):

    # Use argument mapping to allow 50 (4,256,128) buffers to be
    # passed as 50 (256,128,4) slices
    shapes = read_slice(device_type,
                        FakeBuffer((50, 2)),
                        FakeBuffer((50, 4, 256, 128)),
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_function_map(device_type: DeviceType):

    # Use remapping to allow 1000 indices to be batch tested
    # against 50*(4,256,128), resulting in output of 50*(1000)
    shapes = read_slice(device_type,
                        FakeBuffer((1000, 2)),
                        FakeBuffer((50, 256, 128, 4)),
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_readslice_both_map(device_type: DeviceType):

    # Use remapping to allow 1000 indices to be batch tested
    # against 50*(4,256,128), resulting in output of 50*(1000)
    shapes = read_slice(device_type,
                        FakeBuffer((2, 1000)),
                        FakeBuffer((50, 4, 256, 128)),
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
                           FakeBuffer((50, 1)),
                           FakeBuffer((100, 4)),
                           FakeBuffer((100, 4)))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[1], [100, 4], [100, 4]],
            "arg_shapes": [[50], [], []],
            "call_shape": [50],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copyatindex_undersized_output(device_type: DeviceType):

    # Situation we'd ideally detect in which output
    # buffer will overrun as its too small, but we
    # need generics/IBuffer to do so.
    shapes = copy_at_index(device_type,
                           FakeBuffer((50, 1)),
                           FakeBuffer((100, 4)),
                           FakeBuffer((10, 4)))
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[1], [100, 4], [10, 4]],
            "arg_shapes": [[50], [], []],
            "call_shape": [50],
        },
    )
    assert not diff


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copyatindex_undefined_output_size(device_type: DeviceType):

    # Output buffer size is undefined and can't be inferred.
    # This would ideally be solved with generics / IBuffer interface
    with pytest.raises(ValueError, match=re.escape("Arg 2 type shape is ambiguous")):
        shapes = copy_at_index(device_type,
                               FakeBuffer((50, 1)),
                               FakeBuffer((100, 4)),
                               FakeBuffer((None, 4)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
