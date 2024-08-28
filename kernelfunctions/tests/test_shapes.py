import re
import pytest
from kernelfunctions.shapes import TLooseShape, calculate_argument_shapes
import deepdiff

# First set of tests emulate the shape of the following slang function
# float test(float3 a, float3 b) { return dot(a,b); }
# Note that the return value is simply treated as a final 'out' parameter
DOT_PRODUCT_SIGNATURE: list[TLooseShape] = [(3,), (3,), (1,)]

# Second set of tests emulate the shape of the following slang function,
# which has a 2nd parameter with with undefined dimension sizes
# float4 read(int2 index, Slice<2,float4> array) { return array[index];}
READ_SLICE_SIGNATURE: list[TLooseShape] = [(2,), (None, None, 4), (4,)]


def test_dotproduct_scalar():

    # really simple test case emulating slang function that takes
    # 2 x float3 and returns a float. Expecting a scalar call
    shapes = calculate_argument_shapes(DOT_PRODUCT_SIGNATURE, [(3,), (3,), None])
    diff = deepdiff.DeepDiff(
        shapes,
        {"type_shapes": [[3], [3], [1]], "arg_shapes": [[], [], []], "call_shape": []},
    )
    assert not diff


def test_dotproduct_broadcast_a():

    # emulates the same case but being passed a buffer for b
    shapes = calculate_argument_shapes(
        DOT_PRODUCT_SIGNATURE,
        [
            (3,),
            (
                100,
                3,
            ),
            None,
        ],
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[], [100], [100]],
            "call_shape": [100],
        },
    )
    assert not diff


def test_dotproduct_broadcast_b():

    # emulates the same case but being passed a buffer for a
    shapes = calculate_argument_shapes(
        DOT_PRODUCT_SIGNATURE,
        [
            (
                100,
                3,
            ),
            (3,),
            None,
        ],
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[100], [], [100]],
            "call_shape": [100],
        },
    )
    assert not diff


def test_dotproduct_shape_error():

    # attempt to pass a buffer of float4s for a, causes shape error
    with pytest.raises(ValueError, match=re.escape("Arg 0, PS[0] != IS[1], 3 != 4")):
        calculate_argument_shapes(
            DOT_PRODUCT_SIGNATURE,
            [
                (
                    100,
                    4,
                ),
                (3,),
                None,
            ],
        )


def test_dotproduct_broadcast_error():

    # attempt to pass missmatching buffer sizes for a and b
    with pytest.raises(
        ValueError, match=re.escape("Arg 1, CS[0] != AS[0], 100 != 1000")
    ):
        shapes = calculate_argument_shapes(
            DOT_PRODUCT_SIGNATURE,
            [
                (
                    100,
                    3,
                ),
                (
                    1000,
                    3,
                ),
                None,
            ],
        )


def test_dotproduct_broadcast_result():

    # pass an output, which is also broadcast so would in practice be a race condition
    shapes = calculate_argument_shapes(
        DOT_PRODUCT_SIGNATURE,
        [
            (
                100,
                3,
            ),
            (3,),
            (1,),
        ],
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[100], [], []],
            "call_shape": [100],
        },
    )
    assert not diff


def test_dotproduct_broadcast_invalid_result():

    # pass an output of the wrong shape resulting in error
    with pytest.raises(ValueError, match=re.escape("Arg 2, PS[0] != IS[0], 1 != 3")):
        shapes = calculate_argument_shapes(
            DOT_PRODUCT_SIGNATURE,
            [
                (
                    100,
                    3,
                ),
                (3,),
                (3,),
            ],
        )


def test_dotproduct_ambiguous_call_shape():

    # Passing buffer for result with undefined size. In principle
    # this would broadcast to each entry of the buffer, but because
    # the size is undefined it will raise an error
    with pytest.raises(ValueError, match=re.escape("Call shape is ambiguous: [None]")):
        shapes = calculate_argument_shapes(
            DOT_PRODUCT_SIGNATURE, [(3,), (3,), (None, 1)]
        )


def test_dotproduct_infer_buffer_size():

    # Passing buffer for result with undefined size. Because we
    # also pass a fixed size buffer for b, we can infer the call
    # shape, and thus the result buffer size
    shapes = calculate_argument_shapes(
        DOT_PRODUCT_SIGNATURE, [(3,), (100, 3), (None, 1)]
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[3], [3], [1]],
            "arg_shapes": [[], [100], [100]],
            "call_shape": [100],
        },
    )
    assert not diff


def test_readslice_scalar():

    # Scalar call to the read slice function, with a single index
    # and a single slice, and the result undefined.
    shapes = calculate_argument_shapes(
        READ_SLICE_SIGNATURE, [(2,), (256, 128, 4), None]
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[], [], []],
            "call_shape": [],
        },
    )
    assert not diff


def test_readslice_broadcast_slice():

    # Provide a buffer of 50 indices to sample against the 1 slice
    shapes = calculate_argument_shapes(
        READ_SLICE_SIGNATURE, [(50, 2), (256, 128, 4), None]
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[50], [], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


def test_readslice_broadcast_index():

    # Test the same index against 50 slices
    shapes = calculate_argument_shapes(
        READ_SLICE_SIGNATURE, [(2,), (50, 256, 128, 4), None]
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[], [50], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


def test_readslice_vectorcall():

    # Test the 50 indices against 50 slices
    shapes = calculate_argument_shapes(
        READ_SLICE_SIGNATURE, [(50, 2), (50, 256, 128, 4), None]
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[50], [50], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


def test_readslice_invalid_shape():

    # Fail trying to pass a float3 buffer into the float4 slice
    with pytest.raises(ValueError, match=re.escape("Arg 1, PS[2] != IS[3], 4 != 3")):
        shapes = calculate_argument_shapes(
            READ_SLICE_SIGNATURE, [(50, 2), (50, 256, 128, 3), None]
        )


def test_readslice_invalid_broadcast():

    # Fail trying to pass mismatched broadcast dimensions
    with pytest.raises(ValueError, match=re.escape("Arg 1, CS[0] != AS[0], 50 != 75")):
        shapes = calculate_argument_shapes(
            READ_SLICE_SIGNATURE, [(50, 2), (75, 256, 128, 4), None]
        )


def test_readslice_argument_map():

    # Use argument mapping to allow 50 (4,256,128) buffers to be
    # passed as 50 (256,128,4) slices
    shapes = calculate_argument_shapes(
        READ_SLICE_SIGNATURE,
        [(50, 2), (50, 4, 256, 128), None],
        [None, (0, 2, 3, 1), None],
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[50], [50], [50]],
            "call_shape": [50],
        },
    )
    assert not diff


def test_readslice_function_map():

    # Use remapping to allow 1000 indices to be batch tested
    # against 50*(4,256,128), resulting in output of 50*(1000)
    shapes = calculate_argument_shapes(
        READ_SLICE_SIGNATURE,
        [(1000, 2), (50, 256, 128, 4), None],
        None,
        [(1,), (0,), None],
    )
    diff = deepdiff.DeepDiff(
        shapes,
        {
            "type_shapes": [[2], [256, 128, 4], [4]],
            "arg_shapes": [[1000], [50], [50, 1000]],
            "call_shape": [50, 1000],
        },
    )
    assert not diff


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
