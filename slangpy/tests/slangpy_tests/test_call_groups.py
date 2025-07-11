# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests for call group functionality in SlangPy.

This module tests:
1. Call group shapes for dimensions 1-5
2. Shader functions: get_call_id(), get_call_group_id(), get_call_group_thread_id()
3. Edge cases: misaligned shapes, shapes smaller than groups, etc.
4. Function existence and specific validation tests
"""

import numpy as np
import pytest
import slangpy as spy
import math
from slangpy import DeviceType
from slangpy.slangpy import Shape
from slangpy.types.buffer import NDBuffer
from . import helpers


# Test data for different call group shapes.
# Includes some common cases and edge cases that are
# all expected to work.
VALID_CALL_GROUP_SHAPE_TEST_CASES = [
    # (call_shape, call_group_shape, description)
    ((8,), (2,), "1D basic aligned"),
    ((9,), (2,), "1D misaligned"),
    ((1,), (4,), "1D smaller than group"),
    ((8, 6), (2, 3), "2D basic aligned"),
    ((9, 7), (2, 3), "2D misaligned"),
    ((1, 1), (4, 4), "2D smaller than group"),
    ((8, 6, 4), (2, 3, 2), "3D basic aligned"),
    ((9, 7, 5), (2, 3, 2), "3D misaligned"),
    ((8, 6, 4, 4), (2, 3, 2, 2), "4D basic aligned"),
    ((9, 7, 5, 3), (2, 3, 2, 2), "4D misaligned"),
    ((8, 6, 4, 4, 2), (2, 3, 2, 2, 2), "5D basic aligned"),
    ((9, 7, 5, 3, 3), (2, 3, 2, 2, 2), "5D misaligned"),
    ((32,), (1,), "1D linear dispatch (group size 1)"),
    ((32, 32), (32, 1), "2D linear first dimension"),
    ((32, 32), (1, 32), "2D linear second dimension"),
    ((100, 4, 2), (32, 2, 2), "3D with varying dimensions"),
    ((3, 5), (10, 10), "2D call smaller than group in all dims"),
    ((8, 4, 12), (4, 8, 6), "3D call smaller in some dims"),
    ((4, 4), (), "Empty shape should be valid - gets padded to [1, 1]"),
    (
        (4, 4),
        (4),
        "Call group shape with smaller dimensionality than call shape should be valid. Group shape gets padded.",
    ),
]

# Test data for call group shapes that are expected to
# fail with an error message.
INVALID_CALL_GROUP_SHAPE_TEST_CASES = [
    # (test_name, call_shape, call_group_shape, expected_error_fragments, description)
    (
        "zero_value",
        (4, 4),
        (2, 0),
        ["call_group_shape[1] = 0", "must be >= 1"],
        "Zero in call_group_shape should fail",
    ),
    (
        "negative_value",
        (4, 4),
        (2, -1),
        ["call_group_shape[1] = -1", "must be >= 1"],
        "Negative in call_group_shape should fail",
    ),
    (
        "mixed_invalid",
        (4, 4),
        (0, -1),
        ["call_group_shape[0] = 0", "must be >= 1"],
        "Mixed invalid values - first invalid caught",
    ),
    (
        "too_many_dims_2d",
        (4, 4),
        (2, 2, 2),
        [
            "call_group_shape dimensionality (3)",
            "call_shape dimensionality (2)",
            "cannot have more dimensions than call_shape",
        ],
        "3D call_group_shape for 2D call_shape should fail",
    ),
    (
        "too_many_dims_1d",
        (8,),
        (1, 1, 1, 1, 1),
        ["call_group_shape dimensionality (5)", "call_shape dimensionality (1)"],
        "5D call_group_shape for 1D call_shape should fail",
    ),
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_1d_call_groups_with_validation(device_type: DeviceType):
    """Test 1D call groups with specific mathematical validation."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

uint test_1d_groups(uint grid_cell) {
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();

    // Return packed result: high 16 bits = group_id, low 16 bits = thread_id
    return (call_group_id.shape[0] << 16) | call_group_thread_id.shape[0];
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test with call shape (8,) and call group shape (2,)
    call_shape = (8,)
    call_group_shape = (2,)

    result = module.test_1d_groups.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result="numpy"
    )

    # Extract group IDs and thread IDs
    group_ids = (result >> 16) & 0xFFFF
    thread_ids = result & 0xFFFF

    # Validate dimensions
    assert result.shape == call_shape

    # Basic validation: group IDs should be in range [0, ceil(8/2)-1] = [0, 3]
    assert np.all(group_ids >= 0) and np.all(group_ids < 4)

    # Thread IDs should be in range [0, 1] (group size is 2)
    assert np.all(thread_ids >= 0) and np.all(thread_ids < 2)

    # Validate that each call group has exactly one thread with thread_id [0]
    expected_groups = 4  # ceil(8/2) = 4 groups
    for group_id in range(expected_groups):
        zero_count = np.sum((group_ids == group_id) & (thread_ids == 0))
        assert (
            zero_count == 1
        ), f"Call group {group_id} should have exactly 1 thread with thread_id [0], found {zero_count}"

    # Ensure we have a variety of values, not all zeros
    assert np.max(group_ids) > 0, "Expected some non-zero group IDs"
    assert np.max(thread_ids) > 0, "Expected some non-zero thread IDs"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_math_2d_validation(device_type: DeviceType):
    """
    Test call group builtin functions and their mathematical relationships for 2D.

    This test validates:
    1. get_call_group_id() returns valid group indices
    2. get_call_group_thread_id() returns valid thread indices within groups
    3. Mathematical consistency between call_id, call_group_id, and call_group_thread_id
    4. Proper bounds checking for different call group configurations
    """

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

uint test_call_group_math_2d(uint2 grid_cell) {
    CallShapeInfo call_id = CallShapeInfo::get_call_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();

    // Pack the results:
    // Bits 24-31: call_group_id[0] (Y)
    // Bits 16-23: call_group_id[1] (X)
    // Bits 8-15: call_group_thread_id[0] (Y)
    // Bits 0-7: call_group_thread_id[1] (X)
    return (call_group_id.shape[0] << 24) | (call_group_id.shape[1] << 16) |
           (call_group_thread_id.shape[0] << 8) | call_group_thread_id.shape[1];
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test with multiple configurations
    test_cases = [
        ((4, 4), (2, 2)),  # Perfect alignment
        ((6, 4), (2, 2)),  # Also perfectly aligned
        ((8, 6), (2, 3)),  # Different group aspect ratio
        ((5, 7), (2, 2)),  # Unaligned case - will be padded to (6, 8)
    ]

    for call_shape, group_shape in test_cases:
        result = module.test_call_group_math_2d.call_group_shape(Shape(group_shape))(
            spy.grid(call_shape), _result="numpy"
        )

        # Extract packed values
        call_group_id_y = (result >> 24) & 0xFF
        call_group_id_x = (result >> 16) & 0xFF
        call_group_thread_id_y = (result >> 8) & 0xFF
        call_group_thread_id_x = result & 0xFF

        # Calculate expected grid dimensions (aligned up to group boundaries)
        grid_shape_y = math.ceil(call_shape[0] / group_shape[0])
        grid_shape_x = math.ceil(call_shape[1] / group_shape[1])

        # Validate bounds for actual call shape (not padded)
        for y in range(call_shape[0]):
            for x in range(call_shape[1]):
                # Validate call_group_id bounds
                assert (
                    0 <= call_group_id_y[y, x] < grid_shape_y
                ), f"call_group_id[0] out of bounds at [{y},{x}]: {call_group_id_y[y, x]} >= {grid_shape_y}"
                assert (
                    0 <= call_group_id_x[y, x] < grid_shape_x
                ), f"call_group_id[1] out of bounds at [{y},{x}]: {call_group_id_x[y, x]} >= {grid_shape_x}"

                # Validate call_group_thread_id bounds
                assert (
                    0 <= call_group_thread_id_y[y, x] < group_shape[0]
                ), f"call_group_thread_id[0] out of bounds at [{y},{x}]: {call_group_thread_id_y[y, x]} >= {group_shape[0]}"
                assert (
                    0 <= call_group_thread_id_x[y, x] < group_shape[1]
                ), f"call_group_thread_id[1] out of bounds at [{y},{x}]: {call_group_thread_id_x[y, x]} >= {group_shape[1]}"

        # Validate that each call group has exactly one thread with thread_id [0,0]
        for group_y in range(grid_shape_y):
            for group_x in range(grid_shape_x):
                # Count threads with [0,0] thread_id in this call group
                zero_zero_count = 0
                for y in range(call_shape[0]):
                    for x in range(call_shape[1]):
                        if (
                            call_group_id_y[y, x] == group_y
                            and call_group_id_x[y, x] == group_x
                            and call_group_thread_id_y[y, x] == 0
                            and call_group_thread_id_x[y, x] == 0
                        ):
                            zero_zero_count += 1

                # Each call group should have exactly one [0,0] thread (if the group has any threads)
                group_has_threads = False
                for y in range(call_shape[0]):
                    for x in range(call_shape[1]):
                        if call_group_id_y[y, x] == group_y and call_group_id_x[y, x] == group_x:
                            group_has_threads = True
                            break
                    if group_has_threads:
                        break

                if group_has_threads:
                    assert (
                        zero_zero_count == 1
                    ), f"Call group [{group_y},{group_x}] should have exactly 1 thread with thread_id [0,0], found {zero_zero_count}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_math_5d_validation(device_type: DeviceType):
    """
    Test that SlangPy's call group calculations work for 5D (simplified test).

    This test validates that 5D call group functions can be called without errors
    and return reasonable values, rather than doing deep mathematical validation.
    """

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

uint test_5d_basic(uint[5] grid_cell) {
    // Get call group values
    CallShapeInfo call_id = CallShapeInfo::get_call_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();

    // Pack some of the results to validate they're not all zeros
    // Pack dimensions 2, 3, 4 (which have non-trivial group shapes)
    return (call_group_id.shape[2] << 16) | (call_group_thread_id.shape[3] << 8) | call_group_thread_id.shape[4];
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test with simple 5D configuration
    call_shape = (3, 2, 2, 2, 2)
    call_group_shape = (1, 1, 2, 2, 2)

    # Create output buffer
    result_buffer = NDBuffer(device=device, shape=call_shape, dtype=int)

    # Call with call group shape
    module.test_5d_basic.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result=result_buffer
    )

    # Validate that we get meaningful values, not all zeros
    results = result_buffer.to_numpy()
    assert np.any(results > 0), "Expected some non-zero call group values in 5D test"

    # Extract components for basic validation
    call_group_id_2 = (results >> 16) & 0xFFFF
    call_group_thread_id_3 = (results >> 8) & 0xFF
    call_group_thread_id_4 = results & 0xFF

    # Basic bounds checking
    expected_grid_2 = call_shape[2] // call_group_shape[2]
    assert np.all(call_group_id_2 < expected_grid_2), "call_group_id[2] out of bounds"
    assert np.all(
        call_group_thread_id_3 < call_group_shape[3]
    ), "call_group_thread_id[3] out of bounds"
    assert np.all(
        call_group_thread_id_4 < call_group_shape[4]
    ), "call_group_thread_id[4] out of bounds"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_math_edge_cases(device_type: DeviceType):
    """Test edge cases and boundary conditions."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

uint test_edge_case_basic(uint2 grid_cell) {
    // Get call group functionality and pack results
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();

    // Pack results to validate they're meaningful
    return (call_group_id.shape[0] << 24) | (call_group_id.shape[1] << 16) |
           (call_group_thread_id.shape[0] << 8) | call_group_thread_id.shape[1];
}
"""

    module = helpers.create_module(device, kernel_source)

    # Edge case 1: 1x1 groups (should work without error)
    call_shape = (3, 3)

    # With 1x1 groups
    result_with_groups = module.test_edge_case_basic.call_group_shape(Shape((1, 1)))(
        spy.grid(call_shape), _result="numpy"
    )

    # Validate 1x1 groups: each thread should be its own group
    group_id_y = (result_with_groups >> 24) & 0xFF
    group_id_x = (result_with_groups >> 16) & 0xFF
    thread_id_y = (result_with_groups >> 8) & 0xFF
    thread_id_x = result_with_groups & 0xFF

    # With 1x1 groups, call_group_id should match position and thread_id should be 0
    for y in range(call_shape[0]):
        for x in range(call_shape[1]):
            assert (
                group_id_y[y, x] == y
            ), f"Expected group_id_y={y} at [{y},{x}], got {group_id_y[y, x]}"
            assert (
                group_id_x[y, x] == x
            ), f"Expected group_id_x={x} at [{y},{x}], got {group_id_x[y, x]}"
            assert (
                thread_id_y[y, x] == 0
            ), f"Expected thread_id_y=0 at [{y},{x}], got {thread_id_y[y, x]}"
            assert (
                thread_id_x[y, x] == 0
            ), f"Expected thread_id_x=0 at [{y},{x}], got {thread_id_x[y, x]}"

    # Edge case 2: Group larger than call shape
    large_group_result = module.test_edge_case_basic.call_group_shape(Shape((4, 4)))(
        spy.grid((2, 2)), _result="numpy"
    )

    # With large groups, all threads should be in group [0,0]
    large_group_id_y = (large_group_result >> 24) & 0xFF
    large_group_id_x = (large_group_result >> 16) & 0xFF
    large_thread_id_y = (large_group_result >> 8) & 0xFF
    large_thread_id_x = large_group_result & 0xFF

    assert np.all(large_group_id_y == 0), "All threads should be in group_id_y=0 for large groups"
    assert np.all(large_group_id_x == 0), "All threads should be in group_id_x=0 for large groups"
    assert np.all(large_thread_id_y < 2), "thread_id_y should be < call_shape[0]"
    assert np.all(large_thread_id_x < 2), "thread_id_x should be < call_shape[1]"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "call_shape,call_group_shape,description", VALID_CALL_GROUP_SHAPE_TEST_CASES
)
def test_call_group_shape_valid(
    device_type: DeviceType,
    call_shape: tuple,
    call_group_shape: tuple,
    description: str,
):
    """Test different valid call group shapes and validate things pass."""

    device = helpers.get_device(device_type)

    # Infer dimension from call_shape
    dimensionality = len(call_shape)

    # Create a simpler test that just validates the math works
    # For 5D+, use array types like uint[5], uint[6], etc.
    if dimensionality <= 4:
        param_type = {1: "uint", 2: "uint2", 3: "uint3", 4: "uint4"}[dimensionality]
    else:
        param_type = f"uint[{dimensionality}]"
    kernel_source = f"""
import "slangpy";

int test_call_group_math({param_type} grid_pos) {{
    // Just test that the functions can be called without error
    CallShapeInfo call_id = CallShapeInfo::get_call_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();

    // Return a simple validation that functions executed
    return (call_id.dimensionality == {dimensionality}) &&
           (call_group_id.dimensionality == {dimensionality}) &&
           (call_group_thread_id.dimensionality == {dimensionality});
}}
"""

    module = helpers.create_module(device, kernel_source)

    # Create output buffer
    result_buffer = NDBuffer(device=device, shape=call_shape, dtype=float)

    # Call with call group shape
    module.test_call_group_math.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result=result_buffer
    )

    # Validate that all calls completed successfully (all should be 1.0)
    results = result_buffer.to_numpy()
    assert np.all(
        results == 1
    ), f"Failed for {description}: {call_shape} with groups {call_group_shape}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "test_name,call_shape,call_group_shape,expected_error_fragments,description",
    INVALID_CALL_GROUP_SHAPE_TEST_CASES,
)
def test_call_group_error_invalid(
    device_type: DeviceType,
    test_name: str,
    call_shape: tuple,
    call_group_shape: tuple,
    expected_error_fragments: list,
    description: str,
):
    """Test different invalid call group shapes and validate expected errors."""

    device = helpers.get_device(device_type)

    # Create appropriate kernel based on call_shape dimensions
    if len(call_shape) == 2:
        kernel_source = """
import "slangpy";

float test_2d(uint2 grid_pos) {
    return 1.0f;
}
"""
        module = helpers.create_module(device, kernel_source)
        test_func = module.test_2d
    else:  # len(call_shape) == 1
        kernel_source = """
import "slangpy";

float test_1d(uint grid_pos) {
    return 1.0f;
}
"""
        module = helpers.create_module(device, kernel_source)
        test_func = module.test_1d

    # Test should fail with expected error
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        test_func.call_group_shape(Shape(call_group_shape))(spy.grid(call_shape), _result="numpy")

    error_message = str(exc_info.value)

    # Check that all expected error fragments are present
    for fragment in expected_error_fragments:
        assert (
            fragment in error_message
        ), f"Test '{test_name}': Expected '{fragment}' in error message: {error_message}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_math_verified_thread_id_patterns(device_type: DeviceType):
    """
    Test call group math by validating the exact thread_id patterns for known configurations.

    This test validates specific thread_id values that result from call group dispatch arrangements.
    The output format is [call_id_x, call_id_y, thread_id.x] for each position [y, x].
    Note: CallShapeInfo::get_call_id().shape returns [y, x] but we rearrange to [x, y] for clarity.

    Two key scenarios tested:

    1. DEFAULT CASE (no call groups):
       - Uses default linear dispatch (effectively 1×1 call groups)
       - thread_id.x follows pattern: thread_id = y * width + x
       - Examples: [0,0]→thread_id=0, [0,1]→thread_id=1, [1,0]→thread_id=64, [1,1]→thread_id=65

    2. CALL GROUP CASE (4, 8) groups:
       - Threads arranged in spatial 4×8 groups
       - thread_id assignment depends on group layout and changes from linear pattern
       - Examples: [0,0]→thread_id=0, [0,1]→thread_id=1, [1,0]→thread_id=8, [1,1]→thread_id=9

    The key insight: call groups change how thread_id.x is assigned while call_id remains position-based.
    """

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

// Returns call_id and thread_id for pattern validation
// Note: CallShapeInfo::get_call_id().shape returns [y, x] order
uint3 test_thread_id_patterns(uint2 grid_cell, uint3 thread_id) {
    CallShapeInfo call_id = CallShapeInfo::get_call_id();
    return uint3(call_id.shape[1], call_id.shape[0], thread_id.x);  // Return as [x, y, thread_id.x]
}
"""

    module = helpers.create_module(device, kernel_source)
    call_shape = (32, 64)  # 32 rows × 64 columns

    # Test 1: DEFAULT CASE (no call groups)
    # Expected: Linear thread assignment where thread_id.x = y * width + x
    result_default = module.test_thread_id_patterns(
        spy.grid(call_shape), spy.thread_id(), _result="numpy"
    )

    # Row 0: thread_id should equal x coordinate
    # Position [y=0,x=0]: call_id_x=0, call_id_y=0, thread_id=0*64+0=0
    # Position [y=0,x=1]: call_id_x=1, call_id_y=0, thread_id=0*64+1=1
    # Position [y=0,x=63]: call_id_x=63, call_id_y=0, thread_id=0*64+63=63
    for x in range(min(10, call_shape[1])):  # Test first 10 positions
        expected_call_id_x = x
        expected_call_id_y = 0
        expected_thread_id = 0 * call_shape[1] + x  # y * width + x
        expected = [expected_call_id_x, expected_call_id_y, expected_thread_id]
        actual = result_default[0, x]
        assert np.array_equal(
            actual, expected
        ), f"Default [0,{x}]: expected {expected}, got {actual}"

    # Row 1: thread_id should be 64 + x
    # Position [y=1,x=0]: call_id_x=0, call_id_y=1, thread_id=1*64+0=64
    # Position [y=1,x=1]: call_id_x=1, call_id_y=1, thread_id=1*64+1=65
    for x in range(min(10, call_shape[1])):  # Test first 10 positions
        expected_call_id_x = x
        expected_call_id_y = 1
        expected_thread_id = 1 * call_shape[1] + x  # y * width + x = 64 + x
        expected = [expected_call_id_x, expected_call_id_y, expected_thread_id]
        actual = result_default[1, x]
        assert np.array_equal(
            actual, expected
        ), f"Default [1,{x}]: expected {expected}, got {actual}"

    # Last position [y=31,x=63]: call_id_x=63, call_id_y=31, thread_id should be 31*64+63=2047 (total threads - 1)
    expected_31_63 = [63, 31, 31 * call_shape[1] + 63]  # [63, 31, 2047]
    actual_31_63 = result_default[31, 63]
    assert np.array_equal(
        actual_31_63, expected_31_63
    ), f"Default [31,63]: expected {expected_31_63}, got {actual_31_63}"

    # Test 2: CALL GROUP CASE (4, 8) groups
    # Expected: Non-linear thread assignment due to spatial group arrangement
    call_group_shape = (4, 8)
    result_grouped = module.test_thread_id_patterns.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), spy.thread_id(), _result="numpy"
    )

    # With (4, 8) call groups, thread assignment follows group-based pattern:
    # - Grid divided into 8×8 call groups (32÷4 = 8 rows, 64÷8 = 8 columns of groups)
    # - Each group contains 4×8 = 32 threads
    # - thread_id assignment prioritizes spatial locality within groups

    # Row 0 positions: early positions should have thread_id = x (same as default)
    # This happens because first group spans [0,0] to [3,7] and early threads get low IDs
    for x in range(min(8, call_shape[1])):  # Test first group's width
        expected_call_id_x = x
        expected_call_id_y = 0
        expected_thread_id = x  # Should match default for early positions
        expected = [expected_call_id_x, expected_call_id_y, expected_thread_id]
        actual = result_grouped[0, x]
        assert np.array_equal(
            actual, expected
        ), f"Grouped [0,{x}]: expected {expected}, got {actual}"

    # Row 1 key positions: thread_id pattern changes due to spatial grouping
    # Position [1,0]: In same group as [0,0], should have thread_id offset by group row
    # Based on (4,8) groups: thread within group gets ID based on position in group
    expected_1_0 = [0, 1, 8]  # 2nd row of first group: base_id + row_offset*group_width
    actual_1_0 = result_grouped[1, 0]
    assert np.array_equal(
        actual_1_0, expected_1_0
    ), f"Grouped [1,0]: expected {expected_1_0}, got {actual_1_0}"

    expected_1_1 = [1, 1, 9]  # Next column in same row of group
    actual_1_1 = result_grouped[1, 1]
    assert np.array_equal(
        actual_1_1, expected_1_1
    ), f"Grouped [1,1]: expected {expected_1_1}, got {actual_1_1}"

    # Test cross-group boundaries
    # Position [0,8]: Start of second call group in same row
    # This position starts a new group, so thread_id pattern differs from linear
    expected_0_8_thread_id = 32  # Start of 2nd group (group_id=1 * group_size=32)
    expected_0_8 = [8, 0, expected_0_8_thread_id]
    actual_0_8 = result_grouped[0, 8]
    assert np.array_equal(
        actual_0_8, expected_0_8
    ), f"Grouped [0,8]: expected {expected_0_8}, got {actual_0_8}"

    # Position [4,0]: Start of second row of call groups
    # This moves to next "group row", affecting thread_id assignment
    expected_4_0_thread_id = 256  # Start of 2nd group row (8 groups * 32 threads each)
    expected_4_0 = [0, 4, expected_4_0_thread_id]
    actual_4_0 = result_grouped[4, 0]
    assert np.array_equal(
        actual_4_0, expected_4_0
    ), f"Grouped [4,0]: expected {expected_4_0}, got {actual_4_0}"

    # Verify call_id invariant: call_id should always match [x, y] position
    for y in range(0, call_shape[0], 8):  # Sample every 8th row
        for x in range(0, call_shape[1], 8):  # Sample every 8th column
            call_id_x = result_grouped[y, x, 0]
            call_id_y = result_grouped[y, x, 1]
            assert call_id_x == x, f"call_id[0] should equal x at [{y},{x}]: got {call_id_x}"
            assert call_id_y == y, f"call_id[1] should equal y at [{y},{x}]: got {call_id_y}"

    # Validate last position still gets highest thread_id (total threads - 1)
    expected_31_63 = [63, 31, 2047]  # Last thread should still be 2047
    actual_31_63 = result_grouped[31, 63]
    assert np.array_equal(
        actual_31_63, expected_31_63
    ), f"Grouped [31,63]: expected {expected_31_63}, got {actual_31_63}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
