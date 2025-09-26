# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
NumPy validation tests for posrot and posrotscale types.
These tests compare the slangpy implementations against numpy reference implementations.
"""

import numpy as np
from slangpy import float3, quatf, posrotf, posrotscalef


def test_matrix_conversion_identity():
    """Test identity transform matrix conversion."""
    pr = posrotf.identity()
    m4x4 = pr.to_matrix4x4()

    # Convert to numpy for comparison
    np_matrix = np.array(
        [
            [m4x4[0][0], m4x4[0][1], m4x4[0][2], m4x4[0][3]],
            [m4x4[1][0], m4x4[1][1], m4x4[1][2], m4x4[1][3]],
            [m4x4[2][0], m4x4[2][1], m4x4[2][2], m4x4[2][3]],
            [m4x4[3][0], m4x4[3][1], m4x4[3][2], m4x4[3][3]],
        ]
    )

    expected = np.eye(4)
    np.testing.assert_allclose(np_matrix, expected, rtol=1e-6)


def test_translation_matrix():
    """Test translation matrix against numpy."""
    pos = float3(1.5, -2.3, 4.7)
    pr = posrotf.translation(pos)
    m4x4 = pr.to_matrix4x4()

    # Convert to numpy
    np_matrix = np.array(
        [
            [m4x4[0][0], m4x4[0][1], m4x4[0][2], m4x4[0][3]],
            [m4x4[1][0], m4x4[1][1], m4x4[1][2], m4x4[1][3]],
            [m4x4[2][0], m4x4[2][1], m4x4[2][2], m4x4[2][3]],
            [m4x4[3][0], m4x4[3][1], m4x4[3][2], m4x4[3][3]],
        ]
    )

    # Expected numpy translation matrix
    expected = np.eye(4)
    expected[0, 3] = pos.x
    expected[1, 3] = pos.y
    expected[2, 3] = pos.z

    np.testing.assert_allclose(np_matrix, expected, rtol=1e-6)


def test_point_transformation():
    """Test point transformation against numpy matrix multiplication."""
    # Create a transform
    pos = float3(2.0, 3.0, 1.0)
    pr = posrotf.translation(pos)

    # Transform a point using slangpy
    point = float3(1.0, 2.0, 3.0)
    transformed_slang = pr * point

    # Transform using numpy matrix multiplication
    m4x4 = pr.to_matrix4x4()
    np_matrix = np.array(
        [
            [m4x4[0][0], m4x4[0][1], m4x4[0][2], m4x4[0][3]],
            [m4x4[1][0], m4x4[1][1], m4x4[1][2], m4x4[1][3]],
            [m4x4[2][0], m4x4[2][1], m4x4[2][2], m4x4[2][3]],
            [m4x4[3][0], m4x4[3][1], m4x4[3][2], m4x4[3][3]],
        ]
    )

    np_point = np.array([point.x, point.y, point.z, 1.0])
    np_result = np_matrix @ np_point

    # Compare results (only xyz components)
    np.testing.assert_allclose(
        [transformed_slang.x, transformed_slang.y, transformed_slang.z], np_result[:3], rtol=1e-6
    )


def test_transform_composition():
    """Test transform composition against numpy matrix multiplication."""
    # Create two transforms
    pr1 = posrotf.translation(float3(1.0, 2.0, 0.0))
    pr2 = posrotf.translation(float3(3.0, 1.0, 1.0))

    # Compose using slangpy
    result_slang = pr1 * pr2

    # Compose using numpy matrices
    m1 = pr1.to_matrix4x4()
    m2 = pr2.to_matrix4x4()

    np_m1 = np.array(
        [
            [m1[0][0], m1[0][1], m1[0][2], m1[0][3]],
            [m1[1][0], m1[1][1], m1[1][2], m1[1][3]],
            [m1[2][0], m1[2][1], m1[2][2], m1[2][3]],
            [m1[3][0], m1[3][1], m1[3][2], m1[3][3]],
        ]
    )

    np_m2 = np.array(
        [
            [m2[0][0], m2[0][1], m2[0][2], m2[0][3]],
            [m2[1][0], m2[1][1], m2[1][2], m2[1][3]],
            [m2[2][0], m2[2][1], m2[2][2], m2[2][3]],
            [m2[3][0], m2[3][1], m2[3][2], m2[3][3]],
        ]
    )

    np_result = np_m1 @ np_m2
    result_slang_matrix = result_slang.to_matrix4x4()

    np_slang_result = np.array(
        [
            [
                result_slang_matrix[0][0],
                result_slang_matrix[0][1],
                result_slang_matrix[0][2],
                result_slang_matrix[0][3],
            ],
            [
                result_slang_matrix[1][0],
                result_slang_matrix[1][1],
                result_slang_matrix[1][2],
                result_slang_matrix[1][3],
            ],
            [
                result_slang_matrix[2][0],
                result_slang_matrix[2][1],
                result_slang_matrix[2][2],
                result_slang_matrix[2][3],
            ],
            [
                result_slang_matrix[3][0],
                result_slang_matrix[3][1],
                result_slang_matrix[3][2],
                result_slang_matrix[3][3],
            ],
        ]
    )

    np.testing.assert_allclose(np_slang_result, np_result, rtol=1e-6)


def test_scaling_matrix():
    """Test scaling matrix against numpy."""
    scale = float3(2.0, 0.5, 3.0)
    prs = posrotscalef.from_scaling(scale)
    m4x4 = prs.to_matrix4x4()

    # Convert to numpy
    np_matrix = np.array(
        [
            [m4x4[0][0], m4x4[0][1], m4x4[0][2], m4x4[0][3]],
            [m4x4[1][0], m4x4[1][1], m4x4[1][2], m4x4[1][3]],
            [m4x4[2][0], m4x4[2][1], m4x4[2][2], m4x4[2][3]],
            [m4x4[3][0], m4x4[3][1], m4x4[3][2], m4x4[3][3]],
        ]
    )

    # Expected numpy scaling matrix
    expected = np.diag([scale.x, scale.y, scale.z, 1.0])

    np.testing.assert_allclose(np_matrix, expected, rtol=1e-6)


def test_uniform_scaling_matrix():
    """Test uniform scaling matrix."""
    factor = 2.5
    prs = posrotscalef.uniform_scaling(factor)
    m4x4 = prs.to_matrix4x4()

    # Convert to numpy
    np_matrix = np.array(
        [
            [m4x4[0][0], m4x4[0][1], m4x4[0][2], m4x4[0][3]],
            [m4x4[1][0], m4x4[1][1], m4x4[1][2], m4x4[1][3]],
            [m4x4[2][0], m4x4[2][1], m4x4[2][2], m4x4[2][3]],
            [m4x4[3][0], m4x4[3][1], m4x4[3][2], m4x4[3][3]],
        ]
    )

    # Expected numpy uniform scaling matrix
    expected = np.diag([factor, factor, factor, 1.0])

    np.testing.assert_allclose(np_matrix, expected, rtol=1e-6)


def test_scaled_point_transformation():
    """Test scaled point transformation against numpy."""
    # Create a scale transform
    scale = float3(2.0, 0.5, 3.0)
    prs = posrotscalef.from_scaling(scale)

    # Transform a point
    point = float3(1.0, 4.0, 2.0)
    transformed_slang = prs * point

    # Expected result: component-wise multiplication
    expected_np = np.array([point.x * scale.x, point.y * scale.y, point.z * scale.z])

    np.testing.assert_allclose(
        [transformed_slang.x, transformed_slang.y, transformed_slang.z], expected_np, rtol=1e-6
    )


def test_trs_composition():
    """Test translation-rotation-scale composition."""
    # Create individual transforms
    pos = float3(1.0, 2.0, 3.0)
    scale = float3(2.0, 1.5, 0.5)

    # Create combined transform
    prs = posrotscalef(pos, quatf.identity(), scale)

    # Test point transformation
    point = float3(2.0, 3.0, 4.0)
    transformed_slang = prs * point

    # Manual computation: scale first, then translate
    scaled = np.array([point.x * scale.x, point.y * scale.y, point.z * scale.z])
    translated = scaled + np.array([pos.x, pos.y, pos.z])

    np.testing.assert_allclose(
        [transformed_slang.x, transformed_slang.y, transformed_slang.z], translated, rtol=1e-6
    )


def test_inverse_transform():
    """Test inverse transform against numpy matrix inverse."""
    # Create a complex transform
    prs = posrotscalef(float3(2.0, 3.0, 1.0), quatf.identity(), float3(2.0, 0.5, 4.0))

    # Get inverse using slangpy
    inv_slang = prs.inverse()

    # Get matrix and compute numpy inverse
    m4x4 = prs.to_matrix4x4()
    np_matrix = np.array(
        [
            [m4x4[0][0], m4x4[0][1], m4x4[0][2], m4x4[0][3]],
            [m4x4[1][0], m4x4[1][1], m4x4[1][2], m4x4[1][3]],
            [m4x4[2][0], m4x4[2][1], m4x4[2][2], m4x4[2][3]],
            [m4x4[3][0], m4x4[3][1], m4x4[3][2], m4x4[3][3]],
        ]
    )

    np_inverse = np.linalg.inv(np_matrix)

    # Convert slangpy inverse to numpy
    inv_m4x4 = inv_slang.to_matrix4x4()
    np_slang_inverse = np.array(
        [
            [inv_m4x4[0][0], inv_m4x4[0][1], inv_m4x4[0][2], inv_m4x4[0][3]],
            [inv_m4x4[1][0], inv_m4x4[1][1], inv_m4x4[1][2], inv_m4x4[1][3]],
            [inv_m4x4[2][0], inv_m4x4[2][1], inv_m4x4[2][2], inv_m4x4[2][3]],
            [inv_m4x4[3][0], inv_m4x4[3][1], inv_m4x4[3][2], inv_m4x4[3][3]],
        ]
    )

    np.testing.assert_allclose(np_slang_inverse, np_inverse, rtol=1e-5)


def test_lerp_interpolation():
    """Test linear interpolation against numpy lerp."""
    # Create two transforms
    prs1 = posrotscalef(float3(0.0, 0.0, 0.0), quatf.identity(), float3(1.0, 1.0, 1.0))
    prs2 = posrotscalef(float3(4.0, 6.0, 2.0), quatf.identity(), float3(3.0, 2.0, 4.0))

    # Test various t values
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Slangpy lerp
        result_slang = prs1.lerp(prs2, t)

        # Numpy reference lerp
        pos1 = np.array([prs1.pos.x, prs1.pos.y, prs1.pos.z])
        pos2 = np.array([prs2.pos.x, prs2.pos.y, prs2.pos.z])
        scale1 = np.array([prs1.scale.x, prs1.scale.y, prs1.scale.z])
        scale2 = np.array([prs2.scale.x, prs2.scale.y, prs2.scale.z])

        expected_pos = pos1 * (1 - t) + pos2 * t
        expected_scale = scale1 * (1 - t) + scale2 * t

        # Compare positions and scales
        np.testing.assert_allclose(
            [result_slang.pos.x, result_slang.pos.y, result_slang.pos.z], expected_pos, rtol=1e-6
        )
        np.testing.assert_allclose(
            [result_slang.scale.x, result_slang.scale.y, result_slang.scale.z],
            expected_scale,
            rtol=1e-6,
        )
