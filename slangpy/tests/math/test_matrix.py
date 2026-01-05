# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for matrix types and operations.
"""

import numpy as np
import pytest

import slangpy as spy


def test_shape_and_element_types():
    for rows in range(2, 5):
        for cols in range(2, 5):
            floattype = getattr(spy, f"float{rows}x{cols}", None)
            if floattype is not None:
                floatval = floattype()
                assert floatval.element_type == float
                assert floatval.shape == (rows, cols)


class TestMatrixMulFunction:
    """Test spy.math.mul() function for all valid matrix-matrix combinations."""

    # Matrices with 2 columns can multiply by matrices with 2 rows
    @pytest.mark.parametrize(
        "lhs_type,rhs_type,expected_shape",
        [
            ("float2x2", "float2x2", (2, 2)),
            ("float2x2", "float2x3", (2, 3)),
            ("float2x2", "float2x4", (2, 4)),
            ("float3x2", "float2x2", (3, 2)),
            ("float3x2", "float2x3", (3, 3)),
            ("float3x2", "float2x4", (3, 4)),
            ("float4x2", "float2x2", (4, 2)),
            ("float4x2", "float2x3", (4, 3)),
            ("float4x2", "float2x4", (4, 4)),
        ],
    )
    def test_mul_2col_matrices(self, lhs_type, rhs_type, expected_shape):
        """Test multiplication of matrices with 2 columns."""
        lhs_cls = getattr(spy.math, lhs_type)
        rhs_cls = getattr(spy.math, rhs_type)

        lhs = lhs_cls.identity()
        rhs = rhs_cls.identity()

        result = spy.math.mul(lhs, rhs)
        assert result.shape == expected_shape

    # Matrices with 3 columns can multiply by matrices with 3 rows
    @pytest.mark.parametrize(
        "lhs_type,rhs_type,expected_shape",
        [
            ("float2x3", "float3x2", (2, 2)),
            ("float2x3", "float3x3", (2, 3)),
            ("float2x3", "float3x4", (2, 4)),
            ("float3x3", "float3x2", (3, 2)),
            ("float3x3", "float3x3", (3, 3)),
            ("float3x3", "float3x4", (3, 4)),
            ("float4x3", "float3x2", (4, 2)),
            ("float4x3", "float3x3", (4, 3)),
            ("float4x3", "float3x4", (4, 4)),
        ],
    )
    def test_mul_3col_matrices(self, lhs_type, rhs_type, expected_shape):
        """Test multiplication of matrices with 3 columns."""
        lhs_cls = getattr(spy.math, lhs_type)
        rhs_cls = getattr(spy.math, rhs_type)

        lhs = lhs_cls.identity()
        rhs = rhs_cls.identity()

        result = spy.math.mul(lhs, rhs)
        assert result.shape == expected_shape

    # Matrices with 4 columns can multiply by matrices with 4 rows
    @pytest.mark.parametrize(
        "lhs_type,rhs_type,expected_shape",
        [
            ("float2x4", "float4x2", (2, 2)),
            ("float2x4", "float4x3", (2, 3)),
            ("float2x4", "float4x4", (2, 4)),
            ("float3x4", "float4x2", (3, 2)),
            ("float3x4", "float4x3", (3, 3)),
            ("float3x4", "float4x4", (3, 4)),
            ("float4x4", "float4x2", (4, 2)),
            ("float4x4", "float4x3", (4, 3)),
            ("float4x4", "float4x4", (4, 4)),
        ],
    )
    def test_mul_4col_matrices(self, lhs_type, rhs_type, expected_shape):
        """Test multiplication of matrices with 4 columns."""
        lhs_cls = getattr(spy.math, lhs_type)
        rhs_cls = getattr(spy.math, rhs_type)

        lhs = lhs_cls.identity()
        rhs = rhs_cls.identity()

        result = spy.math.mul(lhs, rhs)
        assert result.shape == expected_shape


class TestMatrixMatmulOperator:
    """Test @ operator for all valid matrix-matrix combinations."""

    # Matrices with 2 columns can multiply by matrices with 2 rows
    @pytest.mark.parametrize(
        "lhs_type,rhs_type,expected_shape",
        [
            ("float2x2", "float2x2", (2, 2)),
            ("float2x2", "float2x3", (2, 3)),
            ("float2x2", "float2x4", (2, 4)),
            ("float3x2", "float2x2", (3, 2)),
            ("float3x2", "float2x3", (3, 3)),
            ("float3x2", "float2x4", (3, 4)),
            ("float4x2", "float2x2", (4, 2)),
            ("float4x2", "float2x3", (4, 3)),
            ("float4x2", "float2x4", (4, 4)),
        ],
    )
    def test_matmul_2col_matrices(self, lhs_type, rhs_type, expected_shape):
        """Test @ operator for matrices with 2 columns."""
        lhs_cls = getattr(spy.math, lhs_type)
        rhs_cls = getattr(spy.math, rhs_type)

        lhs = lhs_cls.identity()
        rhs = rhs_cls.identity()

        result = lhs @ rhs
        assert result.shape == expected_shape

    # Matrices with 3 columns can multiply by matrices with 3 rows
    @pytest.mark.parametrize(
        "lhs_type,rhs_type,expected_shape",
        [
            ("float2x3", "float3x2", (2, 2)),
            ("float2x3", "float3x3", (2, 3)),
            ("float2x3", "float3x4", (2, 4)),
            ("float3x3", "float3x2", (3, 2)),
            ("float3x3", "float3x3", (3, 3)),
            ("float3x3", "float3x4", (3, 4)),
            ("float4x3", "float3x2", (4, 2)),
            ("float4x3", "float3x3", (4, 3)),
            ("float4x3", "float3x4", (4, 4)),
        ],
    )
    def test_matmul_3col_matrices(self, lhs_type, rhs_type, expected_shape):
        """Test @ operator for matrices with 3 columns."""
        lhs_cls = getattr(spy.math, lhs_type)
        rhs_cls = getattr(spy.math, rhs_type)

        lhs = lhs_cls.identity()
        rhs = rhs_cls.identity()

        result = lhs @ rhs
        assert result.shape == expected_shape

    # Matrices with 4 columns can multiply by matrices with 4 rows
    @pytest.mark.parametrize(
        "lhs_type,rhs_type,expected_shape",
        [
            ("float2x4", "float4x2", (2, 2)),
            ("float2x4", "float4x3", (2, 3)),
            ("float2x4", "float4x4", (2, 4)),
            ("float3x4", "float4x2", (3, 2)),
            ("float3x4", "float4x3", (3, 3)),
            ("float3x4", "float4x4", (3, 4)),
            ("float4x4", "float4x2", (4, 2)),
            ("float4x4", "float4x3", (4, 3)),
            ("float4x4", "float4x4", (4, 4)),
        ],
    )
    def test_matmul_4col_matrices(self, lhs_type, rhs_type, expected_shape):
        """Test @ operator produces correct result shape."""
        lhs_cls = getattr(spy.math, lhs_type)
        rhs_cls = getattr(spy.math, rhs_type)

        lhs = lhs_cls.identity()
        rhs = rhs_cls.identity()

        result = lhs @ rhs
        assert result.shape == expected_shape


class TestMatrixMulValues:
    """Test that matrix multiplication produces correct values."""

    def test_mul_identity(self):
        """Multiplying by identity should return the original matrix."""
        m = spy.float3x4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        identity = spy.float4x4.identity()

        result = m @ identity
        result_np = result.to_numpy()
        m_np = m.to_numpy()

        assert np.allclose(result_np, m_np)

    def test_mul_known_values(self):
        """Test multiplication with known values."""
        # 2x3 @ 3x2 = 2x2
        a = spy.float2x3([1, 2, 3, 4, 5, 6])  # [[1,2,3], [4,5,6]]
        b = spy.float3x2([1, 2, 3, 4, 5, 6])  # [[1,2], [3,4], [5,6]]

        result = a @ b
        result_np = result.to_numpy()

        # Manual calculation:
        # [1,2,3] @ [[1,2], [3,4], [5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        # [4,5,6] @ [[1,2], [3,4], [5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        expected = np.array([[22, 28], [49, 64]], dtype=np.float32)

        assert np.allclose(result_np, expected)

    def test_chained_multiplication(self):
        """Test chained matrix multiplication (A @ B @ C)."""
        a = spy.float2x3.identity()
        b = spy.float3x4.identity()
        c = spy.float4x2.identity()

        # (2x3) @ (3x4) @ (4x2) = (2x4) @ (4x2) = (2x2)
        result = a @ b @ c
        assert result.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
