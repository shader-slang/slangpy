# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import pytest
from slangpy import (
    bool1,
    bool2,
    bool3,
    float1,
    float16_t1,
    float16_t2,
    float16_t3,
    float16_t4,
    float2,
    float3,
    float4,
    int1,
    int2,
    int3,
    int4,
    uint1,
    uint2,
    uint3,
    uint4,
    bool4,
)


def eq(a: float4 | uint4 | bool4, b: float4 | uint4 | bool4):
    assert type(a) == type(b)
    return a.x == b.x and a.y == b.y and a.z == b.z and a.w == b.w


def test_float4_constructor():
    assert eq(float4(), float4(0, 0, 0, 0))
    assert eq(float4(1), float4(1, 1, 1, 1))
    assert eq(float4(1, 2, 3, 4), float4(1, 2, 3, 4))


def test_float4_fields():
    assert float4(1, 2, 3, 4).x == 1
    assert float4(1, 2, 3, 4).y == 2
    assert float4(1, 2, 3, 4).z == 3
    assert float4(1, 2, 3, 4).w == 4
    a = float4()
    a.x = 1
    a.y = 2
    a.z = 3
    a.w = 4
    assert eq(a, float4(1, 2, 3, 4))


def test_float4_str():
    assert str(float4(1, 2, 3, 4)) == "{1, 2, 3, 4}"


def test_float4_unary_ops():
    assert eq(+float4(1, 2, 3, 4), float4(1, 2, 3, 4))
    assert eq(+float4(2, 3, 4, 5), float4(2, 3, 4, 5))
    assert eq(+float4(3, 4, 5, 6), float4(3, 4, 5, 6))
    assert eq(+float4(4, 5, 6, 7), float4(4, 5, 6, 7))

    assert eq(-float4(1, 2, 3, 4), float4(-1, -2, -3, -4))
    assert eq(-float4(2, 3, 4, 5), float4(-2, -3, -4, -5))
    assert eq(-float4(3, 4, 5, 6), float4(-3, -4, -5, -6))
    assert eq(-float4(4, 5, 6, 7), float4(-4, -5, -6, -7))


def test_float4_binary_ops():
    assert eq(float4(1, 2, 3, 4) + float4(1, 2, 3, 4), float4(2, 4, 6, 8))
    assert eq(float4(1, 2, 3, 4) + float4(2, 3, 4, 5), float4(3, 5, 7, 9))
    assert eq(float4(1, 2, 3, 4) + float4(3, 4, 5, 6), float4(4, 6, 8, 10))
    assert eq(float4(1, 2, 3, 4) + float4(4, 5, 6, 7), float4(5, 7, 9, 11))

    assert eq(float4(1, 2, 3, 4) - float4(1, 2, 3, 4), float4(0, 0, 0, 0))
    assert eq(float4(1, 2, 3, 4) - float4(2, 3, 4, 5), float4(-1, -1, -1, -1))
    assert eq(float4(1, 2, 3, 4) - float4(3, 4, 5, 6), float4(-2, -2, -2, -2))
    assert eq(float4(1, 2, 3, 4) - float4(4, 5, 6, 7), float4(-3, -3, -3, -3))

    assert eq(float4(1, 2, 3, 4) * float4(1, 2, 3, 4), float4(1, 4, 9, 16))
    assert eq(float4(1, 2, 3, 4) * float4(2, 3, 4, 5), float4(2, 6, 12, 20))
    assert eq(float4(1, 2, 3, 4) * float4(3, 4, 5, 6), float4(3, 8, 15, 24))
    assert eq(float4(1, 2, 3, 4) * float4(4, 5, 6, 7), float4(4, 10, 18, 28))

    assert eq(float4(1, 2, 3, 4) / float4(1, 2, 3, 4), float4(1, 1, 1, 1))
    assert eq(float4(1, 2, 3, 4) / float4(2, 3, 4, 5), float4(0.5, 2 / 3, 0.75, 0.8))
    assert eq(float4(1, 2, 3, 4) / float4(3, 4, 5, 6), float4(1 / 3, 0.5, 0.6, 2 / 3))
    assert eq(float4(1, 2, 3, 4) / float4(4, 5, 6, 7), float4(0.25, 0.4, 0.5, 4 / 7))

    assert eq(float4(1, 2, 3, 4) + 1, float4(2, 3, 4, 5))
    assert eq(float4(1, 2, 3, 4) + 2, float4(3, 4, 5, 6))
    assert eq(float4(1, 2, 3, 4) + 3, float4(4, 5, 6, 7))
    assert eq(float4(1, 2, 3, 4) + 4, float4(5, 6, 7, 8))

    assert eq(float4(1, 2, 3, 4) - 1, float4(0, 1, 2, 3))
    assert eq(float4(1, 2, 3, 4) - 2, float4(-1, 0, 1, 2))
    assert eq(float4(1, 2, 3, 4) - 3, float4(-2, -1, 0, 1))
    assert eq(float4(1, 2, 3, 4) - 4, float4(-3, -2, -1, 0))

    assert eq(float4(1, 2, 3, 4) * 1, float4(1, 2, 3, 4))
    assert eq(float4(1, 2, 3, 4) * 2, float4(2, 4, 6, 8))
    assert eq(float4(1, 2, 3, 4) * 3, float4(3, 6, 9, 12))
    assert eq(float4(1, 2, 3, 4) * 4, float4(4, 8, 12, 16))

    assert eq(float4(1, 2, 3, 4) / 1, float4(1, 2, 3, 4))
    assert eq(float4(1, 2, 3, 4) / 2, float4(0.5, 1, 1.5, 2))
    assert eq(float4(1, 2, 3, 4) / 3, float4(1 / 3, 2 / 3, 1, 4 / 3))
    assert eq(float4(1, 2, 3, 4) / 4, float4(0.25, 0.5, 0.75, 1))

    assert eq(1 + float4(1, 2, 3, 4), float4(2, 3, 4, 5))
    assert eq(2 + float4(1, 2, 3, 4), float4(3, 4, 5, 6))
    assert eq(3 + float4(1, 2, 3, 4), float4(4, 5, 6, 7))
    assert eq(4 + float4(1, 2, 3, 4), float4(5, 6, 7, 8))

    assert eq(1 - float4(1, 2, 3, 4), float4(0, -1, -2, -3))
    assert eq(2 - float4(1, 2, 3, 4), float4(1, 0, -1, -2))
    assert eq(3 - float4(1, 2, 3, 4), float4(2, 1, 0, -1))
    assert eq(4 - float4(1, 2, 3, 4), float4(3, 2, 1, 0))

    assert eq(1 * float4(1, 2, 3, 4), float4(1, 2, 3, 4))
    assert eq(2 * float4(1, 2, 3, 4), float4(2, 4, 6, 8))
    assert eq(3 * float4(1, 2, 3, 4), float4(3, 6, 9, 12))
    assert eq(4 * float4(1, 2, 3, 4), float4(4, 8, 12, 16))

    assert eq(1 / float4(1, 2, 3, 4), float4(1, 0.5, 1 / 3, 0.25))
    assert eq(2 / float4(1, 2, 3, 4), float4(2, 1, 2 / 3, 0.5))
    assert eq(3 / float4(1, 2, 3, 4), float4(3, 1.5, 1, 0.75))
    assert eq(4 / float4(1, 2, 3, 4), float4(4, 2, 4 / 3, 1))


def test_uint4_logical_ops():
    assert eq(uint4(1, 2, 3, 4) == uint4(1, 2, 3, 4), bool4(True, True, True, True))
    assert eq(uint4(1, 2, 3, 4) == uint4(2, 3, 4, 5), bool4(False, False, False, False))
    assert eq(uint4(1, 2, 3, 4) == uint4(3, 4, 5, 6), bool4(False, False, False, False))
    assert eq(uint4(1, 2, 3, 4) == uint4(4, 5, 6, 7), bool4(False, False, False, False))

    assert eq(uint4(1, 2, 3, 4) != uint4(1, 2, 3, 4), bool4(False, False, False, False))
    assert eq(uint4(1, 2, 3, 4) != uint4(2, 3, 4, 5), bool4(True, True, True, True))
    assert eq(uint4(1, 2, 3, 4) != uint4(3, 4, 5, 6), bool4(True, True, True, True))
    assert eq(uint4(1, 2, 3, 4) != uint4(4, 5, 6, 7), bool4(True, True, True, True))

    # assert eq(uint4(1, 2, 3, 4) > uint4(1, 2, 3, 4), bool4(False, False, False, False))
    # assert eq(uint4(1, 2, 3, 4) > uint4(2, 3, 4, 5), bool4(False, False, False, False))
    # assert eq(uint4(1, 2, 3, 4) > uint4(3, 4, 5, 6), bool4(False, False, False, False))
    # assert eq(uint4(1, 2, 3, 4) > uint4(4, 5, 6, 7), bool4(False, False, False, False))


def test_element_types():
    assert float1(1).element_type == float
    assert float2(1, 2).element_type == float
    assert float3(1, 2, 3).element_type == float
    assert float4(1, 2, 3, 4).element_type == float
    assert float16_t1().element_type == float
    assert float16_t2().element_type == float
    assert float16_t3().element_type == float
    assert float16_t4().element_type == float
    assert int1(1).element_type == int
    assert int2(1, 2).element_type == int
    assert int3(1, 2, 3).element_type == int
    assert int4(1, 2, 3, 4).element_type == int
    assert uint1(1).element_type == int
    assert uint2(1, 2).element_type == int
    assert uint3(1, 2, 3).element_type == int
    assert uint4(1, 2, 3, 4).element_type == int
    assert bool1(True).element_type == bool
    assert bool2(True, False).element_type == bool
    assert bool3(True, False, True).element_type == bool
    assert bool4(True, False, True, False).element_type == bool


def test_shapes():
    assert float1(1).shape == (1,)
    assert float2(1, 2).shape == (2,)
    assert float3(1, 2, 3).shape == (3,)
    assert float4(1, 2, 3, 4).shape == (4,)
    assert float16_t1().shape == (1,)
    assert float16_t2().shape == (2,)
    assert float16_t3().shape == (3,)
    assert float16_t4().shape == (4,)
    assert int1(1).shape == (1,)
    assert int2(1, 2).shape == (2,)
    assert int3(1, 2, 3).shape == (3,)
    assert int4(1, 2, 3, 4).shape == (4,)
    assert uint1(1).shape == (1,)
    assert uint2(1, 2).shape == (2,)
    assert uint3(1, 2, 3).shape == (3,)
    assert uint4(1, 2, 3, 4).shape == (4,)
    assert bool1(True).shape == (1,)
    assert bool2(True, False).shape == (2,)
    assert bool3(True, False, True).shape == (3,)
    assert bool4(True, False, True, False).shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
