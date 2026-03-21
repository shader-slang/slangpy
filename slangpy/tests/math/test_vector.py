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
    math,
)


def test_float4_constructor():
    assert float4() == float4(0, 0, 0, 0)
    assert float4(1) == float4(1, 1, 1, 1)
    assert float4(1, 2, 3, 4) == float4(1, 2, 3, 4)


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
    assert a == float4(1, 2, 3, 4)


def test_float4_str():
    assert str(float4(1, 2, 3, 4)) == "{1, 2, 3, 4}"


def test_float4_select():
    b = bool4(True, False, True, False)
    a = float4(1.0, 2.0, 3.0, 4.0)
    c = float4(10.0, 20.0, 30.0, 40.0)
    r = float4(1.0, 20.0, 3.0, 40.0)
    assert math.select(b, a, c) == r


def test_float4_unary_ops():
    assert +float4(1, 2, 3, 4) == float4(1, 2, 3, 4)
    assert +float4(2, 3, 4, 5) == float4(2, 3, 4, 5)
    assert +float4(3, 4, 5, 6) == float4(3, 4, 5, 6)
    assert +float4(4, 5, 6, 7) == float4(4, 5, 6, 7)

    assert -float4(1, 2, 3, 4) == float4(-1, -2, -3, -4)
    assert -float4(2, 3, 4, 5) == float4(-2, -3, -4, -5)
    assert -float4(3, 4, 5, 6) == float4(-3, -4, -5, -6)
    assert -float4(4, 5, 6, 7) == float4(-4, -5, -6, -7)


def test_float4_binary_ops():
    assert float4(1, 2, 3, 4) + float4(1, 2, 3, 4) == float4(2, 4, 6, 8)
    assert float4(1, 2, 3, 4) + float4(2, 3, 4, 5) == float4(3, 5, 7, 9)
    assert float4(1, 2, 3, 4) + float4(3, 4, 5, 6) == float4(4, 6, 8, 10)
    assert float4(1, 2, 3, 4) + float4(4, 5, 6, 7) == float4(5, 7, 9, 11)

    assert float4(1, 2, 3, 4) - float4(1, 2, 3, 4) == float4(0, 0, 0, 0)
    assert float4(1, 2, 3, 4) - float4(2, 3, 4, 5) == float4(-1, -1, -1, -1)
    assert float4(1, 2, 3, 4) - float4(3, 4, 5, 6) == float4(-2, -2, -2, -2)
    assert float4(1, 2, 3, 4) - float4(4, 5, 6, 7) == float4(-3, -3, -3, -3)

    assert float4(1, 2, 3, 4) * float4(1, 2, 3, 4) == float4(1, 4, 9, 16)
    assert float4(1, 2, 3, 4) * float4(2, 3, 4, 5) == float4(2, 6, 12, 20)
    assert float4(1, 2, 3, 4) * float4(3, 4, 5, 6) == float4(3, 8, 15, 24)
    assert float4(1, 2, 3, 4) * float4(4, 5, 6, 7) == float4(4, 10, 18, 28)

    assert float4(1, 2, 3, 4) / float4(1, 2, 3, 4) == float4(1, 1, 1, 1)
    assert float4(1, 2, 3, 4) / float4(2, 3, 4, 5) == float4(0.5, 2 / 3, 0.75, 0.8)
    assert float4(1, 2, 3, 4) / float4(3, 4, 5, 6) == float4(1 / 3, 0.5, 0.6, 2 / 3)
    assert float4(1, 2, 3, 4) / float4(4, 5, 6, 7) == float4(0.25, 0.4, 0.5, 4 / 7)

    assert float4(1, 2, 3, 4) + 1 == float4(2, 3, 4, 5)
    assert float4(1, 2, 3, 4) + 2 == float4(3, 4, 5, 6)
    assert float4(1, 2, 3, 4) + 3 == float4(4, 5, 6, 7)
    assert float4(1, 2, 3, 4) + 4 == float4(5, 6, 7, 8)

    assert float4(1, 2, 3, 4) - 1 == float4(0, 1, 2, 3)
    assert float4(1, 2, 3, 4) - 2 == float4(-1, 0, 1, 2)
    assert float4(1, 2, 3, 4) - 3 == float4(-2, -1, 0, 1)
    assert float4(1, 2, 3, 4) - 4 == float4(-3, -2, -1, 0)

    assert float4(1, 2, 3, 4) * 1 == float4(1, 2, 3, 4)
    assert float4(1, 2, 3, 4) * 2 == float4(2, 4, 6, 8)
    assert float4(1, 2, 3, 4) * 3 == float4(3, 6, 9, 12)
    assert float4(1, 2, 3, 4) * 4 == float4(4, 8, 12, 16)

    assert float4(1, 2, 3, 4) / 1 == float4(1, 2, 3, 4)
    assert float4(1, 2, 3, 4) / 2 == float4(0.5, 1, 1.5, 2)
    assert float4(1, 2, 3, 4) / 3 == float4(1 / 3, 2 / 3, 1, 4 / 3)
    assert float4(1, 2, 3, 4) / 4 == float4(0.25, 0.5, 0.75, 1)

    assert 1 + float4(1, 2, 3, 4) == float4(2, 3, 4, 5)
    assert 2 + float4(1, 2, 3, 4) == float4(3, 4, 5, 6)
    assert 3 + float4(1, 2, 3, 4) == float4(4, 5, 6, 7)
    assert 4 + float4(1, 2, 3, 4) == float4(5, 6, 7, 8)

    assert 1 - float4(1, 2, 3, 4) == float4(0, -1, -2, -3)
    assert 2 - float4(1, 2, 3, 4) == float4(1, 0, -1, -2)
    assert 3 - float4(1, 2, 3, 4) == float4(2, 1, 0, -1)
    assert 4 - float4(1, 2, 3, 4) == float4(3, 2, 1, 0)

    assert 1 * float4(1, 2, 3, 4) == float4(1, 2, 3, 4)
    assert 2 * float4(1, 2, 3, 4) == float4(2, 4, 6, 8)
    assert 3 * float4(1, 2, 3, 4) == float4(3, 6, 9, 12)
    assert 4 * float4(1, 2, 3, 4) == float4(4, 8, 12, 16)

    assert 1 / float4(1, 2, 3, 4) == float4(1, 0.5, 1 / 3, 0.25)
    assert 2 / float4(1, 2, 3, 4) == float4(2, 1, 2 / 3, 0.5)
    assert 3 / float4(1, 2, 3, 4) == float4(3, 1.5, 1, 0.75)
    assert 4 / float4(1, 2, 3, 4) == float4(4, 2, 4 / 3, 1)


def test_uint4_logical_ops():
    assert uint4(1, 2, 3, 4) == uint4(1, 2, 3, 4)
    assert not uint4(1, 2, 3, 4) == uint4(2, 3, 4, 5)
    assert not uint4(1, 2, 3, 4) == uint4(3, 4, 5, 6)
    assert not uint4(1, 2, 3, 4) == uint4(4, 5, 6, 7)

    assert not uint4(1, 2, 3, 4) != uint4(1, 2, 3, 4)
    assert uint4(1, 2, 3, 4) != uint4(2, 3, 4, 5)
    assert uint4(1, 2, 3, 4) != uint4(3, 4, 5, 6)
    assert uint4(1, 2, 3, 4) != uint4(4, 5, 6, 7)


def test_uint4_comparison_ops():
    # Lexicographic comparison
    assert uint4(1, 2, 3, 4) < uint4(1, 2, 3, 5)
    assert uint4(1, 2, 3, 4) < uint4(2, 0, 0, 0)
    assert not uint4(1, 2, 3, 4) < uint4(1, 2, 3, 4)
    assert uint4(1, 2, 3, 4) > uint4(1, 2, 3, 3)
    assert not uint4(1, 2, 3, 4) > uint4(1, 2, 3, 4)
    assert uint4(1, 2, 3, 4) <= uint4(1, 2, 3, 4)
    assert uint4(1, 2, 3, 4) >= uint4(1, 2, 3, 4)


def test_componentwise_comparisons():
    assert math.eq(float3(1, 2, 3), float3(1, 3, 3)) == bool3(True, False, True)
    assert math.ne(float3(1, 2, 3), float3(1, 3, 3)) == bool3(False, True, False)
    assert math.lt(float3(1, 2, 3), float3(2, 2, 2)) == bool3(True, False, False)
    assert math.gt(float3(1, 2, 3), float3(2, 2, 2)) == bool3(False, False, True)
    assert math.le(float3(1, 2, 3), float3(2, 2, 2)) == bool3(True, True, False)
    assert math.ge(float3(1, 2, 3), float3(2, 2, 2)) == bool3(False, True, True)


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


def test_float_vector_hashing():
    """Test that equal float vectors produce equal hashes."""
    # float2
    v1 = float2(1.0, 2.0)
    v2 = float2(1.0, 2.0)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # float3
    v1 = float3(1.0, 2.0, 3.0)
    v2 = float3(1.0, 2.0, 3.0)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # float4
    v1 = float4(1.0, 2.0, 3.0, 4.0)
    v2 = float4(1.0, 2.0, 3.0, 4.0)
    assert v1 == v2
    assert hash(v1) == hash(v2)


def test_int_vector_hashing():
    """Test that equal int vectors produce equal hashes."""
    # int2
    v1 = int2(10, 20)
    v2 = int2(10, 20)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # int3
    v1 = int3(10, 20, 30)
    v2 = int3(10, 20, 30)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # int4
    v1 = int4(10, 20, 30, 40)
    v2 = int4(10, 20, 30, 40)
    assert v1 == v2
    assert hash(v1) == hash(v2)


def test_uint_vector_hashing():
    """Test that equal uint vectors produce equal hashes."""
    # uint2
    v1 = uint2(10, 20)
    v2 = uint2(10, 20)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # uint3
    v1 = uint3(10, 20, 30)
    v2 = uint3(10, 20, 30)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # uint4
    v1 = uint4(10, 20, 30, 40)
    v2 = uint4(10, 20, 30, 40)
    assert v1 == v2
    assert hash(v1) == hash(v2)


def test_bool_vector_hashing():
    """Test that equal bool vectors produce equal hashes."""
    # bool2
    v1 = bool2(True, False)
    v2 = bool2(True, False)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # bool3
    v1 = bool3(True, False, True)
    v2 = bool3(True, False, True)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # bool4
    v1 = bool4(True, False, True, False)
    v2 = bool4(True, False, True, False)
    assert v1 == v2
    assert hash(v1) == hash(v2)


def test_float16_vector_hashing():
    """Test that equal float16_t vectors produce equal hashes."""
    # float16_t2
    v1 = float16_t2(1.0, 2.0)
    v2 = float16_t2(1.0, 2.0)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # float16_t3
    v1 = float16_t3(1.0, 2.0, 3.0)
    v2 = float16_t3(1.0, 2.0, 3.0)
    assert v1 == v2
    assert hash(v1) == hash(v2)

    # float16_t4
    v1 = float16_t4(1.0, 2.0, 3.0, 4.0)
    v2 = float16_t4(1.0, 2.0, 3.0, 4.0)
    assert v1 == v2
    assert hash(v1) == hash(v2)


def test_vector_dict_usage():
    """Test that vectors can be used as dictionary keys."""
    cache = {}

    # Test with float3
    key1 = float3(0.5, 0.5, 0.5)
    key2 = float3(0.5, 0.5, 0.5)
    cache[key1] = "test_value"
    assert key2 in cache
    assert cache[key2] == "test_value"

    # Test with int2
    cache = {}
    key1 = int2(42, 100)
    key2 = int2(42, 100)
    cache[key1] = "another_value"
    assert key2 in cache
    assert cache[key2] == "another_value"

    # Test with bool4
    cache = {}
    key1 = bool4(True, False, True, True)
    key2 = bool4(True, False, True, True)
    cache[key1] = "bool_value"
    assert key2 in cache
    assert cache[key2] == "bool_value"


def test_different_vectors_different_hashes():
    """Test that different vectors act as distinct keys in containers."""
    # Use dictionary insertion to verify distinct keys are properly distinguished
    # float3 case
    v1 = float3(1.0, 2.0, 3.0)
    v2 = float3(4.0, 5.0, 6.0)

    cache = {}
    cache[v1] = "v1"
    cache[v2] = "v2"
    assert len(cache) == 2
    assert v1 in cache
    assert v2 in cache
    assert cache[v1] == "v1"
    assert cache[v2] == "v2"

    # int4 case
    v1 = int4(1, 2, 3, 4)
    v2 = int4(5, 6, 7, 8)

    key_set = {v1, v2}
    assert len(key_set) == 2
    assert v1 in key_set
    assert v2 in key_set


def test_float3_dict_key_pattern_from_properties():
    """Regression test matching the historical float3 dict-key pattern."""
    cache = {}

    assert float3(0.5) not in cache
    cache[float3(0.5)] = "{0.5, 0.5, 0.5}:test"
    assert float3(0.5) in cache
    assert cache[float3(0.5)] == "{0.5, 0.5, 0.5}:test"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
