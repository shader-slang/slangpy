# SPDX-License-Identifier: Apache-2.0
import pytest

from slangpy.core.utils import parse_generic_signature


def test_no_generic():
    res = parse_generic_signature("MyType")
    assert res == ("MyType", [])


def test_vector():
    res = parse_generic_signature("vector<float,4>")
    assert res == ("vector", ['float', '4'])


def test_vector_padded():
    res = parse_generic_signature("vector<float, 4>")
    assert res == ("vector", ['float', '4'])


def test_ndbuffer_padded():
    res = parse_generic_signature("NDBuffer<vector<float, 4>, 2>")
    assert res == ("NDBuffer", ['vector<float, 4>', '2'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
