# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy


def test_shape_and_element_types():
    assert spy.quatf(1, 2, 3, 4).element_type == float
    assert spy.quatf(1, 2, 3, 4).shape == (4,)


def test_quaternion_hashing():
    q1 = spy.quatf(1.0, 2.0, 3.0, 4.0)
    q2 = spy.quatf(1.0, 2.0, 3.0, 4.0)
    assert q1 == q2
    assert hash(q1) == hash(q2)


def test_quaternion_dict_key_usage():
    cache = {}

    key1 = spy.quatf(0.5, 0.5, 0.5, 0.5)
    key2 = spy.quatf(0.5, 0.5, 0.5, 0.5)

    cache[key1] = "quat_value"
    assert key2 in cache
    assert cache[key2] == "quat_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
