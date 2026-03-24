# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy


def test_shape_and_element_types():
    assert spy.quatf(1, 2, 3, 4).element_type == float
    assert spy.quatf(1, 2, 3, 4).shape == (4,)


def test_hashing():
    """Test value-based hash semantics for quaternion types."""
    # Equal values must produce equal hashes and map to the same dict key.
    for a, b in [
        (spy.quatf(1, 2, 3, 4), spy.quatf(1, 2, 3, 4)),
        (spy.quatf(0.5, 0.5, 0.5, 0.5), spy.quatf(0.5, 0.5, 0.5, 0.5)),
    ]:
        assert a == b
        assert hash(a) == hash(b)
        d = {a: "x"}
        assert b in d

    # Distinct values must act as distinct dict keys.
    assert len({spy.quatf(1, 0, 0, 0): 0, spy.quatf(0, 1, 0, 0): 0}) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
