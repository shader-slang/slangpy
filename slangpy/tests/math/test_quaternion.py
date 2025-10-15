# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy


def test_shape_and_element_types():
    assert spy.quatf(1, 2, 3, 4).element_type == float
    assert spy.quatf(1, 2, 3, 4).shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
