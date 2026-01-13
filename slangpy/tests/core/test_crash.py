# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy


def test_crash():
    spy.crash()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
