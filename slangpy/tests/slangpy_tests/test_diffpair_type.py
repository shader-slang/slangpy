# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from slangpy.core.enums import PrimType
from slangpy.types.diffpair import DiffPair, diffPair, floatDiffPair


class TestDiffPairConstruction:
    def test_basic(self):
        dp = DiffPair(1.0, 2.0)
        assert dp.primal == 1.0
        assert dp.grad == 2.0
        assert dp.needs_grad is True

    def test_none_primal_defaults_to_zero(self):
        dp = DiffPair(None, 3.0)
        assert dp.primal == 0.0
        assert dp.grad == 3.0

    def test_none_grad_defaults_to_type_default(self):
        dp = DiffPair(5.0, None)
        assert dp.primal == 5.0
        assert isinstance(dp.grad, float)
        assert dp.grad == 0.0

    def test_both_none(self):
        dp = DiffPair(None, None)
        assert dp.primal == 0.0
        assert dp.grad == 0.0

    def test_needs_grad_false(self):
        dp = DiffPair(1.0, 2.0, needs_grad=False)
        assert dp.needs_grad is False

    def test_int_types(self):
        dp = DiffPair(5, None)
        assert dp.primal == 5
        assert isinstance(dp.grad, int)
        assert dp.grad == 0


class TestDiffPairGetSet:
    def test_get_primal(self):
        dp = DiffPair(10.0, 20.0)
        assert dp.get(PrimType.primal) == 10.0

    def test_get_derivative(self):
        dp = DiffPair(10.0, 20.0)
        assert dp.get(PrimType.derivative) == 20.0

    def test_set_primal(self):
        dp = DiffPair(0.0, 0.0)
        dp.set(PrimType.primal, 42.0)
        assert dp.primal == 42.0

    def test_set_derivative(self):
        dp = DiffPair(0.0, 0.0)
        dp.set(PrimType.derivative, 99.0)
        assert dp.grad == 99.0


class TestDiffPairSignature:
    def test_float_signature(self):
        dp = DiffPair(1.0, 2.0)
        sig = dp.slangpy_signature
        assert sig == "[float,float,True]"

    def test_int_signature(self):
        dp = DiffPair(1, 2)
        assert dp.slangpy_signature == "[int,int,True]"

    def test_no_grad_signature(self):
        dp = DiffPair(1.0, 2.0, needs_grad=False)
        assert dp.slangpy_signature == "[float,float,False]"


class TestFactoryFunctions:
    def test_diffPair_defaults(self):
        dp = diffPair()
        assert dp.primal == 0.0
        assert dp.grad == 0.0
        assert dp.needs_grad is True

    def test_diffPair_with_values(self):
        dp = diffPair(3.0, 4.0)
        assert dp.primal == 3.0
        assert dp.grad == 4.0

    def test_diffPair_no_grad(self):
        dp = diffPair(1.0, 2.0, needs_grad=False)
        assert dp.needs_grad is False

    def test_floatDiffPair_defaults(self):
        dp = floatDiffPair()
        assert dp.primal == 0.0
        assert dp.grad == 1.0
        assert dp.needs_grad is True

    def test_floatDiffPair_with_values(self):
        dp = floatDiffPair(5.0, 10.0)
        assert dp.primal == 5.0
        assert dp.grad == 10.0

    def test_floatDiffPair_no_grad(self):
        dp = floatDiffPair(needs_grad=False)
        assert dp.needs_grad is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
