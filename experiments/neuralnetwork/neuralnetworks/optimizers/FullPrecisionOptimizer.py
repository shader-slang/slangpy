# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import Real

from .Optimizer import Optimizer


class FullPrecisionOptimizer(Optimizer):
    def __init__(self, nested_optimizer: Optimizer, gradient_scale: float = 1.0):
        super().__init__()

        self.nested_optim = nested_optimizer
        self.gradient_scale = gradient_scale

    def get_type_name(self, dtype: Real) -> str:
        return f"FullPrecisionOptimizer<{dtype}, {self.nested_optim.get_type_name(Real.float)}>"

    def get_this(self):
        return {
            "gradientScale": self.gradient_scale,
            "nestedOptim": self.nested_optim.get_this()
        }
