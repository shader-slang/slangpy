# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import Real

from .Optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-6):
        super().__init__()

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_type_name(self, dtype: Real) -> str:
        return f"AdamOptimizer<{dtype}>"

    def get_this(self):
        return {
            "learningRate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        }
