# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional

from slangpy import Module

from ..basetypes import IModel, Real, RealArray, ArrayKind, TypeLike, AutoSettable, Auto

# Root class for all activations (i.e. that implement IActivation)


class Activation(IModel):
    def __init__(self, act_name: str, width: AutoSettable[int], dtype: AutoSettable[Real], kind: AutoSettable[ArrayKind] = Auto):
        super().__init__()

        self.act_name = act_name
        self.inout_array = RealArray(kind, dtype, width)

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        input_array = RealArray.from_slangtype(input_type)
        self.inout_array.resolve(input_array, must_match=True)

        self.input_type = self.lookup_mandatory_type(module, self.inout_array.name())
        self.output_type = self.input_type
        self.validate(module)

    @property
    def type_name(self) -> str:
        return f"Activation::{self.act_name}<{self.inout_array.dtype}, {self.inout_array.length}>"

    def get_this(self):
        return {"_type": self.type_name}


class Identity(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__("Identity", width, dtype)


class ReLU(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__("ReLU", width, dtype)


class LeakyReLU(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, negative_slope: float = 0.01, dtype: AutoSettable[Real] = Auto):
        super().__init__("LeakyReLU", width, dtype)
        self.negative_slope = negative_slope

    def get_this(self):
        return super().get_this() | {"negativeSlope": self.negative_slope}


class ELU(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, a: float = 1.0, dtype: AutoSettable[Real] = Auto):
        super().__init__("ELU", width, dtype)
        self.a = a

    def get_this(self):
        return super().get_this() | {"a": self.a}


class Swish(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__("Swish", width, dtype)


class Tanh(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__("Tanh", width, dtype)


class Sigmoid(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__("Sigmoid", width, dtype)


class Exp(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__("Exp", width, dtype)
