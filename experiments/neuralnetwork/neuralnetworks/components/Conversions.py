# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional

from slangpy import Module

from ..basetypes import IModel, Real, ArrayKind, RealArray, TypeLike, Auto, AutoSettable


class Convert(IModel):
    def __init__(self, input_kind: AutoSettable[ArrayKind], input_dtype: AutoSettable[Real], output_kind: AutoSettable[ArrayKind], output_dtype: AutoSettable[Real], width: AutoSettable[int]):
        super().__init__()
        self.input_array = RealArray(input_kind, input_dtype, width)
        self.output_array = RealArray(output_kind, output_dtype, width)

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        input_array = RealArray.from_slangtype(input_type)
        self.input_array.resolve(input_array, must_match=True)
        self.output_array.resolve(self.input_array)

        if self.input_array.length != self.output_array.length:
            self.model_error("Number of input and output elements for ConvertArray do not match "
                             f"(received '{self.input_array}' and '{self.output_array}')")
        if self.input_array.dtype != self.output_array.dtype and self.input_array.kind != self.output_array.kind:
            self.model_error(f"Can't simultaneously convert array kind and element type "
                             f"(received '{self.input_array}' and '{self.output_array}')")

        self.input_type = self.lookup_mandatory_type(module, input_array.name())
        self.output_type = self.lookup_mandatory_type(module, self.output_array.name())
        self.validate(module)

    @property
    def type_name(self) -> str:
        assert self.input_array.length == self.output_array.length

        if self.input_array.kind != self.output_array.kind:
            assert self.input_array.dtype == self.output_array.dtype

            if self.output_array.kind == ArrayKind.array:
                base_name = "ConvertToArray"
            elif self.output_array.kind == ArrayKind.vector:
                base_name = "ConvertToVector"
            elif self.output_array.kind == ArrayKind.coopvec:
                base_name = "ConvertToCoopVec"
            type_name = f"{base_name}<{self.input_array.dtype}, {self.input_array.length}>"
        elif self.input_array.dtype != self.output_array.dtype:
            type_name = f"ConvertArrayPrecision<{self.input_array.dtype}, {self.output_array.dtype}, {self.input_array.length}>"
        else:
            type_name = f"Identity<{self.input_array}>"

        return type_name

    def get_this(self):
        return {"_type": self.type_name}

    @staticmethod
    def to_array_kind(kind: ArrayKind):
        return Convert(Auto, Auto, kind, Auto, Auto)

    @staticmethod
    def to_coopvec():
        return Convert.to_array_kind(ArrayKind.coopvec)

    @staticmethod
    def to_array():
        return Convert.to_array_kind(ArrayKind.array)

    @staticmethod
    def to_vector():
        return Convert.to_array_kind(ArrayKind.vector)

    @staticmethod
    def to_precision(dtype: Real):
        return Convert(Auto, Auto, Auto, dtype, Auto)
