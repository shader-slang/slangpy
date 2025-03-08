# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import IModel, Real, ArrayKind, RealArray, TypeLike, Auto, AutoSettable

from slangpy import Module

from typing import Optional


class FrequencyEncoding(IModel):
    # Frequency encoding that maps each input parameter into a series
    # of sines and cosines with increasing frequency
    def __init__(self, input_width: AutoSettable[int], num_octaves: int, dtype: AutoSettable[Real] = Auto):
        super().__init__()

        self.num_octaves = num_octaves
        self.input_array = RealArray(ArrayKind.array, dtype, input_width)

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        input_array = RealArray.from_slangtype(input_type)
        self.input_array.resolve(input_array, must_match=True)

        num_outputs = self.num_octaves * self.input_array.length * 2
        self.output_array = RealArray(self.input_array.kind, self.input_array.dtype, num_outputs)

        self.input_type = self.lookup_mandatory_type(module, self.input_array.name())
        self.output_type = self.lookup_mandatory_type(module, self.output_array.name())
        self.validate(module)

    @property
    def type_name(self) -> str:
        return f"FrequencyEncoding<{self.input_array.dtype}, {self.input_array.length}, {self.num_octaves}>"

    def get_this(self):
        return {"_type": self.type_name}
