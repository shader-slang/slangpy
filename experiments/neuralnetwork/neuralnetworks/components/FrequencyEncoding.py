# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..basetypes import IModel, Real, RealArray, ArrayKind, SlangType, Auto, AutoSettable, resolve_auto

from slangpy import Module


class FrequencyEncoding(IModel):
    # Frequency encoding that maps each input parameter into a series
    # of sines and cosines with increasing frequency
    def __init__(self, num_octaves: int, input_width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__()

        self.num_octaves = num_octaves
        self._input_width = input_width
        self._dtype = dtype

    def model_init(self, module: Module, input_type: SlangType):
        input_array = RealArray.from_slangtype(input_type)
        self.input_width = resolve_auto(self._input_width, input_array.length)
        self.dtype = resolve_auto(self._dtype, input_array.dtype)

    @property
    def type_name(self) -> str:
        return f"FrequencyEncoding<{self.dtype}, {self.input_width}, {self.num_octaves}>"

    def resolve_input_type(self, module: Module):
        if self._input_width is Auto:
            return None

        return RealArray(
            ArrayKind.array,
            resolve_auto(self._dtype, Real.float),
            self._input_width
        )
