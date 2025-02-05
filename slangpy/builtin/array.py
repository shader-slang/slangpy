# SPDX-License-Identifier: Apache-2.0
from typing import Any, Union

from slangpy.core.native import Shape

import slangpy.bindings.typeregistry as tr
from slangpy.builtin.value import ValueMarshall
from slangpy.reflection import SlangType, SlangProgramLayout


class ArrayMarshall(ValueMarshall):
    def __init__(self, layout: SlangProgramLayout, element_type: SlangType, shape: Shape):
        super().__init__(layout)

        st = element_type
        for dim in reversed(shape.as_tuple()):
            st = layout.array_type(st, dim)
        self.slang_type = st
        self.concrete_shape = shape


def _distill_array(layout: SlangProgramLayout, value: Union[list[Any], tuple[Any]]):
    shape = (len(value), )
    while True:
        if len(value) == 0:
            return shape, tr.get_or_create_type(layout, int).slang_type
        if not isinstance(value[0], (list, tuple)):
            et = tr.get_or_create_type(layout, type(value[0]), value[0]).slang_type
            return shape, et

        N = len(value[0])
        if not all(len(x) == N for x in value):
            raise ValueError("Elements of nested array must all have equal lengths")

        shape = shape + (N, )
        value = value[0]


def python_lookup_array_type(layout: SlangProgramLayout, value: list[Any]):
    shape, et = _distill_array(layout, value)
    return ArrayMarshall(layout, et, Shape(shape))


tr.PYTHON_TYPES[list] = python_lookup_array_type
