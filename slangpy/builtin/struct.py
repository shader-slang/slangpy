# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, cast

from slangpy.core.native import Shape

import slangpy.bindings.typeregistry as tr
from slangpy.bindings import PYTHON_TYPES, Marshall, BindContext, BoundVariable
from slangpy.reflection import SlangProgramLayout, SlangType

from .value import ValueMarshall


class StructMarshall(ValueMarshall):

    def __init__(self, layout: SlangProgramLayout, fields: dict[str, Marshall], slang_type: Optional[SlangType] = None):
        super().__init__(layout)
        if slang_type is None:
            slang_type = layout.find_type_by_name("Unknown")
            if slang_type is None:
                raise ValueError(
                    f"Could not find Struct slang type. This usually indicates the slangpy module has not been imported.")
        self.slang_type = slang_type
        self.concrete_shape = Shape()
        self._fields = fields

    @property
    def has_derivative(self) -> bool:
        return True

    @property
    def is_writable(self) -> bool:
        return True

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return bound_type

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: SlangType):
        assert binding.children is not None
        if not self._fields:
            return 0
        return max(cast(int, binding.children[name].call_dimensionality)
                   for name in self._fields.keys())

    # A struct type should get a dictionary, and just return that for raw dispatch

    def create_dispatchdata(self, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        else:
            raise ValueError(f"Expected dictionary for struct type, got {type(data)}")


def create_vr_type_for_value(layout: SlangProgramLayout, value: dict[str, Any]):
    assert isinstance(value, dict)
    slang_type: Optional[SlangType] = None

    if "_type" in value:
        type_name = value["_type"]
        slang_type = layout.find_type_by_name(type_name)
        if slang_type is None:
            raise ValueError(f"Could not find type {type_name}")
        del value["_type"]

    fields = {name: tr.get_or_create_type(layout, type(val), val)
              for name, val in value.items()}

    return StructMarshall(layout, fields, slang_type)


PYTHON_TYPES[dict] = create_vr_type_for_value
