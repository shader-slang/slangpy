from typing import Any

from kernelfunctions.core import BaseType, Shape

from kernelfunctions.backend import TypeReflection
from kernelfunctions.core.basetype import BindContext
from kernelfunctions.core.reflection import SlangProgramLayout
from kernelfunctions.typeregistry import PYTHON_TYPES
import kernelfunctions.typeregistry as tr

from .valuetype import ValueType


class StructType(ValueType):

    def __init__(self, layout: SlangProgramLayout, fields: dict[str, BaseType]):
        super().__init__(layout)
        self.slang_type = layout.find_type_by_name("Unknown")
        self.concrete_shape = Shape()
        self._fields = fields

    @property
    def has_derivative(self) -> bool:
        return True

    @property
    def is_writable(self) -> bool:
        return True

    @property
    def fields(self):
        return self._fields

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        return bound_type


def create_vr_type_for_value(layout: SlangProgramLayout, value: dict[str, Any]):
    assert isinstance(value, dict)
    fields = {name: tr.get_or_create_type(layout, type(val), val)
              for name, val in value.items()}
    return StructType(layout, fields)


PYTHON_TYPES[dict] = create_vr_type_for_value
