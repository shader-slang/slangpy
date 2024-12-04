from .valuetype import ValueType
from typing import Any
from slangpy.core.enums import PrimType, IOType
from slangpy.core.native import Shape, CallContext, AccessType, TypeReflection
from slangpy.backend import ResourceUsage, Buffer, ResourceView, ResourceViewType, Texture, ResourceType, FormatType, get_format_info
from slangpy.bindings import CodeGenBlock, BindContext, ReturnContext, BaseType, BaseTypeImpl, BoundVariable, PYTHON_TYPES, PYTHON_SIGNATURES, BoundVariableRuntime, get_or_create_type
from slangpy.reflection import SlangProgramLayout, SlangType, TYPE_OVERRIDES, is_matching_array_type
from slangpy.types import NDBuffer, NDDifferentiableBuffer
import slangpy.bindings.typeregistry as tr
import slangpy.reflection as kfr


class StructType(ValueType):

    def __init__(self, layout: SlangProgramLayout, fields: dict[str, BaseType]):
        super().__init__(layout)
        st = layout.find_type_by_name("Unknown")
        if st is None:
            raise ValueError(
                f"Could not find Struct slang type. This usually indicates the slangpy module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape()
        self._fields = fields

    @property
    def has_derivative(self) -> bool:
        return True

    @property
    def is_writable(self) -> bool:
        return True

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        return bound_type

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: BaseType):
        return max(binding.children[name].call_dimensionality for name in self._fields.keys())

    # A struct type should get a dictionary, and just return that for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        else:
            raise ValueError(
                f"Expected dictionary for struct type, got {type(data)}")


def create_vr_type_for_value(layout: SlangProgramLayout, value: dict[str, Any]):
    assert isinstance(value, dict)
    fields = {name: tr.get_or_create_type(layout, type(val), val)
              for name, val in value.items()}
    return StructType(layout, fields)


PYTHON_TYPES[dict] = create_vr_type_for_value
