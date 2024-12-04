from typing import Any
from slangpy.core.enums import PrimType, IOType
from slangpy.core.native import Shape, CallContext, AccessType, TypeReflection
from slangpy.backend import ResourceUsage, Buffer, ResourceView, ResourceViewType, Texture, ResourceType, FormatType, get_format_info
from slangpy.bindings import CodeGenBlock, BindContext, ReturnContext, BaseType, BaseTypeImpl, BoundVariable, PYTHON_TYPES, PYTHON_SIGNATURES, BoundVariableRuntime, get_or_create_type
from slangpy.reflection import SlangProgramLayout, SlangType, TYPE_OVERRIDES, is_matching_array_type
from slangpy.types import NDBuffer, NDDifferentiableBuffer
import slangpy.bindings.typeregistry as tr
import slangpy.reflection as kfr


class RangeType(BaseTypeImpl):
    def __init__(self, layout: SlangProgramLayout):
        super().__init__(layout)
        st = layout.find_type_by_name(f"RangeType")
        if st is None:
            raise ValueError(
                f"Could not find RangeType slang type. This usually indicates the slangpy module has not been imported.")
        self.slang_type = st

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: range) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'start': data.start,
                'stop': data.stop,
                'step': data.step
            }

    def get_shape(self, data: range):
        s = ((data.stop-data.start)//data.step,)
        return s

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return context.layout.scalar_type(TypeReflection.ScalarType.int32)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return 1 - len(vector_target_type.shape)


PYTHON_TYPES[range] = lambda l, x: RangeType(l)
