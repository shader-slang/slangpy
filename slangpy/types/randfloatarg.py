from typing import Any

from slangpy.core.native import AccessType, CallContext, Shape

from slangpy.backend import TypeReflection
from slangpy.bindings import (BaseType, BaseTypeImpl, BindContext,
                              BoundVariable, BoundVariableRuntime,
                              CodeGenBlock)
from slangpy.bindings.typeregistry import PYTHON_TYPES
from slangpy.reflection import SlangProgramLayout, SlangType


class RandFloatArg:
    """
    Request random floats from a wang hash function. eg
    void myfunc(float3 input) { }
    """

    def __init__(self, min: float, max: float, dim: int, seed: int = 0):
        super().__init__()
        self.seed = seed
        self.min = float(min)
        self.max = float(max)
        self.dims = int(dim)

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


class RandFloatArgType(BaseTypeImpl):
    def __init__(self, layout: SlangProgramLayout, dim: int):
        super().__init__(layout)
        self.dims = dim
        st = layout.find_type_by_name(f"RandFloatArg<{self.dims}>")
        if st is None:
            raise ValueError(
                f"Could not find RandFloatArg slang type. This usually indicates the randfloatarg module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape(self.dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("randfloatarg")
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: RandFloatArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed,
                'min': data.min,
                'max': data.max
            }

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        return context.layout.vector_type(TypeReflection.ScalarType.float32, self.dims)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return 1 - len(vector_target_type.shape)


PYTHON_TYPES[RandFloatArg] = lambda l, x: RandFloatArgType(l, x.dims)
