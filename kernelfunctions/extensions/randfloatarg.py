

from typing import Any, Optional

from kernelfunctions.core import CodeGenBlock, BindContext, BaseType, BaseTypeImpl, AccessType, BoundVariable, BoundVariableRuntime, CallContext, Shape

from kernelfunctions.backend import TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_SCALAR_TYPES, SLANG_VECTOR_TYPES, get_or_create_type


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
    def __init__(self, dim: int):
        super().__init__()
        self.dims = dim
        self.element_type = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.float32]
        self.name = f"RandFloatArg<{self.dims}>"

    def get_container_shape(self, value: Optional[RandFloatArg] = None) -> Shape:
        return Shape(self.dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("randfloatarg")
            cgb.type_alias(f"_t_{name}", self.name)

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


PYTHON_TYPES[RandFloatArg] = lambda x: RandFloatArgType(x.dims)
