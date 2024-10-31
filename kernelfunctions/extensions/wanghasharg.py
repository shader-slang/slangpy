
from typing import Any, Optional

from kernelfunctions.core import CodeGenBlock, BindContext, BaseTypeImpl, AccessType, BoundVariable, BoundVariableRuntime, CallContext, Shape

from kernelfunctions.backend import TypeReflection
from kernelfunctions.core.basetype import BaseType
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_SCALAR_TYPES, SLANG_VECTOR_TYPES


class WangHashArg:
    """
    Request random uints using a wang hash function. eg
    void myfunc(uint3 input) { }
    """

    def __init__(self, dims: int = 3, seed: int = 0):
        super().__init__()
        self.dims = dims
        self.seed = seed

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


class WangHashArgType(BaseTypeImpl):
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims
        self.element_type = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.uint32]
        self.name = f"WangHashArg<{self.dims}>"

    def get_container_shape(self, value: Optional[WangHashArg] = None) -> Shape:
        return Shape(self.dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("wanghasharg")
            cgb.type_alias(f"_t_{name}", self.name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: WangHashArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed
            }

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        return SLANG_VECTOR_TYPES[TypeReflection.ScalarType.uint32][self.dims]


PYTHON_TYPES[WangHashArg] = lambda x: WangHashArgType(x.dims)
