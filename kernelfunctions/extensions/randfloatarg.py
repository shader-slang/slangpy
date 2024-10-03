

from typing import Any, Optional

from kernelfunctions.core import CodeGenBlock, BaseTypeImpl, AccessType, BoundVariable, BoundVariableRuntime, CallContext, NativeShape

from kernelfunctions.backend import TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_SCALAR_TYPES


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

    @property
    def name(self) -> str:
        return f"RandFloatArg<{self.dims}>"

    def get_container_shape(self, value: Optional[RandFloatArg] = None) -> NativeShape:
        return NativeShape(self.dims)

    @property
    def element_type(self):
        return SLANG_SCALAR_TYPES[TypeReflection.ScalarType.float32]

    def gen_calldata(self, cgb: CodeGenBlock, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("randfloatarg")
            cgb.type_alias(f"_{name}", self.name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: RandFloatArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed,
                'min': data.min,
                'max': data.max
            }


PYTHON_TYPES[RandFloatArg] = lambda x: RandFloatArgType(x.dims)
