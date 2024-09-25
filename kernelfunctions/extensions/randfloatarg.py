

from typing import Any, Optional

from kernelfunctions.core import CodeGenBlock, BaseTypeImpl, AccessType, BoundVariable

from kernelfunctions.backend import Device, TypeReflection
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


class RandFloatArgType(BaseTypeImpl):
    def __init__(self, dim: int):
        super().__init__()
        self.dims = dim

    @property
    def name(self) -> str:
        return f"RandFloatArg<{self.dims}>"

    def shape(self, value: Optional[RandFloatArg] = None):
        return (self.dims,)

    @property
    def element_type(self):
        return SLANG_SCALAR_TYPES[TypeReflection.ScalarType.float32]

    def gen_calldata(self, cgb: CodeGenBlock, input_value: BoundVariable):
        access = input_value.access
        name = input_value.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("randfloatarg")
            cgb.type_alias(f"_{name}", input_value.python.primal_type_name)

    def create_calldata(self, device: Device, input_value: BoundVariable, broadcast: list[bool], data: RandFloatArg) -> Any:
        access = input_value.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed,
                'min': data.min,
                'max': data.max
            }


PYTHON_TYPES[RandFloatArg] = lambda x: RandFloatArgType(x.dims)
