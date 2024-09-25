
from typing import Any, Optional

from kernelfunctions.core import CodeGenBlock, BaseTypeImpl, AccessType, BoundVariable

from kernelfunctions.backend import Device, TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_SCALAR_TYPES


class WangHashArg:
    """
    Request random uints using a wang hash function. eg
    void myfunc(uint3 input) { }
    """

    def __init__(self, dims: int = 3, seed: int = 0):
        super().__init__()
        self.dims = dims
        self.seed = seed


class WangHashArgType(BaseTypeImpl):
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    @property
    def name(self) -> str:
        return f"WangHashArg<{self.dims}>"

    def shape(self, value: Optional[WangHashArg] = None):
        return (self.dims,)

    @property
    def element_type(self):
        return SLANG_SCALAR_TYPES[TypeReflection.ScalarType.uint32]

    def gen_calldata(self, cgb: CodeGenBlock, input_value: BoundVariable):
        access = input_value.access
        name = input_value.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("wanghasharg")
            cgb.type_alias(f"_{name}", self.name)

    def create_calldata(self, device: Device, input_value: BoundVariable, broadcast: list[bool], data: WangHashArg) -> Any:
        access = input_value.access
        if access[0] == AccessType.read:
            return {
                'seed': data.seed
            }


PYTHON_TYPES[WangHashArg] = lambda x: WangHashArgType(x.dims)
