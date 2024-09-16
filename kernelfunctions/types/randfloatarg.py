

from typing import Any, Optional

from sgl import Device, TypeReflection
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_SCALAR_TYPES
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.enums import AccessType
from kernelfunctions.types.pythonvalue import PythonVariable


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
    def __init__(self):
        super().__init__()

    def name(self, value: Optional[RandFloatArg] = None) -> str:
        if value is not None:
            return f"RandFloatArg<{value.dims}>"
        else:
            return "RandFloatArg<N>"

    def shape(self, value: Optional[RandFloatArg] = None):
        assert value is not None
        return (value.dims,)

    def element_type(self, value: Optional[RandFloatArg] = None):
        return SLANG_SCALAR_TYPES[TypeReflection.ScalarType.float32]

    def gen_calldata(self, cgb: CodeGenBlock, input_value: PythonVariable, name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        if access[0] == AccessType.read:
            cgb.add_import("randfloatarg")
            cgb.type_alias(f"_{name}", f"RandFloatArg<{input_value.shape[0]}>")

    def create_calldata(self, device: Device, input_value: PythonVariable, access: tuple[AccessType, AccessType], data: RandFloatArg) -> Any:
        if access[0] == AccessType.read:
            return {
                'seed': data.seed,
                'min': data.min,
                'max': data.max
            }


PYTHON_TYPES[RandFloatArg] = RandFloatArgType()
