
from typing import Any, Optional

from kernelfunctions.core import CodeGenBlock, BaseTypeImpl, AccessType, PythonVariable

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
    def __init__(self):
        super().__init__()

    def name(self, value: Optional[WangHashArg] = None) -> str:
        if value is not None:
            return f"WangHashArg<{value.dims}>"
        else:
            return f"WangHashArg<N>"

    def shape(self, value: Optional[WangHashArg] = None):
        assert value is not None
        return (value.dims,)

    def element_type(self, value: Optional[WangHashArg] = None):
        return SLANG_SCALAR_TYPES[TypeReflection.ScalarType.uint32]

    def gen_calldata(self, cgb: CodeGenBlock, input_value: PythonVariable, name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        if access[0] == AccessType.read:
            cgb.add_import("wanghasharg")
            cgb.type_alias(f"_{name}", input_value.primal_type_name)

    def create_calldata(self, device: Device, input_value: PythonVariable, access: tuple[AccessType, AccessType], data: WangHashArg) -> Any:
        if access[0] == AccessType.read:
            return {
                'seed': data.seed
            }


PYTHON_TYPES[WangHashArg] = WangHashArgType()
