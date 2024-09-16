

from typing import Optional

from sgl import TypeReflection
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_SCALAR_TYPES
from kernelfunctions.core.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.enums import AccessType
from kernelfunctions.core.pythonvariable import PythonVariable


class ThreadIdArg:
    """
    Request the thread id. eg
    void myfunc(int3 input) { }
    """

    def __init__(self, dims: int = 3):
        super().__init__()
        self.dims = dims


class ThreadIdArgType(BaseTypeImpl):
    def __init__(self):
        super().__init__()

    def name(self, value: Optional[ThreadIdArg] = None) -> str:
        if value is not None:
            return f"ThreadIdArg<{value.dims}>"
        else:
            return f"ThreadIdArg<N>"

    def shape(self, value: Optional[ThreadIdArg] = None):
        assert value is not None
        return (value.dims,)

    def element_type(self, value: Optional[ThreadIdArg] = None):
        return SLANG_SCALAR_TYPES[TypeReflection.ScalarType.uint32]

    def gen_calldata(self, cgb: CodeGenBlock, input_value: PythonVariable, name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        if access[0] == AccessType.read:
            cgb.add_import("threadidarg")
            cgb.type_alias(f"_{name}", f"ThreadIdArg<{input_value.shape[0]}>")


PYTHON_TYPES[ThreadIdArg] = ThreadIdArgType()
