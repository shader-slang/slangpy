

from typing import Optional

from kernelfunctions.core import CodeGenBlock, BindContext, BaseTypeImpl, AccessType, BoundVariable, Shape

from kernelfunctions.backend import TypeReflection
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_SCALAR_TYPES


class ThreadIdArg:
    """
    Request the thread id. eg
    void myfunc(int3 input) { }
    """

    def __init__(self, dims: int = 3):
        super().__init__()
        self.dims = dims

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


class ThreadIdArgType(BaseTypeImpl):
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    @property
    def name(self) -> str:
        return f"ThreadIdArg<{self.dims}>"

    def get_container_shape(self, value: Optional[ThreadIdArg] = None) -> Shape:
        return Shape(self.dims)

    @property
    def element_type(self):
        return SLANG_SCALAR_TYPES[TypeReflection.ScalarType.uint32]

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("threadidarg")
            cgb.type_alias(f"_{name}", self.name)


PYTHON_TYPES[ThreadIdArg] = lambda x: ThreadIdArgType(x.dims)
