
from slangpy.core import CodeGenBlock, BindContext, BaseType, BaseTypeImpl, AccessType, BoundVariable, Shape

from slangpy.backend import TypeReflection
from slangpy.core.reflection import SlangProgramLayout, SlangType
from slangpy.typeregistry import PYTHON_TYPES


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
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"ThreadIdArg<{self.dims}>")
        if st is None:
            raise ValueError(
                f"Could not find ThreadIdArg slang type. This usually indicates the threadidarg module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape(self.dims)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("threadidarg")
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        return context.layout.vector_type(TypeReflection.ScalarType.uint32, self.dims)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return 1 - len(vector_target_type.shape)


PYTHON_TYPES[ThreadIdArg] = lambda l, x: ThreadIdArgType(l, x.dims)