from slangpy.backend import TypeReflection
from slangpy.bindings import (BaseType, BaseTypeImpl, BindContext,
                              BoundVariable, BoundVariableRuntime,
                              CodeGenBlock, PYTHON_TYPES, AccessType,
                              CallContext, Shape)
from slangpy.reflection import SlangProgramLayout, SlangType


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

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.add_import("threadidarg")
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return context.layout.vector_type(TypeReflection.ScalarType.uint32, self.dims)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return 1 - len(vector_target_type.shape)


PYTHON_TYPES[ThreadIdArg] = lambda l, x: ThreadIdArgType(l, x.dims)
