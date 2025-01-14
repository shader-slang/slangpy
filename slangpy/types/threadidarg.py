# SPDX-License-Identifier: Apache-2.0

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection


class ThreadIdArg:
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, dims: int = 3):
        super().__init__()
        self.dims = dims

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


def thread_id(dims: int):
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    return ThreadIdArg(dims)


class ThreadIdArgMarshall(Marshall):
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
            # cgb.add_import("threadidarg")
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return context.layout.vector_type(TypeReflection.ScalarType.uint32, self.dims)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return 1 - len(vector_target_type.shape)


PYTHON_TYPES[ThreadIdArg] = lambda l, x: ThreadIdArgMarshall(l, x.dims)
