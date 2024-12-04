# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING, Any

from slangpy.core.native import CallMode, NativeType

from slangpy.bindings.codegen import CodeGenBlock

if TYPE_CHECKING:
    from slangpy.backend import SlangModule
    from slangpy.bindings.boundvariable import BoundVariable
    from slangpy.reflection import SlangProgramLayout, SlangType


class BindContext:
    def __init__(self, layout: 'SlangProgramLayout', call_mode: CallMode, device_module: 'SlangModule', options: dict[str, Any]):
        super().__init__()
        self.layout = layout
        self.call_dimensionality = -1
        self.call_mode = call_mode
        self.device_module = device_module
        self.options = options


class ReturnContext:
    def __init__(self, slang_type: 'SlangType', bind_context: BindContext):
        super().__init__()
        self.slang_type = slang_type
        self.bind_context = bind_context


class Marshall(NativeType):
    def __init__(self, layout: 'SlangProgramLayout'):
        super().__init__()
        self.slang_type: 'SlangType'

    @property
    def has_derivative(self) -> bool:
        return False

    @property
    def is_writable(self) -> bool:
        return False

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        raise NotImplementedError()

    def reduce_type(self, context: BindContext, dimensions: int):
        raise NotImplementedError()

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        # Default to just casting to itself (i.e. no implicit cast)
        return self.slang_type

    def resolve_dimensionality(self, context: BindContext, binding: 'BoundVariable', vector_target_type: 'SlangType'):
        # Default implementation requires that both this type and the target type
        # have fully known element types. If so, dimensionality is just the difference
        # between the length of the 2 shapes
        if self.slang_type is None:
            raise ValueError(
                f"Cannot resolve dimensionality of {type(self)} without slang type")
        return len(self.slang_type.shape) - len(vector_target_type.shape)
