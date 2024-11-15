

from typing import Any, TYPE_CHECKING

from .native import NativeType, CallMode
from .codegen import CodeGenBlock

if TYPE_CHECKING:
    from .boundvariable import BoundVariable
    from .reflection import SlangProgramLayout, SlangType
    from kernelfunctions.backend import SlangModule


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


class BaseType(NativeType):
    def __init__(self, layout: 'SlangProgramLayout'):
        super().__init__()
        self.slang_type: 'SlangType'

    @property
    def has_derivative(self) -> bool:
        raise NotImplementedError()

    @property
    def is_writable(self) -> bool:
        raise NotImplementedError()

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        raise NotImplementedError()

    def reduce_type(self, context: BindContext, dimensions: int):
        raise NotImplementedError()

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        raise NotImplementedError()

    def resolve_dimensionality(self, context: BindContext, binding: 'BoundVariable', vector_target_type: 'SlangType'):
        raise NotImplementedError()
