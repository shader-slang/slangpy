

from typing import Any, Optional, TYPE_CHECKING

import numpy.typing as npt

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
    def __init__(self, slang_type: 'BaseType', bind_context: BindContext):
        super().__init__()
        self.slang_type = slang_type
        self.bind_context = bind_context


class BaseType(NativeType):
    def __init__(self, layout: 'SlangProgramLayout'):
        super().__init__()

    @property
    def has_derivative(self) -> bool:
        raise NotImplementedError()

    @property
    def is_writable(self) -> bool:
        raise NotImplementedError()

    @property
    def differentiable(self) -> bool:
        raise NotImplementedError()

    @property
    def derivative(self) -> Optional['BaseType']:
        raise NotImplementedError()

    @property
    def python_return_value_type(self) -> type:
        raise NotImplementedError()

    @property
    def fields(self) -> Optional[dict[str, 'BaseType']]:
        raise NotImplementedError()

    def get_slang_type(self, context: 'BindContext') -> 'SlangType':
        raise NotImplementedError()

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        raise NotImplementedError()

    def to_numpy(self, value: Any) -> npt.NDArray[Any]:
        raise NotImplementedError()

    def from_numpy(self, array: npt.ArrayLike) -> Any:
        raise NotImplementedError()

    def reduce_type(self, context: BindContext, dimensions: int):
        raise NotImplementedError()

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        raise NotImplementedError()

    def resolve_dimensionality(self, context: BindContext, vector_target_type: 'SlangType'):
        raise NotImplementedError()
