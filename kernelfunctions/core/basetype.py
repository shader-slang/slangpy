

from typing import Any, Optional, TYPE_CHECKING

import numpy.typing as npt

from .native import NativeType, CallMode
from .codegen import CodeGenBlock

if TYPE_CHECKING:
    from .boundvariable import BoundVariable


class BindContext:
    def __init__(self, call_dimensionality: int, call_mode: CallMode):
        super().__init__()
        self.call_dimensionality = call_dimensionality
        self.call_mode = call_mode


class ReturnContext:
    def __init__(self, slang_type: 'BaseType', bind_context: BindContext):
        self.slang_type = slang_type
        self.bind_context = bind_context


class BaseType(NativeType):
    def __init__(self):
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

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        raise NotImplementedError()

    def to_numpy(self, value: Any) -> npt.NDArray[Any]:
        raise NotImplementedError()

    def from_numpy(self, array: npt.ArrayLike) -> Any:
        raise NotImplementedError()

    def update_from_bound_type(self, bound_type: 'BaseType'):
        raise NotImplementedError()
