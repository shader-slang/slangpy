import json
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, Union
from kernelfunctions.core import hash_signature

from kernelfunctions.backend import FunctionReflection, CommandBuffer, TypeConformance
from kernelfunctions.core.logging import runtime_exception_info
from kernelfunctions.shapes import TShapeOrTuple
from kernelfunctions.typeregistry import PYTHON_SIGNATURES

import kernelfunctions.core.reflection as kfr

if TYPE_CHECKING:
    from kernelfunctions.calldata import CallData
    from kernelfunctions.struct import Struct
    from kernelfunctions.module import Module

ENABLE_CALLDATA_CACHE = True

TDispatchHook = Callable[[dict[str, Any]], None]


def _cache_value_to_id(val: Any) -> str:
    cb = PYTHON_SIGNATURES.get(type(val))
    if cb is None:
        return ""
    else:
        return cb(val)


class IThis(Protocol):
    def get_this(self) -> Any:
        ...

    def update_this(self, value: Any) -> None:
        ...


class FunctionChainBase:
    def __init__(self, parent: Optional["FunctionChainBase"], module: 'Module') -> None:
        super().__init__()
        self.module = module
        self.parent = parent
        self.this: Any = parent.this if parent is not None else None
        self.slangpy_signature = f"{parent.slangpy_signature}." if parent is not None else ""

    def call(self, *args: Any, **kwargs: Any) -> Any:
        calldata: Optional['CallData'] = None
        try:
            if self.this:
                args = (self.this,)+args
            calldata = self._build_call_data(*args, **kwargs)
            return calldata.call(*args, **kwargs)
        except ValueError as e:
            self._handle_error(e, calldata)

    def append_to(self, command_buffer: CommandBuffer, *args: Any, **kwargs: Any):
        calldata: Optional['CallData'] = None
        try:
            if self.this:
                args = (self.this,)+args
            calldata = self._build_call_data(*args, **kwargs)
            return calldata.append_to(command_buffer, *args, **kwargs)
        except ValueError as e:
            self._handle_error(e, calldata)

    def _handle_error(self, e: ValueError, calldata: Optional['CallData']):
        if len(e.args) != 1 or not isinstance(e.args[0], dict):
            raise e
        if not 'message' in e.args[0] or not 'source' in e.args[0]:
            raise e
        msg = e.args[0]['message']
        source = e.args[0]['source']
        raise ValueError(
            f"Exception dispatching kernel: {msg}\n."
            f"{runtime_exception_info(calldata.runtime, [], source)}\n")  # type: ignore

    @property
    def bwds_diff(self):
        return FunctionChainBwdsDiff(self)

    def set(self, *args: Any, **kwargs: Any):
        return FunctionChainSet(self, *args, **kwargs)

    def transform_output(self, transforms: dict[str, TShapeOrTuple]):
        return FunctionChainOutputTransform(self, transforms)

    def map(self, *args: Any, **kwargs: Any):
        return FunctionChainMap(self, *args, **kwargs)

    def instance(self, this: IThis):
        return FunctionChainThis(self, this)

    def hook(self, before_dispatch: Optional[TDispatchHook] = None, after_dispatch: Optional[TDispatchHook] = None):
        return FunctionChainHook(self, before_dispatch, after_dispatch)

    def return_type(self, return_type: Any):
        return FunctionChainReturnType(self, return_type)

    def type_conformance(self, type_conformances: list[TypeConformance]):
        return FunctionChainTypeConformance(self, type_conformances)

    def debug_build_call_data(self, *args: Any, **kwargs: Any):
        return self._build_call_data(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.call(*args, **kwargs)

    def _build_call_data(self, *args: Any, **kwargs: Any):
        sig = hash_signature(
            _cache_value_to_id, self, *args, **kwargs)
        if ENABLE_CALLDATA_CACHE and sig in self.module.call_data_cache:
            cd = self.module.call_data_cache[sig]
            if cd.device != self.module.device:
                raise NameError("Cached CallData is linked to wrong device")
            return cd

        chain = []
        current = self
        while current is not None:
            chain.append(current)
            current = current.parent
        chain.reverse()

        from .calldata import CallData
        res = CallData(chain, *args, **kwargs)
        if ENABLE_CALLDATA_CACHE:
            self.module.call_data_cache[sig] = res
        return res


class FunctionChainMap(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase, *args: Any, **kwargs: Any) -> None:
        super().__init__(parent, parent.module)
        self.args = args
        self.kwargs = kwargs


class FunctionChainSet(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase, *args: Any, **kwargs: Any) -> None:
        super().__init__(parent, parent.module)
        self.props: Optional[dict[str, Any]] = None
        self.callback: Optional[Callable] = None  # type: ignore

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError(
                "Set accepts either positional or keyword arguments, not both"
            )
        if len(args) > 1:
            raise ValueError(
                "Set accepts only one positional argument (a dictionary or callback)"
            )

        if len(kwargs) > 0:
            self.props = kwargs
        elif len(args) > 0:
            if callable(args[0]):
                self.callback = args[0]
            elif isinstance(args[0], dict):
                self.props = args[0]
            else:
                raise ValueError(
                    "Set requires a dictionary or callback as a single positional argument"
                )
        else:
            raise ValueError("Set requires at least one argument")


class FunctionChainOutputTransform(FunctionChainBase):
    def __init__(
        self, parent: FunctionChainBase, transforms: dict[str, TShapeOrTuple]
    ) -> None:
        super().__init__(parent, parent.module)
        self.transforms = transforms


class FunctionChainThis(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase, this: IThis) -> None:
        super().__init__(parent, parent.module)
        self.this = this


class FunctionChainBwdsDiff(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase) -> None:
        super().__init__(parent, parent.module)


class FunctionChainHook(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase, before_dispatch: Optional[TDispatchHook], after_dispatch: Optional[TDispatchHook]) -> None:
        super().__init__(parent, parent.module)
        self.before_dispatch = before_dispatch
        self.after_dispatch = after_dispatch


class FunctionChainReturnType(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase, return_type: Any) -> None:
        super().__init__(parent, parent.module)
        self.return_type = return_type


class FunctionChainTypeConformance(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase, type_conformances: list[TypeConformance]) -> None:
        super().__init__(parent, parent.module)
        self.type_conformances = type_conformances
        self.slangpy_signature += f"[{','.join([str(tc) for tc in type_conformances])}]"

# A callable kernel function. This assumes the function is in the root
# of the module, however a parent in the abstract syntax tree can be provided
# to search for the function in a specific scope.


class Function(FunctionChainBase):
    def __init__(
        self,
        module: 'Module',
        struct: Optional['Struct'],
        func: Union[str, list[FunctionReflection], kfr.SlangFunction],
        options: dict[str, Any] = {},
    ) -> None:
        super().__init__(None, module)
        self.module = module
        self.options = options

        if isinstance(func, str):
            if struct is None:
                sf = module.layout.find_function_by_name(func)
            else:
                sf = module.layout.find_function_by_name_in_type(struct.struct, func)
            if sf is None:
                raise ValueError(f"Function '{func}' not found")
            func = sf

        if isinstance(func, kfr.SlangFunction):
            func_reflections = [func.reflection]
        else:
            func_reflections = func

        # Store function reflections (should normally be 1 unless forced to do AST based search)
        self.reflections = func_reflections

        # Store type parent name if found
        if struct is not None:
            self.type_reflection = struct.struct.type_reflection
        else:
            self.type_reflection = None

        # Calc hash of input options for signature
        if not 'implicit_element_casts' in self.options:
            self.options['implicit_element_casts'] = True
        if not 'implicit_tensor_casts' in self.options:
            self.options['implicit_tensor_casts'] = True
        if not 'strict_broadcasting' in self.options:
            self.options['strict_broadcasting'] = True
        options_hash = json.dumps(self.options)

        # Generate signature for hashing
        type_parent = self.type_reflection.full_name if self.type_reflection is not None else None
        self.slangpy_signature = f"[{type_parent or ''}::{self.name},{options_hash}]"

    @property
    def name(self):
        return self.reflections[0].name

    def as_func(self) -> 'Function':
        return self

    def as_struct(self) -> 'Struct':
        raise ValueError("Cannot convert a function to a struct")
