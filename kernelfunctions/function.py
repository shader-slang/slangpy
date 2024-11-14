from copy import copy
import json
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, Union

from sgl import TypeReflection
from kernelfunctions.backend.slangpynativeemulation import CallMode, NativeCallRuntimeOptions
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


# A callable kernel function. This assumes the function is in the root
# of the module, however a parent in the abstract syntax tree can be provided
# to search for the function in a specific scope.
class Function:
    def __init__(self) -> None:
        super().__init__()
        self.module: 'Module'
        self.slangpy_signature: Optional[str] = None
        self.type_reflection: Optional['TypeReflection']
        self.reflections: list['FunctionReflection']

        # Static options that affect compilation, and thus the signature
        self.map_args: Optional[tuple[Any]] = None
        self.map_kwargs: Optional[dict[str, Any]] = None
        self.options: Optional[dict[str, Any]] = None
        self.type_conformances: Optional[list[TypeConformance]] = None
        self.mode = CallMode.prim
        self.python_return_type: Optional[type] = None

        # Runtime options that affect dispatch only
        self.this: Optional[IThis] = None
        self.uniform_values: Optional[dict[str, Any]] = None
        self.uniform_callbacks: Optional[list[Callable[['CallData'], Any]]] = None
        self.before_dispatch: Optional[list[TDispatchHook]] = None
        self.after_dispatch: Optional[list[TDispatchHook]] = None

    def _copy(self) -> 'Function':
        res = copy(self)
        res.slangpy_signature = None
        return res

    def attach(self, module: 'Module', func: Union[str, kfr.SlangFunction], struct: Optional['Struct'] = None, options: dict[str, Any] = {}) -> None:
        self.module = module

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
        self.options = options.copy()
        if not 'implicit_element_casts' in self.options:
            self.options['implicit_element_casts'] = True
        if not 'implicit_tensor_casts' in self.options:
            self.options['implicit_tensor_casts'] = True
        if not 'strict_broadcasting' in self.options:
            self.options['strict_broadcasting'] = True

        # Generate signature for hashing
        # type_parent = self.type_reflection.full_name if self.type_reflection is not None else None
        # self.slangpy_signature = f"[{type_parent or ''}::{self.name},{options_hash}]"

    def bind(self, this: IThis) -> 'Function':
        res = self._copy()
        res.this = this
        return res

    def map(self, *args: Any, **kwargs: Any):
        res = self._copy()
        res.map_args = args
        res.map_kwargs = kwargs
        return res

    def set(self, *args: Any, **kwargs: Any):
        res = self._copy()

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError(
                "Set accepts either positional or keyword arguments, not both"
            )
        if len(args) > 1:
            raise ValueError(
                "Set accepts only one positional argument (a dictionary or callback)"
            )

        if len(kwargs) > 0:
            res._add_uniform_values(kwargs)
        elif len(args) > 0:
            if callable(args[0]):
                res._add_uniform_callback(args[0])
            elif isinstance(args[0], dict):
                res._add_uniform_values(args[0])
            else:
                raise ValueError(
                    "Set requires a dictionary or callback as a single positional argument"
                )
        else:
            raise ValueError("Set requires at least one argument")

        return res

    def _add_uniform_values(self, uniform_values: dict[str, Any]):
        if self.uniform_values is None:
            self.uniform_values = uniform_values
        else:
            self.uniform_values = copy(self.uniform_values)
            self.uniform_values.update(uniform_values)

    def _add_uniform_callback(self, uniform_callback: Callable[['CallData'], Any]):
        if self.uniform_callbacks is None:
            self.uniform_callbacks = [uniform_callback]
        else:
            self.uniform_callbacks = copy(self.uniform_callbacks)
            self.uniform_callbacks.append(uniform_callback)

    def type_conformance(self, type_conformances: list[TypeConformance]):
        res = self._copy()
        if res.type_conformances is None:
            res.type_conformances = type_conformances
        else:
            res.type_conformances = copy(res.type_conformances)
            res.type_conformances.extend(type_conformances)
        return res

    def hook(self, before_dispatch: Optional[TDispatchHook] = None, after_dispatch: Optional[TDispatchHook] = None):
        res = self._copy()
        if before_dispatch is not None:
            if res.before_dispatch is None:
                res.before_dispatch = [before_dispatch]
            else:
                res.before_dispatch = copy(res.before_dispatch)
                res.before_dispatch.append(before_dispatch)
        if after_dispatch is not None:
            if res.after_dispatch is None:
                res.after_dispatch = [after_dispatch]
            else:
                res.after_dispatch = copy(res.after_dispatch)
                res.after_dispatch.append(after_dispatch)
        return res

    @property
    def bwds_diff(self):
        res = self._copy()
        res.mode = CallMode.bwds
        return res

    def return_type(self, return_type: type):
        res = self._copy()
        res.python_return_type = return_type
        return res

    @property
    def name(self):
        r = self.reflections[0]
        if r.is_overloaded:
            return r.overloads[0].name
        else:
            return r.name

    def as_func(self) -> 'Function':
        return self

    def as_struct(self) -> 'Struct':
        raise ValueError("Cannot convert a function to a struct")

    def call(self, *args: Any, **kwargs: Any) -> Any:
        calldata: Optional['CallData'] = None
        try:
            if self.this:
                args = (self.this,)+args
            calldata = self._build_call_data(*args, **kwargs)
            opts = NativeCallRuntimeOptions()
            opts.after_dispatch = self.after_dispatch
            opts.before_dispatch = self.before_dispatch
            opts.uniform_callbacks = self.uniform_callbacks
            opts.uniform_values = self.uniform_values
            return calldata.call(opts, *args, **kwargs)
        except ValueError as e:
            self._handle_error(e, calldata)

    def append_to(self, command_buffer: CommandBuffer, *args: Any, **kwargs: Any):
        calldata: Optional['CallData'] = None
        try:
            if self.this:
                args = (self.this,)+args
            calldata = self._build_call_data(*args, **kwargs)
            opts = NativeCallRuntimeOptions()
            opts.after_dispatch = self.after_dispatch
            opts.before_dispatch = self.before_dispatch
            opts.uniform_callbacks = self.uniform_callbacks
            opts.uniform_values = self.uniform_values
            return calldata.append_to(opts, command_buffer, *args, **kwargs)
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

    def debug_build_call_data(self, *args: Any, **kwargs: Any):
        return self._build_call_data(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.call(*args, **kwargs)

    def _build_call_data(self, *args: Any, **kwargs: Any):

        if self.slangpy_signature is None:
            lines = []
            if self.type_reflection is not None:
                lines.append(f"{self.type_reflection.full_name}::{self.name}")
            else:
                lines.append(self.name)
            lines.append(str(self.options))
            lines.append(str(self.map_args))
            lines.append(str(self.map_kwargs))
            lines.append(str(self.type_conformances))
            lines.append(str(self.mode))
            lines.append(str(self.python_return_type))
            self.slangpy_signature = "\n".join(lines)

        sig = hash_signature(
            _cache_value_to_id, self, *args, **kwargs)
        if ENABLE_CALLDATA_CACHE and sig in self.module.call_data_cache:
            cd = self.module.call_data_cache[sig]
            if cd.device != self.module.device:
                raise NameError("Cached CallData is linked to wrong device")
            return cd

        from .calldata import CallData
        res = CallData(self, *args, **kwargs)
        if ENABLE_CALLDATA_CACHE:
            self.module.call_data_cache[sig] = res
        return res
