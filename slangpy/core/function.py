# SPDX-License-Identifier: Apache-2.0
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Union

from slangpy.core.logging import runtime_exception_info
from slangpy.core.native import (CallMode, NativeCallRuntimeOptions,
                                 hash_signature)

import slangpy.reflection as kfr
from slangpy.backend import (CommandBuffer, FunctionReflection,
                             TypeConformance, TypeReflection, uint3)
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES

if TYPE_CHECKING:
    from slangpy.core.calldata import CallData
    from slangpy.core.module import Module
    from slangpy.core.struct import Struct

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
        self._map_args: Optional[tuple[Any]] = None
        self._map_kwargs: Optional[dict[str, Any]] = None
        self._options: Optional[dict[str, Any]] = None
        self._type_conformances: Optional[list[TypeConformance]] = None
        self._mode = CallMode.prim
        self._return_type: Optional[type] = None
        self._constants: Optional[dict[str, Any]] = None
        self._thread_group_size: Optional[uint3] = None

        # Runtime options that affect dispatch only
        self.this: Optional[IThis] = None
        self.uniforms: Optional[list[Union[Callable[[
            'CallData'], Any], dict[str, Any]]]] = None
        self.before_dispatch: Optional[list[TDispatchHook]] = None
        self.after_dispatch: Optional[list[TDispatchHook]] = None

    def _copy(self) -> 'Function':
        res = copy(self)
        res.slangpy_signature = None
        return res

    def attach(self, module: 'Module', func: Union[str, kfr.SlangFunction, list[FunctionReflection]], struct: Optional['Struct'] = None, options: dict[str, Any] = {}) -> None:
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
        self._options = options.copy()
        if not 'implicit_element_casts' in self._options:
            self._options['implicit_element_casts'] = True
        if not 'implicit_tensor_casts' in self._options:
            self._options['implicit_tensor_casts'] = True
        if not 'strict_broadcasting' in self._options:
            self._options['strict_broadcasting'] = True

        # Generate signature for hashing
        # type_parent = self.type_reflection.full_name if self.type_reflection is not None else None
        # self.slangpy_signature = f"[{type_parent or ''}::{self.name},{options_hash}]"

    def bind(self, this: IThis) -> 'Function':
        res = self._copy()
        res.this = this
        return res

    def map(self, *args: Any, **kwargs: Any):
        res = self._copy()
        res._map_args = args
        res._map_kwargs = kwargs
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
        if self.uniforms is None:
            self.uniforms = [uniform_values]
        else:
            self.uniforms = copy(self.uniforms)
            self.uniforms.append(uniform_values)

    def _add_uniform_callback(self, uniform_callback: Callable[['CallData'], Any]):
        if self.uniforms is None:
            self.uniforms = [uniform_callback]
        else:
            self.uniforms = copy(self.uniforms)
            self.uniforms.append(uniform_callback)

    def constants(self, constants: dict[str, Any]):
        res = self._copy()
        if res._constants is None:
            res._constants = constants
        else:
            res._constants = copy(res._constants)
            res._constants.update(constants)
        return res

    def type_conformances(self, type_conformances: list[TypeConformance]):
        res = self._copy()
        if res._type_conformances is None:
            res._type_conformances = type_conformances
        else:
            res._type_conformances = copy(res._type_conformances)
            res._type_conformances.extend(type_conformances)
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
        res._mode = CallMode.bwds
        return res

    def return_type(self, return_type: type):
        res = self._copy()
        res._return_type = return_type
        return res

    def thread_group_size(self, thread_group_size: uint3):
        res = self._copy()
        res._thread_group_size = thread_group_size
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
            opts.uniforms = self.uniforms  # type: ignore
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
            opts.uniforms = self.uniforms  # type: ignore
            return calldata.append_to(opts, command_buffer, *args, **kwargs)
        except ValueError as e:
            self._handle_error(e, calldata)

    def dispatch(self, thread_count: uint3, vars: dict[str, Any] = {}, command_buffer: CommandBuffer | None = None, **kwargs: Any) -> None:
        if ENABLE_CALLDATA_CACHE:
            if self.slangpy_signature is None:
                lines = []
                if self.type_reflection is not None:
                    lines.append(f"{self.type_reflection.full_name}::{self.name}")
                else:
                    lines.append(self.name)
                lines.append(str(self._options))
                lines.append(str(self._map_args))
                lines.append(str(self._map_kwargs))
                lines.append(str(self._type_conformances))
                lines.append(str(self._mode))
                lines.append(str(self._return_type))
                lines.append(str(self._constants))
                lines.append(str(self._thread_group_size))
                self.slangpy_signature = "\n".join(lines)
            sig = hash_signature(
                _cache_value_to_id, self, **kwargs)

            if sig in self.module.dispatch_data_cache:
                dispatch_data = self.module.dispatch_data_cache[sig]
                if dispatch_data.device != self.module.device:
                    raise NameError("Cached CallData is linked to wrong device")
            else:
                from slangpy.core.dispatchdata import DispatchData
                dispatch_data = DispatchData(self, **kwargs)
                self.module.dispatch_data_cache[sig] = dispatch_data
        else:
            from slangpy.core.dispatchdata import DispatchData
            dispatch_data = DispatchData(self, **kwargs)

        opts = NativeCallRuntimeOptions()
        opts.after_dispatch = self.after_dispatch
        opts.before_dispatch = self.before_dispatch
        opts.uniforms = self.uniforms  # type: ignore
        dispatch_data.dispatch(opts, thread_count, vars, command_buffer, **kwargs)

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
            lines.append(str(self._options))
            lines.append(str(self._map_args))
            lines.append(str(self._map_kwargs))
            lines.append(str(self._type_conformances))
            lines.append(str(self._mode))
            lines.append(str(self._return_type))
            lines.append(str(self._constants))
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
