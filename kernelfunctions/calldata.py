import os
from typing import Any, Optional, Union, cast
import numpy as np
import numpy.typing as npt
import sgl
import hashlib

from kernelfunctions.buffer import StructuredBuffer
from kernelfunctions.function import (
    Function,
    FunctionChainBase,
    FunctionChainInputTransform,
    FunctionChainOutputTransform,
    FunctionChainSet,
)
from kernelfunctions.shapes import (
    TConcreteShape,
    TLooseShape,
    build_indexer,
    calculate_argument_shapes,
)
import kernelfunctions.translation as kft
import kernelfunctions.codegen as cg
from kernelfunctions.utils import ScalarDiffPair, ScalarRef, is_differentiable_buffer

TYPES = r"""
int _idx<let N: int>(int[N] index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) {
        idx += index[i] * stride[i];
    }
    return idx;
}

struct TensorBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) {
        return buffer[_idx(index, strides)];
    }
    __subscript(int[N] index)->T
    {
        get { return get(index); }
    }
}

struct RWTensorBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) {
        return buffer[_idx(index, strides)];
    }
    void set(int[N] index, T value) {
        buffer[_idx(index, strides)] = value;
    }
    __subscript(int[N] index)->T
    {
        get { return get(index); }
        set { set(index, newValue); }
    }
}
"""


def match_function_overload_to_python_args(
    overload: sgl.FunctionReflection,
    deep_check: bool,
    backwards: bool,
    *args: Any,
    **kwargs: Any,
) -> Optional[dict[str, kft.Argument]]:
    overload_parameters = overload.parameters

    # If calculating backwards pass and the function has a return value, need to treat the
    # return value as an input parameter.
    if backwards and overload.return_type is not None:
        if "_result" in kwargs:
            del kwargs["_result"]
        else:
            args = args[:-1]

    # If there are more positional arguments than parameters, it can't match.
    if len(args) > len(overload_parameters):
        return None

    # Dictionary of slang arguments and corresponding python arguments
    handled_args: dict[str, kft.Argument] = {
        x.name: cast(kft.Argument, None) for x in overload_parameters
    }

    # Positional arguments must all match perfectly
    for i, arg in enumerate(args):
        param = overload_parameters[i]
        converter = kft.try_create_argument(param, arg, deep_check=deep_check)
        if not converter:
            return None
        handled_args[param.name] = converter

    # Pair up kw arguments with slang arguments
    overload_parameters_dict = {x.name: x for x in overload_parameters}
    for name, arg in kwargs.items():
        param = overload_parameters_dict.get(name)
        if param is None:
            return None
        converter = kft.try_create_argument(param, arg, deep_check=deep_check)
        if not converter:
            return None
        handled_args[param.name] = converter

    # Check if all arguments have been handled
    for arg in handled_args.values():
        if arg is None:
            return None

    return handled_args


def calculate_broadcast_dimensions(
    shapes: list[Optional[tuple[int, ...]]]
) -> tuple[int, ...]:
    # Get maximum number of dimensions
    max_dims = 0
    for shape in shapes:
        if shape is not None:
            max_dims = max(max_dims, len(shape))

    # Verify bathces are compatible
    dim_sizes = [0 for _ in range(max_dims)]
    for shape in shapes:
        if shape is not None:
            shape_dims = len(shape)
            for i in range(shape_dims):
                global_dim_index = max_dims - i - 1
                shape_dim_index = shape_dims - i - 1
                global_dim = dim_sizes[global_dim_index]
                shape_dim = shape[shape_dim_index]
                if global_dim > 1 and shape_dim > 1 and global_dim != shape_dim:
                    raise ValueError("Incompatible batch sizes")
                dim_sizes[global_dim_index] = max(global_dim, shape_dim)

    return tuple(dim_sizes)


class CallData:
    def __init__(
        self,
        chain: list["FunctionChainBase"],
        backwards: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(chain[0], Function):
            raise ValueError("First entry in chain should be a function")
        self.function = chain[0]
        self.chain = chain
        self.backwards = backwards
        self.args = args
        self.kwargs = kwargs
        self.input_transforms: dict[str, TConcreteShape] = {}
        self.outut_transforms: dict[str, TConcreteShape] = {}
        sets = {}
        for item in chain:
            if isinstance(item, FunctionChainSet):
                if item.props is not None:
                    sets.update(item.props)
                elif item.callback is not None:
                    sets.update(item.callback(self))
                else:
                    raise ValueError(
                        "FunctionChainSet must have either a props or callback"
                    )
            if isinstance(item, FunctionChainInputTransform):
                self.input_transforms.update(item.transforms)
            if isinstance(item, FunctionChainOutputTransform):
                self.outut_transforms.update(item.transforms)

        self.sets = sets

        # Find an overload that matches the arguments, and pull out the mapping of slang variables to python variables.
        parameters = None
        overload = None
        for ast_function in self.function.ast_functions:
            parameters = match_function_overload_to_python_args(
                ast_function.as_function(), False, backwards, *args, **kwargs
            )
            if parameters:
                overload = ast_function
                parameters = parameters
                break
        if parameters is None or overload is None:
            raise ValueError("No matching overload found")
        self.parameters = parameters
        self.overload = overload

        # For backwards calls, return value is passed in as an argument, for
        # forwards calls, it is still relevant, but has no initial python value
        self.return_value = None
        func_return_type = self.overload.as_function().return_type
        if func_return_type is not None and func_return_type.name in ["float", "int"]:
            if not backwards:
                self.return_value = kft.ReturnValue(
                    self.overload.as_function(),
                    None,
                    kft.convert_type(self.overload.as_function().return_type),
                )
            else:
                if "_result" in kwargs:
                    self.return_value = kft.ReturnValue(
                        self.overload.as_function(),
                        kwargs["_result"],
                        kft.convert_type(self.overload.as_function().return_type),
                    )
                else:
                    self.return_value = kft.ReturnValue(
                        self.overload.as_function(),
                        args[-1],
                        kft.convert_type(self.overload.as_function().return_type),
                    )

        # Record whether function is differentiable
        self.differentiable = self.overload.as_function().has_modifier(
            sgl.ModifierID.differentiable
        )
        if backwards and not self.differentiable:
            raise ValueError("Function is not differentiable")

        all_params = list(self.parameters.values()) + (
            [self.return_value] if self.return_value is not None else []
        )

        # Calculate call shape
        self.shape = calculate_argument_shapes(
            [x.parameter_shape for x in all_params],
            [x.value_shape for x in all_params],
            [self.input_transforms.get(x.name) for x in all_params],
            [self.outut_transforms.get(x.name) for x in all_params],
        )

        # Store total threads
        self.total_threads = 1
        self.strides = []
        for dim in reversed(self.shape["call_shape"]):
            self.strides.append(self.total_threads)
            self.total_threads *= dim
        self.strides.reverse()

        # Build variable names list for call data struct.
        variable_name_list = []
        ptrs: list[str] = []
        ptws: list[str] = []
        targs: list[str] = []
        if not self.backwards:
            variable_name_list = []
            for i, x in enumerate(self.parameters):
                self._append_forward_variable_code_for_argument(
                    self.parameters[x],
                    variable_name_list,
                    ptrs,
                    ptws,
                    targs,
                    build_indexer(
                        self.shape["call_shape"], self.shape["arg_shapes"][i]
                    ),
                )
            if self.return_value is not None:
                self._append_forward_variable_code_for_return_value(
                    self.return_value,
                    variable_name_list,
                    ptrs,
                    ptws,
                    targs,
                    build_indexer(
                        self.shape["call_shape"],
                        self.shape["arg_shapes"][len(self.parameters)],
                    ),
                )
        else:
            for i, x in enumerate(self.parameters):
                self._append_backward_variable_code_for_argument(
                    self.parameters[x],
                    variable_name_list,
                    ptrs,
                    ptws,
                    targs,
                    build_indexer(
                        self.shape["call_shape"], self.shape["arg_shapes"][i]
                    ),
                )
            if self.return_value is not None:
                self._append_backward_variable_code_for_return_value(
                    self.return_value,
                    variable_name_list,
                    ptrs,
                    ptws,
                    targs,
                    build_indexer(
                        self.shape["call_shape"],
                        self.shape["arg_shapes"][len(self.parameters)],
                    ),
                )

        self.variable_names = "\n".join(
            [cg.statement(x, 1) for x in variable_name_list]
        )
        self.pre_trampoline_reads = "\n".join([cg.statement(x, 1) for x in ptrs])
        self.post_trampoline_writes = "\n".join([cg.statement(x, 1) for x in ptws])
        self.trampoline_args = ", ".join(targs)

        # Read return type name
        rtname = self.overload.as_function().return_type.name
        if not rtname in ["int", "float", "void"]:
            raise ValueError(f"Unsupported return type {rtname}")
        self.has_return_type = rtname != "void"

        # Indexer will just be x axis of dispatch thread for now
        indexer = "dispatchThreadID.x"

        # Build the parameters for the trampoline function (eg "float a, float b, out float result")
        self.trampoline_params = ", ".join(
            [f"{x.param_string}" for x in self.parameters.values()]
        )

        # Build the arguments to pass to the function call (eg "a, b, result")
        self.func_args = ", ".join([f"{x.name}" for x in self.parameters.values()])

        # Function call is just the function name
        self.func_call = self.function.name

        # If we have a return value, inject extra variable names, arguments and params for the result.
        if self.return_value is not None:
            self.func_call = f"_res = {self.func_call}"
            self.trampoline_params += f", out {self.return_value.param_string}"

        # Modifiers for trampoline function.
        self.trampoline_modifiers = "[Differentiable]" if self.differentiable else ""

        # Generate the shader code.
        self.shader = self._code_gen()

    def _append_forward_variable_code_for_argument(
        self,
        arg: kft.Argument,
        out_names: list[str],
        out_reads: list[str],
        out_write: list[str],
        out_targs: list[str],
        indexer: str,
    ):
        # Get access type for this variable + use to choose indexer string
        access = arg.forward_access
        if access == kft.ArgumentAccessType.read:
            idx = arg.get_variable_index_string_for_read(indexer)
        else:
            idx = arg.get_variable_index_string_for_write(indexer)

        shape = arg.buffer_shape
        if shape is not None and len(shape) > 0:
            idx = f"[{{{idx[1:-1]}}}]"

        call_data = cg.attribute("call_data", f"{arg.name}{idx}")

        # Append variable code based on access type
        if access == kft.ArgumentAccessType.read:
            # Read-only variable adds the variable in read mode + reads from the call data
            out_names.append(cg.declare(arg.input_def_string_for_read, arg.name))
            out_reads.append(cg.declarevar(arg.name, call_data))
            out_targs.append(arg.name)
        elif access == kft.ArgumentAccessType.write:
            # Write-only variable adds the variable in write mode + writes to the call data
            # Note: whilst it doesn't read input, it does need to declare the output variable
            out_names.append(cg.declare(arg.input_def_string_for_write, arg.name))
            out_reads.append(
                cg.declare(arg.translation_type.param_def_string, arg.name)
            )
            out_write.append(cg.assign(call_data, arg.name))
            out_targs.append(arg.name)
        elif access == kft.ArgumentAccessType.readwrite:
            # Read-write variable adds the variable in write mode + reads and writes to the call data
            out_names.append(cg.declare(arg.input_def_string_for_write, arg.name))
            out_reads.append(cg.declarevar(arg.name, call_data))
            out_write.append(cg.assign(call_data, arg.name))
            out_targs.append(arg.name)

    def _append_forward_variable_code_for_return_value(
        self,
        arg: kft.ReturnValue,
        out_names: list[str],
        out_reads: list[str],
        out_write: list[str],
        out_targs: list[str],
        indexer: str,
    ):
        # Return value is effectively a write-only variable called _res
        idx = arg.get_variable_index_string_for_write(indexer)
        shape = arg.buffer_shape
        if shape is not None and len(shape) > 0:
            idx = f"[{{{idx[1:-1]}}}]"
        call_data = cg.attribute("call_data", f"{arg.name}{idx}")
        out_names.append(cg.declare(arg.input_def_string_for_write, arg.name))
        out_reads.append(
            cg.declare(arg.translation_type.param_def_string, arg.name)
        )  # still need uninitialized value to write to
        out_write.append(cg.assign(call_data, arg.name))
        out_targs.append(arg.name)

    def _append_backward_variable_code_for_argument(
        self,
        arg: kft.Argument,
        out_names: list[str],
        out_reads: list[str],
        out_write: list[str],
        out_targs: list[str],
        indexer: str,
    ):
        # Get access type for both primal and derivatives for variable + calculate both indexers
        (primal_access, derivative_access) = arg.backward_access
        readable_idx = arg.get_variable_index_string_for_read(indexer)
        writable_idx = arg.get_variable_index_string_for_write(indexer)

        # Generate variable names for primals + record index mode
        if primal_access == kft.ArgumentAccessType.read:
            out_names.append(cg.declare(arg.input_def_string_for_read, arg.name))
            primal_idx = readable_idx
        elif primal_access == kft.ArgumentAccessType.write:
            out_names.append(cg.declare(arg.input_def_string_for_write, arg.name))
            primal_idx = writable_idx
        elif primal_access == kft.ArgumentAccessType.readwrite:
            out_names.append(cg.declare(arg.input_def_string_for_write, arg.name))
            primal_idx = writable_idx
        else:
            primal_idx = ""

        # Generate variable names for derivatives + record index mode
        if derivative_access == kft.ArgumentAccessType.read:
            out_names.append(
                cg.declare(arg.inputgrad_def_string_for_read, arg.name + "_grad")
            )
            derivative_idx = readable_idx
        elif derivative_access == kft.ArgumentAccessType.write:
            out_names.append(
                cg.declare(arg.inputgrad_def_string_for_write, arg.name + "_grad")
            )
            derivative_idx = writable_idx
        elif derivative_access == kft.ArgumentAccessType.readwrite:
            out_names.append(
                cg.declare(arg.inputgrad_def_string_for_write, arg.name + "_grad")
            )
            derivative_idx = writable_idx
        else:
            derivative_idx = ""

        # Build primal + grad call data strings
        primal_call_data = cg.attribute("call_data", f"{arg.name}{primal_idx}")
        derivative_call_data = cg.attribute(
            "call_data", f"{arg.name}_grad{derivative_idx}"
        )

        # Now generate the read/write operations for the various combinations of primal/derivative access
        # These are a bit more complex than above, as some combinations are direct reads/writes and others
        # require differential pairs
        if (
            primal_access == kft.ArgumentAccessType.none
            and derivative_access == kft.ArgumentAccessType.none
        ):
            # expected scenario for non-differentiable OUT parameters
            pass
        elif (
            primal_access == kft.ArgumentAccessType.read
            and derivative_access == kft.ArgumentAccessType.none
        ):
            # expected scenario for non-differentiable IN or INOUT parameters
            name = arg.name + "_primal"
            out_reads.append(cg.declarevar(name, primal_call_data))
            out_targs.append(name)
        elif (
            primal_access == kft.ArgumentAccessType.none
            and derivative_access == kft.ArgumentAccessType.read
        ):
            # expected scenario for differentiable OUT parameters
            name = arg.name + "_grad"
            out_reads.append(cg.declarevar(name, derivative_call_data))
            out_targs.append(name)
        elif (
            primal_access == kft.ArgumentAccessType.read
            and derivative_access == kft.ArgumentAccessType.write
        ):
            # expected scenario for differentiable IN parameters
            name = arg.name + "_pair"
            out_reads.append(cg.declarevar(name, cg.diff_pair(primal_call_data)))
            out_write.append(cg.assign(derivative_call_data, cg.attribute(name, "d")))
            out_targs.append(name)
        elif (
            primal_access == kft.ArgumentAccessType.read
            and derivative_access == kft.ArgumentAccessType.readwrite
        ):
            # expected scenario for differentiable INOUT parameters
            name = arg.name + "_pair"
            out_reads.append(
                cg.declarevar(
                    name, cg.diff_pair(primal_call_data, derivative_call_data)
                )
            )
            out_write.append(cg.assign(derivative_call_data, cg.attribute(name, "d")))
            out_targs.append(name)

    def _append_backward_variable_code_for_return_value(
        self,
        arg: kft.ReturnValue,
        out_names: list[str],
        out_reads: list[str],
        out_write: list[str],
        out_targs: list[str],
        indexer: str,
    ):
        # Return values are only used if differentiable, and always just read the grad
        if arg.is_differentiable:
            derivative_idx = arg.get_variable_index_string_for_read(indexer)
            name = arg.name + "_grad"
            out_names.append(cg.declare(arg.inputgrad_def_string_for_read, name))
            out_reads.append(
                cg.declarevar(
                    name, cg.attribute("call_data", f"_res_grad{derivative_idx}")
                )
            )
            out_targs.append(name)

    def _code_gen(self):
        # Build function call (defaults to just the name).
        if not self.backwards:
            trampoline_call = "_trampoline"
        else:
            trampoline_call = "bwd_diff(_trampoline)"

        if len(self.strides) > 0:
            self.variable_names += f"\n    int[{len(self.strides)}] _call_stride;"
            self.variable_names += f"\n    int[{len(self.strides)}] _call_dim;"
            load_call_id = f"    int[{len(self.strides)}] call_id;\n"
            load_call_id += "".join(
                [
                    f"    call_id[{i}] = (dispatchThreadID.x/call_data._call_stride[{i}]) % call_data._call_dim[{i}];\n"
                    for i in range(len(self.strides))
                ]
            )
            self.pre_trampoline_reads = load_call_id + self.pre_trampoline_reads

        # Build the shader string.
        shader = f"""
import "{self.function.module.name}";

{TYPES}

struct CallData {{
    uint3 _thread_count;
{self.variable_names}
}};
ParameterBlock<CallData> call_data;

{self.trampoline_modifiers}
void _trampoline({self.trampoline_params}) {{
    {self.func_call}({self.func_args});
}}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID) {{
    if (any(dispatchThreadID >= call_data._thread_count))
        return;
{self.pre_trampoline_reads}
    {trampoline_call}({self.trampoline_args});
{self.post_trampoline_writes}
}}
"""

        # Write the shader to a file for debugging.
        os.makedirs(".temp", exist_ok=True)
        with open(
            f".temp/{self.function.module.name}_{self.function.name}{'_backwards' if self.backwards else ''}.slang",
            "w",
        ) as f:
            f.write(shader)

        return shader

    def _write_call_data_for_argument(
        self,
        argument: kft.BaseFuncValue,
        out_dict: dict[str, Any],
        layouts: dict[str, sgl.TypeLayoutReflection],
    ) -> None:
        if not self.backwards:
            primal_access = argument.forward_access
            derivative_access = kft.ArgumentAccessType.none
        else:
            primal_access, derivative_access = argument.backward_access

        if primal_access == kft.ArgumentAccessType.read:
            out_dict[argument.name] = self._get_primal_value_for_input(
                argument, False, layouts
            )
        elif (
            primal_access == kft.ArgumentAccessType.write
            or primal_access == kft.ArgumentAccessType.readwrite
        ):
            out_dict[argument.name] = self._get_primal_value_for_input(
                argument, True, layouts
            )

        if derivative_access == kft.ArgumentAccessType.read:
            out_dict[argument.name + "_grad"] = self._get_derivative_value_for_input(
                argument, False, layouts
            )
        elif (
            derivative_access == kft.ArgumentAccessType.write
            or derivative_access == kft.ArgumentAccessType.readwrite
        ):
            out_dict[argument.name + "_grad"] = self._get_derivative_value_for_input(
                argument, True, layouts
            )

    def _read_call_data_for_argument(
        self,
        argument: kft.BaseFuncValue,
        out_dict: dict[str, Any],
        layouts: dict[str, sgl.TypeLayoutReflection],
    ) -> None:
        if not self.backwards:
            primal_access = argument.forward_access
            derivative_access = kft.ArgumentAccessType.none
        else:
            primal_access, derivative_access = argument.backward_access

        if primal_access == kft.ArgumentAccessType.read:
            pass
        elif (
            primal_access == kft.ArgumentAccessType.write
            or primal_access == kft.ArgumentAccessType.readwrite
        ):
            self._set_primal_value_from_output(
                argument, True, layouts, out_dict[argument.name]
            )

        if derivative_access == kft.ArgumentAccessType.read:
            pass
        elif (
            derivative_access == kft.ArgumentAccessType.write
            or derivative_access == kft.ArgumentAccessType.readwrite
        ):
            self._set_derivative_value_from_output(
                argument, True, layouts, out_dict[argument.name + "_grad"]
            )

    def _get_primal_value_for_input(
        self,
        argument: kft.BaseFuncValue,
        writable: bool,
        layouts: dict[str, sgl.TypeLayoutReflection],
    ) -> Any:
        if isinstance(argument.value, StructuredBuffer):
            return {
                "buffer": argument.value.buffer,
                "strides": list(argument.value.strides),
            }
        elif isinstance(argument.value, ScalarDiffPair):
            if writable:
                return self._create_buffer_for_scalar_argument(argument.name, layouts)
            else:
                return argument.value.primal
        elif isinstance(argument.value, ScalarRef):
            if writable:
                return self._create_buffer_for_scalar_argument(argument.name, layouts)
            else:
                return argument.value.value
        else:
            if writable:
                if isinstance(argument, kft.ReturnValue):
                    return self._create_buffer_for_scalar_argument(
                        argument.name, layouts
                    )
                else:
                    raise ValueError(
                        "Scalar value types can not be used for out arguments"
                    )
            else:
                return argument.value

    def _get_derivative_value_for_input(
        self,
        argument: kft.BaseFuncValue,
        writable: bool,
        layouts: dict[str, sgl.TypeLayoutReflection],
    ) -> Any:
        if isinstance(argument.value, StructuredBuffer):
            return argument.value.grad_buffer
        elif isinstance(argument.value, ScalarDiffPair):
            if writable:
                return self._create_buffer_for_scalar_argument(
                    argument.name + "_grad", layouts
                )
            else:
                return argument.value.grad
        else:
            raise ValueError("No derivative value found")

    def _set_primal_value_from_output(
        self,
        argument: kft.BaseFuncValue,
        writable: bool,
        layouts: dict[str, sgl.TypeLayoutReflection],
        output: Any,
    ) -> Any:
        if isinstance(argument.value, StructuredBuffer):
            assert isinstance(
                output, dict
            )  # No need to do anything if this was a buffer
        elif isinstance(argument.value, ScalarDiffPair):
            if writable:
                argument.value.primal = self._read_buffer_for_scalar_argument(
                    argument.name, layouts, output
                )
        elif isinstance(argument.value, ScalarRef):
            if writable:
                argument.value.value = self._read_buffer_for_scalar_argument(
                    argument.name, layouts, output
                )
        else:
            if writable:
                argument.value = self._read_buffer_for_scalar_argument(
                    argument.name, layouts, output
                )

    def _set_derivative_value_from_output(
        self,
        argument: kft.BaseFuncValue,
        writable: bool,
        layouts: dict[str, sgl.TypeLayoutReflection],
        output: Any,
    ) -> Any:
        if isinstance(argument.value, StructuredBuffer):
            assert isinstance(
                output, sgl.Buffer
            )  # No need to do anything if this was a buffer
        elif isinstance(argument.value, ScalarDiffPair):
            if writable:
                argument.value.grad = self._read_buffer_for_scalar_argument(
                    argument.name + "_grad", layouts, output
                )
        else:
            raise ValueError("No derivative value found")

    def _create_buffer_for_scalar_argument(
        self, arg_name: str, layouts: dict[str, sgl.TypeLayoutReflection]
    ) -> sgl.Buffer:
        layout_name = layouts[arg_name].element_type_layout.name
        if layout_name in ["int", "float"]:
            return self.function.module.session.device.create_buffer(
                element_count=1,
                struct_type=layouts[arg_name],
                usage=sgl.ResourceUsage.unordered_access,
                debug_name=arg_name,
            )
        else:
            raise ValueError(f"Unsupported scalar type {layout_name}")

    def _read_buffer_for_scalar_argument(
        self,
        arg_name: str,
        layouts: dict[str, sgl.TypeLayoutReflection],
        buffer: sgl.Buffer,
    ) -> Any:
        layout_name = layouts[arg_name].element_type_layout.name
        if layout_name == "int":
            return int(buffer.to_numpy().view(np.int32)[0])
        elif layout_name == "float":
            return float(buffer.to_numpy().view(np.float32)[0])
        else:
            raise ValueError(f"Unsupported scalar type {layout_name}")

    def call(self):
        # Get session and device for later use.
        session = self.function.module.session
        device = session.device

        # Build new module and link it with the one that contains the function being called.
        module: sgl.SlangModule = session.load_module_from_source(
            hashlib.sha256(self.shader.encode()).hexdigest()[0:16], self.shader
        )
        ep = module.entry_point("main")
        program = session.link_program([module, self.function.module], [ep])
        kernel = device.create_compute_kernel(program)

        # Find the cell_data structure via the module's global constant buffer, so we
        # can extract the fields and concrete layouts for them.
        cbuffer_type_layout = module.layout.globals_type_layout.unwrap_array()
        cbuffer_fields = [x for x in cbuffer_type_layout.fields]
        element_type_layout = cbuffer_type_layout.element_type_layout
        if (
            element_type_layout is not None
            and element_type_layout.kind == sgl.TypeReflection.Kind.struct
        ):
            cbuffer_type_layout = element_type_layout
        assert len(cbuffer_type_layout.fields) == 1
        call_data_variable_layout = cbuffer_type_layout.fields[0]
        assert call_data_variable_layout.name == "call_data"
        call_data_type_layout = (
            call_data_variable_layout.type_layout.element_type_layout
        )
        field_layouts = {
            field.name: field.type_layout for field in call_data_type_layout.fields
        }

        # Generate data to be passed to the shader as globals.
        call_data = {}
        for name, value in self.parameters.items():
            self._write_call_data_for_argument(value, call_data, field_layouts)
        if self.return_value is not None:
            self._write_call_data_for_argument(
                self.return_value, call_data, field_layouts
            )
        if len(self.strides) > 0:
            call_data["_call_stride"] = self.strides
            call_data["_call_dim"] = self.shape["call_shape"]
        call_data["_thread_count"] = sgl.uint3(self.total_threads, 1, 1)

        # Dispatch the kernel.
        kernel.dispatch(sgl.uint3(self.total_threads, 1, 1), {"call_data": call_data})

        # Read back and return the result, or just return None.
        for name, value in self.parameters.items():
            self._read_call_data_for_argument(value, call_data, field_layouts)
        if self.return_value is not None:
            self._read_call_data_for_argument(
                self.return_value, call_data, field_layouts
            )

        return self.return_value.value if self.return_value is not None else None
