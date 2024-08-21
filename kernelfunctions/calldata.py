import os
from typing import Any, Optional, cast
import numpy as np
import numpy.typing as npt
import sgl
import hashlib

from kernelfunctions.buffer import StructuredBuffer
from kernelfunctions.function import Function, FunctionChainBase, FunctionChainSet
import kernelfunctions.translation as kft


def match_function_overload_to_python_args(
    overload: sgl.FunctionReflection, deep_check: bool, *args: Any, **kwargs: Any
) -> Optional[dict[str, kft.Argument]]:
    overload_parameters = overload.parameters

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


def generate_call_data(argument: kft.Argument):
    if isinstance(argument.value, dict):
        res = {}
        for name, value in argument.value.items():
            res[name] = generate_call_data(value)
        return res
    elif isinstance(argument.value, StructuredBuffer):
        return argument.value.buffer
    else:
        return argument.value


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
    def __init__(self, chain: list["FunctionChainBase"]) -> None:
        super().__init__()
        if not isinstance(chain[0], Function):
            raise ValueError("First entry in chain should be a function")
        self.function = chain[0]
        self.chain = chain
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

        self.sets = sets
        self.parameters: Optional[dict[str, kft.Argument]] = None

    def call(self, *args: Any, **kwargs: Any):

        # Find an overload that matches the arguments, and pull out the mapping of slang variables to python variables.
        for ast_function in self.function.ast_functions:
            parameters = match_function_overload_to_python_args(
                ast_function.as_function(), False, *args, **kwargs
            )
            if parameters:
                self.overload = ast_function
                self.parameters = parameters
                break
        if self.parameters is None:
            raise ValueError("No matching overload found")

        # Get session and device for later use.
        session = self.function.module.session
        device = session.device

        # Calculate broadcast dimensions using numpy rules with parameter shapes
        dim_sizes = calculate_broadcast_dimensions(
            [x.python_shape for x in self.parameters.values()]
        )

        # Calc total threads required
        total_threads = 1
        for dim in dim_sizes:
            total_threads *= dim

        # Build variable names list for call data struct.
        variable_names = "".join(
            [f"   {x.input_def_string} {x.name};\n" for x in self.parameters.values()]
        )

        # Indexer will just be x axis of dispatch thread for now
        indexer = "dispatchThreadID.x"

        # Build function arguments list.
        trampoline_params = ", ".join(
            [f"{x.param_string}" for x in self.parameters.values()]
        )
        trampoline_args = ", ".join(
            [
                f"call_data.{x.get_variable_access_string(indexer)}"
                for x in self.parameters.values()
            ]
        )
        func_args = ", ".join([f"{x.name}" for x in self.parameters.values()])

        # Build function call (defaults to just the name).
        trampoline_call = "_trampoline"
        func_call = self.function.name

        # If the function has a return type, we need to create a buffer to store the result
        # and modify the function call to store the result in the buffer.
        # Note: Currently only supporting only int and float.
        rtname = self.overload.as_function().return_type.name
        if not rtname in ["int", "float", "void"]:
            raise ValueError(f"Unsupported return type {rtname}")
        if rtname != "void":
            variable_names = f"{variable_names}   RWStructuredBuffer<{self.overload.as_function().return_type.name}> _res;"
            func_call = f"_res = {func_call}"
            trampoline_args += f", call_data._res[0]"
            trampoline_params += (
                f", out {self.overload.as_function().return_type.name} _res"
            )

        # Build the shader string.
        shader = f"""
import "{self.function.module.name}";

struct CallData {{
{variable_names}
}};
CallData call_data;

static const uint3 TOTAL_THREADS = uint3({total_threads}, 1, 1);

[Differentiable]
void _trampoline({trampoline_params}) {{
    {func_call}({func_args});
}}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID) {{
    if (any(dispatchThreadID >= TOTAL_THREADS))
        return;
    {trampoline_call}({trampoline_args});
}}
"""

        # Write the shader to a file for debugging.
        os.makedirs(".temp", exist_ok=True)
        with open(
            f".temp/{self.function.module.name}_{self.function.name}.slang", "w"
        ) as f:
            f.write(shader)

        # Build new module and link it with the one that contains the function being called.
        module: sgl.SlangModule = session.load_module_from_source(
            hashlib.sha256(shader.encode()).hexdigest()[0:16], shader
        )
        ep = module.entry_point("main")
        program = session.link_program([module, self.function.module], [ep])
        kernel = device.create_compute_kernel(program)

        # Find the cell_data structure via the module's global constant buffer, so we
        # can extract the fields and concrete layouts for them.
        cbuffer_type_layout = module.layout.globals_type_layout.unwrap_array()
        element_type_layout = cbuffer_type_layout.element_type_layout
        if (
            element_type_layout is not None
            and element_type_layout.kind == sgl.TypeReflection.Kind.struct
        ):
            cbuffer_type_layout = element_type_layout
        assert len(cbuffer_type_layout.fields) == 1
        call_data_variable_layout = cbuffer_type_layout.fields[0]
        assert call_data_variable_layout.name == "call_data"
        call_data_type_layout = call_data_variable_layout.type_layout
        field_layouts = {field.name: field for field in call_data_type_layout.fields}

        # Generate data to be passed to the shader as globals.
        call_data = {}
        for name, value in self.parameters.items():
            call_data[name] = generate_call_data(value)

        # Create a buffer to store the result if necessary.
        res_field = field_layouts.get("_res")
        if res_field is not None:
            call_data["_res"] = device.create_buffer(
                element_count=1,
                struct_type=res_field.type_layout,
                usage=sgl.ResourceUsage.unordered_access,
                debug_name="_res",
            )

        # Dispatch the kernel.
        kernel.dispatch(sgl.uint3(total_threads, 1, 1), {"call_data": call_data})

        # Read back and return the result, or just return None.
        if res_field is not None:
            res_buffer: sgl.Buffer = call_data["_res"]
            if rtname == "int":
                return cast(npt.NDArray[np.int32], res_buffer.to_numpy()).astype(int)[0]
            elif rtname == "float":
                return cast(npt.NDArray[np.float32], res_buffer.to_numpy()).astype(
                    float
                )[0]
        else:
            return None
