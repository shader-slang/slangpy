import re
from typing import cast
import numpy as np
import numpy.typing as npt
import sgl
import hashlib

from .function import *
from .buffer import StructuredBuffer
from .typemappings import (
    is_valid_scalar_type_conversion,
    is_valid_vector_type_conversion,
)


def scalar_typename_2_slang_typename(scalar_type: str) -> str:
    scalar_type = scalar_type.replace("32", "")
    scalar_type = re.sub(r"(\d+)", r"$1_t", scalar_type)
    return scalar_type


class VariablePair:
    def __init__(
        self,
        slang_variable: sgl.VariableReflection,
        slang_type: sgl.TypeReflection,
        python_variable: Any,
    ) -> None:
        super().__init__()
        self.slang_variable = slang_variable
        self.python_variable = python_variable
        self.slang_type = slang_type

    def variable_type_def_string(self):
        if isinstance(self.python_variable, StructuredBuffer):
            if self.python_variable.element_type in [
                int,
                float,
                bool,
                sgl.int1,
                sgl.float1,
                sgl.bool1,
            ]:
                return f"RWStructuredBuffer<{self.slang_type.name}>"
            else:
                return f"RWStructuredBuffer<{type(self.python_variable).__name__}>"
        elif self.slang_type.kind == sgl.TypeReflection.Kind.scalar:
            return type(self.python_variable).__name__
        elif self.slang_type.kind == sgl.TypeReflection.Kind.vector:
            return type(self.python_variable).__name__
        elif self.slang_type.kind == sgl.TypeReflection.Kind.matrix:
            return type(self.python_variable).__name__

    def variable_access_string(self, indexer: str = "i"):
        if isinstance(self.python_variable, StructuredBuffer):
            return f"{self.slang_variable.name}[{indexer}]"
        else:
            return f"{self.slang_variable.name}"

    def shape(self):
        if isinstance(self.python_variable, StructuredBuffer):
            return (self.python_variable.element_count,)
        else:
            return (1,)


def _scalar_types_compatible(
    slang_type: sgl.TypeReflection, python_variable: Any
) -> bool:

    python_type = type(python_variable)
    if isinstance(python_variable, StructuredBuffer):
        python_type = python_variable.element_type

    if slang_type.kind == sgl.TypeReflection.Kind.scalar:
        return is_valid_scalar_type_conversion(slang_type.scalar_type, python_type)
    elif slang_type.kind == sgl.TypeReflection.Kind.vector:
        return is_valid_vector_type_conversion(
            slang_type.scalar_type, python_type, slang_type.col_count
        )
    else:
        return False


def _walk(slang_variable: sgl.VariableReflection, python_variable: Any) -> VariablePair:
    slang_type = slang_variable.type

    if slang_type.kind == sgl.TypeReflection.Kind.struct:
        # Slang struct - step into it and walk child fields
        newdict = {}
        for member in slang_type.fields:
            newdict[member.name] = _walk(member, python_variable[member.name])
        return VariablePair(slang_variable, slang_type, newdict)
    elif slang_type.kind == sgl.TypeReflection.Kind.array:
        newlist = []
        for i, element in enumerate(python_variable):
            newlist.append(_walk(slang_variable, element))
        return VariablePair(slang_variable, slang_type, newlist)
    elif (
        slang_type.kind == sgl.TypeReflection.Kind.scalar
        or slang_type.kind == sgl.TypeReflection.Kind.vector
    ):
        if not _scalar_types_compatible(slang_type, python_variable):
            raise ValueError(
                f"Type mismatch: {slang_type.scalar_type} and {type(python_variable)}"
            )
        return VariablePair(slang_variable, slang_type, python_variable)
    else:
        raise ValueError(f"Unsupported type {slang_type.kind}")


def _try_walk(
    slang_variable: sgl.VariableReflection, python_variable: Any
) -> Optional[VariablePair]:
    try:
        return _walk(slang_variable, python_variable)
    except ValueError as e:
        return None


class FunctionMatch:
    def __init__(
        self, function: sgl.FunctionReflection, parameters: dict[str, VariablePair]
    ) -> None:
        super().__init__()
        self.function = function
        self.parameters = parameters

    def call(self):
        pass


def match_function_overload_to_python_args(
    overload: sgl.FunctionReflection, *args: Any, **kwargs: Any
) -> Optional[dict[str, VariablePair]]:
    overload_parameters = overload.parameters

    # If there are more positional arguments than parameters, it can't match.
    if len(args) > len(overload_parameters):
        return None

    # Dictionary of slang arguments and corresponding python arguments
    handled_args: dict[str, VariablePair] = {
        x.name: cast(VariablePair, None) for x in overload_parameters
    }

    # Positional arguments must all match perfectly
    for i, arg in enumerate(args):
        pair = _try_walk(overload_parameters[i], arg)
        if not pair:
            return None
        handled_args[overload_parameters[i].name] = pair

    # Pair up kw arguments with slang arguments
    overload_parameters_dict = {x.name: x for x in overload_parameters}
    for name, arg in kwargs.items():
        param = overload_parameters_dict.get(name)
        if param is None:
            return None
        pair = _try_walk(param, arg)
        if not pair:
            return None
        handled_args[name] = pair

    # Check if all arguments have been handled
    for arg in handled_args.values():
        if arg is None:
            return None

    return handled_args


def generate_call_data(variable: VariablePair):
    if isinstance(variable.python_variable, dict):
        res = {}
        for name, value in variable.python_variable.items():
            res[name] = generate_call_data(value)
        return res
    elif isinstance(variable.python_variable, StructuredBuffer):
        return variable.python_variable.buffer
    else:
        return variable.python_variable


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
        self.parameters: Optional[dict[str, VariablePair]] = None

    def call(self, *args: Any, **kwargs: Any):

        # Find an overload that matches the arguments, and pull out the mapping of slang variables to python variables.
        for ast_function in self.function.ast_functions:
            parameters = match_function_overload_to_python_args(
                ast_function.as_function(), *args, **kwargs
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
            [x.shape() for x in self.parameters.values()]
        )

        # Calc total threads required
        total_threads = 1
        for dim in dim_sizes:
            total_threads *= dim

        # Build variable names list for call data struct.
        variable_names = "".join(
            [
                f"   {x.variable_type_def_string()} {x.slang_variable.name};\n"
                for x in self.parameters.values()
            ]
        )

        # Indexer will just be x axis of dispatch thread for now
        indexer = "dispatchThreadID.x"

        # Build function arguments list.
        func_args = ", ".join(
            [
                f"call_data.{x.variable_access_string(indexer)}"
                for x in self.parameters.values()
            ]
        )

        # Build function call (defaults to just the name).
        func_call = self.function.name

        # If the function has a return type, we need to create a buffer to store the result
        # and modify the function call to store the result in the buffer.
        # Note: Currently only supporting only int and float.
        rtname = self.overload.as_function().return_type.name
        if not rtname in ["int", "float", "void"]:
            raise ValueError(f"Unsupported return type {rtname}")
        if rtname != "void":
            variable_names = f"{variable_names}   RWStructuredBuffer<{self.overload.as_function().return_type.name}> _res;"
            func_call = f"call_data._res[0] = {func_call}"

        # Build the shader string.
        shader = f"""
import "{self.function.module.name}";

struct CallData {{
{variable_names}
}};
CallData call_data;

static const uint3 TOTAL_THREADS = uint3({total_threads}, 1, 1);

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID) {{
    if (any(dispatchThreadID >= TOTAL_THREADS))
        return;
    {func_call}({func_args});
}}
"""

        # Write the shader to a file for debugging.
        # with open(
        #     f".temp/{self.function.module.name}_{self.function.name}.slang", "w"
        # ) as f:
        #     f.write(shader)

        # Build new module and link it with the one that contains the function being called.
        module: sgl.SlangModule = session.load_module_from_source(
            hashlib.sha256(shader.encode()).hexdigest()[0:16], shader
        )
        ep = module.entry_point("main")
        program = session.link_program([module, self.function.module], [ep])
        kernel = device.create_compute_kernel(program)

        # Find the cell_data structure via the module's global constant buffer, so we
        # can extract the fields and concrete layouts for them.
        cbuffer_type_layout = module.layout.globals_type_layout
        if cbuffer_type_layout.element_type_layout:
            cbuffer_type_layout = cbuffer_type_layout.element_type_layout
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
