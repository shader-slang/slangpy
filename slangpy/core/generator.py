# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any

from slangpy.bindings.codegen import CodeGen, CodeGenBlock
from slangpy.core.native import AccessType, CallMode

if TYPE_CHECKING:
    from slangpy.core.function import FunctionBuildInfo
    from slangpy.bindings.boundvariable import BoundVariable, BoundCall
    from slangpy.bindings.marshall import BindContext

#: Type names longer than this threshold get a ``typealias _t_{name}`` alias
#: to keep the generated ``CallData`` struct readable. Shorter names are
#: inlined directly.
MAX_INLINE_TYPE_LEN = 60


class KernelGenException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _is_slangpy_vector(type: Any) -> bool:
    return (
        hasattr(type, "element_type")
        and hasattr(type, "shape")
        and len(type.shape) == 1
        and type.shape[0] <= 4
    )


def generate_constants(build_info: "FunctionBuildInfo", cg: CodeGen) -> None:
    if build_info.constants is not None:
        for k, v in build_info.constants.items():
            if isinstance(v, bool):
                cg.constants.append_statement(
                    f"export static const bool {k} = {'true' if v else 'false'}"
                )
            elif isinstance(v, (int, float)):
                cg.constants.append_statement(f"export static const {type(v).__name__} {k} = {v}")
            elif _is_slangpy_vector(v):
                # Cheeky logic to take, eg, {0,0,0} -> float3(0,0,0)
                tn = type(v).__name__
                txt = f"{tn}({str(v)[1:-1]})"
                cg.constants.append_statement(f"export static const {tn} {k} = {txt}")
            else:
                raise KernelGenException(
                    f"Constant value '{k}' must be an int, float or bool, not {type(v).__name__}"
                )


def gen_calldata_type_name(
    binding: "BoundVariable", cgb: CodeGenBlock, type_name: str
) -> None:
    """Record the Slang type name for this variable's CallData field.

    If the type name exceeds ``MAX_INLINE_TYPE_LEN``, a
    ``typealias _t_{name}`` is emitted and the alias is stored.
    Otherwise the raw type name is stored directly.

    :param binding: The bound variable to update.
    :param cgb: The code-gen block to write the type alias to (if needed).
    :param type_name: The resolved Slang type name.
    """
    if len(type_name) > MAX_INLINE_TYPE_LEN:
        alias = f"_t_{binding.variable_name}"
        cgb.type_alias(alias, type_name)
        binding.calldata_type_name = alias
    else:
        binding.calldata_type_name = type_name


def gen_call_data_code(
    binding: "BoundVariable", cg: CodeGen, context: "BindContext", depth: int = 0
) -> None:
    """Emit Slang call-data struct and mapping constants for one bound variable.

    For struct/dict variables, emits a ``_t_{name}`` struct with ``__slangpy_load``
    and ``__slangpy_store`` methods. For leaf variables, delegates to the marshall's
    ``gen_calldata``. At depth 0, appends the variable's type to ``call_data``
    (or ``entry_point_params`` for the fast path).

    :param binding: The bound variable to emit code for.
    :param cg: The active CodeGen object.
    :param context: The bind context for the current call.
    :param depth: Recursion depth (0 = root, >0 = struct field).
    """
    from slangpy.bindings.boundvariable import BoundVariableException

    if binding.children is not None:
        cgb = cg.call_data_structs

        if binding.direct_bind:
            # Direct-bind: use raw type name directly
            assert binding.vector_type is not None
            gen_calldata_type_name(binding, cgb, binding.vector_type.full_name)
        else:
            struct_name = f"_t_{binding.variable_name}"
            cgb.begin_struct(struct_name)

            for field, variable in binding.children.items():
                gen_call_data_code(variable, cg, context, depth + 1)

            for var in binding.children.values():
                assert (
                    var.calldata_type_name is not None
                ), f"calldata_type_name not set for '{var.variable_name}'"
                cgb.declare(var.calldata_type_name, var.variable_name)

            assert binding.vector_type is not None
            context_decl = f"ContextND<{binding.call_dimensionality}> context"
            value_decl = f"{binding.vector_type.full_name} value"
            prefix = "[Differentiable]" if binding.access[1] != AccessType.none else ""

            if binding.access[0] in (AccessType.read, AccessType.readwrite):
                cgb.empty_line()
                cgb.append_line(
                    f"{prefix} void __slangpy_load({context_decl}, out {value_decl})"
                )
                cgb.begin_block()
                for field, var in binding.children.items():
                    gen_load = getattr(var.python, "gen_trampoline_load", None)
                    if gen_load is not None and gen_load(
                        cgb, var, var.variable_name, f"value.{field}"
                    ):
                        continue
                    cgb.append_statement(
                        f"{var.variable_name}.__slangpy_load(context.map(_m_{var.variable_name}),value.{field})"
                    )
                cgb.end_block()

            if binding.access[0] in (AccessType.write, AccessType.readwrite):
                cgb.empty_line()
                cgb.append_line(
                    f"{prefix} void __slangpy_store({context_decl}, in {value_decl})"
                )
                cgb.begin_block()
                for field, var in binding.children.items():
                    gen_store = getattr(var.python, "gen_trampoline_store", None)
                    if gen_store is not None and gen_store(
                        cgb, var, var.variable_name, f"value.{field}"
                    ):
                        continue
                    cgb.append_statement(
                        f"{var.variable_name}.__slangpy_store(context.map(_m_{var.variable_name}),value.{field})"
                    )
                cgb.end_block()

            cgb.end_struct()
            binding.calldata_type_name = struct_name

    else:
        # Generate call data
        binding.python.gen_calldata(cg.call_data_structs, context, binding)

    # Skip mapping constants for direct-bind variables (they bypass __slangpy_load/store)
    if not binding.direct_bind:
        if len(binding.vector_mapping) > 0:
            cg.call_data_structs.append_statement(
                f"static const int[] _m_{binding.variable_name} = {{ {','.join([str(x) for x in binding.vector_mapping.as_tuple()])} }}"
            )
        else:
            cg.call_data_structs.append_statement(
                f"static const int _m_{binding.variable_name} = 0"
            )

    if depth == 0:
        assert (
            binding.calldata_type_name is not None
        ), f"calldata_type_name not set for '{binding.variable_name}'"
        if binding.create_param_block:
            cg.add_parameter_block(binding.calldata_type_name, "_param_" + binding.variable_name)
        elif cg.skip_call_data:
            cg.entry_point_params.append(
                f"uniform {binding.calldata_type_name} {binding.variable_name}"
            )
        else:
            cg.call_data.declare(binding.calldata_type_name, binding.variable_name)


# ---------------------------------------------------------------------------
# generate_code sub-functions
# ---------------------------------------------------------------------------


def _validate_and_compute_group_shape(
    build_info: "FunctionBuildInfo",
    call_data_len: int,
) -> tuple[int, list[int], list[int]]:
    """Validate ``call_group_shape`` and compute the flat group size and strides.

    Returns ``(call_group_size, call_group_strides, call_group_shape_vector)``.
    When no call_group_shape is set, returns ``(1, [], [])``.
    """
    call_group_size = 1
    call_group_strides: list[int] = []
    call_group_shape_vector: list[int] = []

    call_group_shape = build_info.call_group_shape
    if call_group_shape is not None:
        call_group_shape_vector = call_group_shape.as_list()

        if len(call_group_shape_vector) > call_data_len:
            raise KernelGenException(
                f"call_group_shape dimensionality ({len(call_group_shape_vector)}) must be <= "
                f"call_shape dimensionality ({call_data_len}). "
                f"call_group_shape cannot have more dimensions than call_shape."
            )
        elif len(call_group_shape_vector) < call_data_len:
            missing_dims = call_data_len - len(call_group_shape_vector)
            call_group_shape_vector = [1] * missing_dims + call_group_shape_vector

        for i, dim in enumerate(call_group_shape_vector):
            if dim < 1:
                raise KernelGenException(
                    f"call_group_shape[{i}] = {dim} is invalid. "
                    f"All call_group_shape elements must be >= 1."
                )

        for dim in call_group_shape_vector[::-1]:
            call_group_strides.append(call_group_size)
            call_group_size *= dim
        call_group_strides.reverse()

        if call_group_size > 1024:
            raise KernelGenException(
                f"call_group_size ({call_group_size}) exceeds the typical 1024 maximum "
                f"enforced by most APIs. Consider reducing your call_group_shape dimensions."
            )

    return call_group_size, call_group_strides, call_group_shape_vector


def _emit_link_time_constants(
    cg: CodeGen,
    build_info: "FunctionBuildInfo",
    call_data_len: int,
    call_group_size: int,
    call_group_strides: list[int],
    call_group_shape_vector: list[int],
) -> None:
    """Emit link-time constant declarations.

    Emits Slang code like::

        export static const int call_data_len = 2;
        export static const int call_group_size = 1;
        export static const int[call_data_len] call_group_strides = {};
        export static const int[call_data_len] call_group_shape_vector = {};
    """
    generate_constants(build_info, cg)
    cg.constants.append_statement(f"export static const int call_data_len = {call_data_len}")
    cg.constants.append_statement(f"export static const int call_group_size = {call_group_size}")

    cg.constants.append_line(f"export static const int[call_data_len] call_group_strides = {{")
    cg.constants.inc_indent()
    if call_group_size != 1:
        for i in range(call_data_len):
            cg.constants.append_line(f"{call_group_strides[i]},")
    cg.constants.dec_indent()
    cg.constants.append_statement("}")

    cg.constants.append_line(
        f"export static const int[call_data_len] call_group_shape_vector = {{"
    )
    cg.constants.inc_indent()
    if call_group_size != 1:
        for i in range(call_data_len):
            cg.constants.append_line(f"{call_group_shape_vector[i]},")
    cg.constants.dec_indent()
    cg.constants.append_statement("}")


def _emit_shape_and_metadata_params(
    cg: CodeGen,
    call_data_len: int,
    use_entrypoint_args: bool,
) -> None:
    """Emit shape arrays and ``_thread_count``.

    Fast path (entry-point params)::

        uniform int[N] _grid_stride
        uniform int[N] _grid_dim
        uniform int[N] _call_dim
        uniform uint3 _thread_count

    Fallback (CallData struct fields)::

        int[N] _grid_stride;
        int[N] _grid_dim;
        int[N] _call_dim;
        uint3 _thread_count;
    """
    if call_data_len > 0:
        if use_entrypoint_args:
            cg.entry_point_params.append(f"uniform int[{call_data_len}] _grid_stride")
            cg.entry_point_params.append(f"uniform int[{call_data_len}] _grid_dim")
            cg.entry_point_params.append(f"uniform int[{call_data_len}] _call_dim")
        else:
            cg.call_data.append_statement(f"int[{call_data_len}] _grid_stride")
            cg.call_data.append_statement(f"int[{call_data_len}] _grid_dim")
            cg.call_data.append_statement(f"int[{call_data_len}] _call_dim")

    if use_entrypoint_args:
        cg.entry_point_params.append("uniform uint3 _thread_count")
    else:
        cg.call_data.append_statement("uint3 _thread_count")


def _emit_call_data_definitions(
    cg: CodeGen,
    context: "BindContext",
    signature: "BoundCall",
) -> None:
    """Emit per-variable call-data structs and type aliases for all signature nodes."""
    for node in signature.values():
        node.gen_call_data_code(cg, context)


def _data_name(x: "BoundVariable", use_entrypoint_args: bool) -> str:
    """Return the Slang name used to access a variable's call data in the trampoline.

    - ``_param_{name}`` for param-block variables (both paths).
    - ``__in_{name}`` in the fast (entry-point-args) path.
    - ``call_data.{name}`` in the fallback path.
    """
    if x.create_param_block:
        return f"_param_{x.variable_name}"
    elif use_entrypoint_args:
        return f"__in_{x.variable_name}"
    else:
        return f"call_data.{x.variable_name}"


def _emit_trampoline(
    cg: CodeGen,
    context: "BindContext",
    build_info: "FunctionBuildInfo",
    root_params: list["BoundVariable"],
    use_entrypoint_args: bool,
) -> None:
    """Emit the ``_trampoline`` helper function.

    Fast path signature::

        [Differentiable]
        void _trampoline(Context __slangpy_context__,
                         no_diff MyType __in_param0, ...)

    Fallback signature::

        [Differentiable]
        void _trampoline(Context __slangpy_context__)
    """
    from slangpy.bindings.boundvariable import BoundVariableException

    if context.call_mode != CallMode.prim:
        cg.trampoline.append_line("[Differentiable]")

    if use_entrypoint_args:
        trampoline_params = ["Context __slangpy_context__"]
        for x in root_params:
            if x.create_param_block:
                continue
            assert x.calldata_type_name is not None
            trampoline_params.append(f"no_diff {x.calldata_type_name} __in_{x.variable_name}")
        cg.trampoline.append_line(f"void _trampoline({', '.join(trampoline_params)})")
    else:
        cg.trampoline.append_line("void _trampoline(Context __slangpy_context__)")
    cg.trampoline.begin_block()

    # Declare local variables for each parameter
    for x in root_params:
        assert x.vector_type is not None
        cg.trampoline.declare(x.vector_type.full_name, x.variable_name)

    # Load inputs from call data
    for x in root_params:
        data_name = _data_name(x, use_entrypoint_args)
        gen_load = getattr(x.python, "gen_trampoline_load", None)
        if gen_load is not None and gen_load(cg.trampoline, x, data_name, x.variable_name):
            continue
        if x.access[0] == AccessType.read or x.access[0] == AccessType.readwrite:
            cg.trampoline.append_statement(
                f"{data_name}.__slangpy_load(__slangpy_context__.map(_m_{x.variable_name}), {x.variable_name})"
            )

    # Emit function call
    cg.trampoline.append_indent()
    if any(x.variable_name == "_result" for x in root_params):
        cg.trampoline.append_code("_result = ")

    func_name = build_info.name
    if func_name == "$init":
        results = [x for x in root_params if x.variable_name == "_result"]
        assert len(results) == 1
        assert results[0].vector_type is not None
        func_name = results[0].vector_type.full_name
    elif len(root_params) > 0 and root_params[0].variable_name == "_this":
        func_name = f"_this.{func_name}"

    normal_params = [
        x for x in root_params if x.variable_name != "_result" and x.variable_name != "_this"
    ]
    cg.trampoline.append_code(
        f"{func_name}(" + ", ".join(x.variable_name for x in normal_params) + ");\n"
    )

    # Store outputs back to call data
    for x in root_params:
        if (
            x.access[0] == AccessType.write
            or x.access[0] == AccessType.readwrite
            or x.access[1] == AccessType.read
        ):
            data_name = _data_name(x, use_entrypoint_args)
            gen_store = getattr(x.python, "gen_trampoline_store", None)
            if gen_store is not None and gen_store(cg.trampoline, x, data_name, x.variable_name):
                continue
            if not x.python.is_writable:
                raise BoundVariableException(
                    f"Cannot read back value for non-writable type", x
                )
            cg.trampoline.append_statement(
                f"{data_name}.__slangpy_store(__slangpy_context__.map(_m_{x.variable_name}), {x.variable_name})"
            )

    cg.trampoline.end_block()
    cg.trampoline.append_line("")


def _emit_entry_point_signature(
    cg: CodeGen,
    build_info: "FunctionBuildInfo",
    call_data_len: int,
    call_group_size: int,
    use_entrypoint_args: bool,
) -> None:
    """Emit the ``[shader(...)]`` attribute line and entry-point function signature.

    Compute fast path::

        [shader("compute")]
        [numthreads(32, 1, 1)]
        void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID,
                          int3 flat_call_group_id: SV_GroupID,
                          int flat_call_group_thread_id: SV_GroupIndex,
                          uniform int[N] _grid_stride, ...)

    Ray-tracing fallback::

        [shader("raygen")]
        void raygen_main()
    """
    from slangpy.core.function import PipelineType

    if build_info.pipeline_type == PipelineType.compute:
        cg.kernel.append_line('[shader("compute")]')
        if call_group_size != 1:
            cg.kernel.append_line(f"[numthreads({call_group_size}, 1, 1)]")
        else:
            cg.kernel.append_line("[numthreads(32, 1, 1)]")
        if use_entrypoint_args:
            sig_parts = ["int3 flat_call_thread_id: SV_DispatchThreadID"]
            if call_data_len > 0:
                sig_parts.append("int3 flat_call_group_id: SV_GroupID")
                sig_parts.append("int flat_call_group_thread_id: SV_GroupIndex")
            sig_parts.extend(cg.entry_point_params)
            cg.kernel.append_line(f"void compute_main({', '.join(sig_parts)})")
        else:
            cg.kernel.append_line(
                "void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID, int3 flat_call_group_id: SV_GroupID, int flat_call_group_thread_id: SV_GroupIndex)"
            )
    elif build_info.pipeline_type == PipelineType.ray_tracing:
        cg.kernel.append_line('[shader("raygen")]')
        if use_entrypoint_args:
            sig_parts = list(cg.entry_point_params)
            cg.kernel.append_line(f"void raygen_main({', '.join(sig_parts)})")
        else:
            cg.kernel.append_line("void raygen_main()")
    else:
        raise RuntimeError(f"Unknown pipeline type: {build_info.pipeline_type}")


def _emit_kernel_body(
    cg: CodeGen,
    context: "BindContext",
    build_info: "FunctionBuildInfo",
    root_params: list["BoundVariable"],
    call_data_len: int,
    use_entrypoint_args: bool,
) -> None:
    """Emit the body of the compute/raygen entry-point function.

    Emits the bounds check, ``init_thread_local_call_shape_info``, Context
    construction, and the trampoline call::

        if (any(flat_call_thread_id >= _thread_count)) return;
        if (!init_thread_local_call_shape_info(...)) return;
        Context __slangpy_context__ = {flat_call_thread_id, ...};
        _trampoline(__slangpy_context__, ...);
    """
    from slangpy.core.function import PipelineType

    if build_info.pipeline_type == PipelineType.ray_tracing:
        cg.kernel.append_statement("int3 flat_call_thread_id = DispatchRaysIndex();")

    if use_entrypoint_args:
        cg.kernel.append_statement("if (any(flat_call_thread_id >= _thread_count)) return")
    else:
        cg.kernel.append_statement(
            "if (any(flat_call_thread_id >= call_data._thread_count)) return"
        )

    context_args = "flat_call_thread_id"

    if call_data_len > 0:
        grid_prefix = "" if use_entrypoint_args else "call_data."
        if build_info.pipeline_type == PipelineType.compute:
            cg.kernel.append_line(
                f"""
    if (!init_thread_local_call_shape_info(flat_call_group_thread_id,
        flat_call_group_id, flat_call_thread_id, {grid_prefix}_grid_stride,
        {grid_prefix}_grid_dim, {grid_prefix}_call_dim))
        return;"""
            )
        elif build_info.pipeline_type == PipelineType.ray_tracing:
            cg.kernel.append_line(
                f"""
    if (!init_thread_local_call_shape_info(0,
        uint3(0), flat_call_thread_id, {grid_prefix}_grid_stride,
        {grid_prefix}_grid_dim, {grid_prefix}_call_dim))
        return;"""
            )
        context_args += ", CallShapeInfo::get_call_id().shape"

    cg.kernel.append_statement(f"Context __slangpy_context__ = {{{context_args}}}")

    fn = "_trampoline"
    if context.call_mode == CallMode.bwds:
        fn = f"bwd_diff({fn})"

    if use_entrypoint_args:
        trampoline_args = ["__slangpy_context__"]
        for x in root_params:
            if x.create_param_block:
                continue
            trampoline_args.append(x.variable_name)
        cg.kernel.append_statement(f"{fn}({', '.join(trampoline_args)})")
    else:
        cg.kernel.append_statement(f"{fn}(__slangpy_context__)")


def generate_code(
    context: "BindContext",
    build_info: "FunctionBuildInfo",
    signature: "BoundCall",
    cg: CodeGen,
) -> None:
    """Generate Slang kernel code for the given function call signature.

    Orchestrates all sub-steps: constants, shape params, call-data structs,
    trampoline, entry-point signature, and kernel body.
    """
    use_entrypoint_args = context.use_entrypoint_args
    cg.add_import("slangpy")
    call_data_len = context.call_dimensionality

    call_group_size, call_group_strides, call_group_shape_vector = (
        _validate_and_compute_group_shape(build_info, call_data_len)
    )

    cg.add_import(build_info.module.name)
    if use_entrypoint_args:
        cg.skip_call_data = True

    _emit_link_time_constants(
        cg, build_info, call_data_len, call_group_size, call_group_strides, call_group_shape_vector
    )
    _emit_shape_and_metadata_params(cg, call_data_len, use_entrypoint_args)
    _emit_call_data_definitions(cg, context, signature)

    root_params = sorted(signature.values(), key=lambda x: x.param_index)

    _emit_trampoline(cg, context, build_info, root_params, use_entrypoint_args)
    _emit_entry_point_signature(cg, build_info, call_data_len, call_group_size, use_entrypoint_args)
    cg.kernel.begin_block()
    _emit_kernel_body(cg, context, build_info, root_params, call_data_len, use_entrypoint_args)
    cg.kernel.end_block()
