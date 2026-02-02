# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# int _idx<I: IInteger, let N : int>(I[N] indices, uint stride[N], uint offset)
# {
#    int idx = 0;
#    [ForceUnroll]
#    for (int i = 0; i < N; i++) { idx += indices[i].toUInt() * stride[i]; }
#    return idx + offset;
# }

# public extension<T, TensorType : ITensor<T, 2>> TensorType
# {
#    [Differentiable]
#    public T load(int i0, int i1)
#    {
#        return load(int[2](i0, i1));
#    }
# }
# public extension<T, TensorType : IWTensor<T, 3>> TensorType
# {
#    [Differentiable]
#    public void store(int i0, int i1, int i2, T value)
#    {
#        store(int[3](i0, i1, i2), value);
#    }
# }
# public extension<T> Tensor<T, 2>
# {
#    public __subscript(int i0, int i1)->T
#    {
#        get { return load(i0, i1); }
#    }
# }
# public extension<T> WTensor<T, 2>
# {
#    public __subscript(int i0, int i1)->T
#    {
#        set { store(i0, i1, newValue); }
#    }
# }


from pathlib import Path


def cg_idx(dimensions: int):
    if dimensions == 0:
        return "int _idx(uint stride[0], uint offset) { return offset; }"
    idx_args = [f"int idx{i}" for i in range(dimensions)]
    code = []
    code.append(f"int _idx({', '.join(idx_args)}, uint stride[{dimensions}], uint offset)")
    code.append("{")
    code.append("    int idx = 0;")
    for i in range(dimensions):
        code.append(f"    idx += idx{i} * stride[{i}];")
    code.append("    return idx + offset;")
    code.append("}")
    code.append("")
    return "\n".join(code)


def cg_load_decl(dimensions: int, differentiable: bool = False):
    diff = "[Differentiable]\n    " if differentiable else ""
    if dimensions == 0:
        return f"""\
    {diff}public T load();
"""
    else:
        args = ", ".join([f"int i{i}" for i in range(dimensions)])
        return f"""\
    {diff}public T load({args});
"""


def cg_load(dimensions: int, differentiable: bool = False, primal: bool = False):
    if differentiable:
        if dimensions == 0:
            return f"""\
    [ForceInline]
    [Differentiable]
    [BackwardDerivative(_load_bwd_indices)]
    public T load()
    {{
        return this._read_primal_each();
    }}
    [ForceInline]
    void _load_bwd_indices(T.Differential grad)
    {{
        this._accumulate_grad_each(grad);
    }}
"""
        else:
            args = ", ".join([f"int i{i}" for i in range(dimensions)])
            idx_args = ", ".join([f"i{i}" for i in range(dimensions)])

            return f"""\
    [ForceInline]
    [Differentiable]
    [BackwardDerivative(_load_bwd_indices)]
    public T load({args})
    {{
        return this._read_primal_each({idx_args});
    }}
    [ForceInline]
    void _load_bwd_indices({args}, T.Differential grad)
    {{
        this._accumulate_grad_each(grad, {idx_args});
    }}
"""
    elif primal:
        if dimensions == 0:
            return f"""\
    [ForceInline]
    [TreatAsDifferentiable]
    public T load()
    {{
        return this._read_primal_each();
    }}
"""
        else:
            args = ", ".join([f"int i{i}" for i in range(dimensions)])
            idx_args = ", ".join([f"i{i}" for i in range(dimensions)])

            return f"""\
    [ForceInline]
    [TreatAsDifferentiable]
    public T load({args})
    {{
        return this._read_primal_each({idx_args});
    }}
"""
    else:
        if dimensions == 0:
            return f"""\
    [ForceInline]
    public T load()
    {{
        return this._read_each();
    }}
"""
        else:
            args = ", ".join([f"int i{i}" for i in range(dimensions)])
            idx_args = ", ".join([f"i{i}" for i in range(dimensions)])

            return f"""\
    [ForceInline]
    public T load({args})
    {{
        return this._read_each({idx_args});
    }}
"""


def cg_store_decl(dimensions: int, differentiable: bool = False):

    diff = "[Differentiable]\n    " if differentiable else ""
    if dimensions == 0:
        return f"""\
    {diff}public void store(T value);
"""
    else:
        args = ", ".join([f"int i{i}" for i in range(dimensions)] + ["T value"])
        return f"""\
    {diff}public void store({args});
"""


def cg_store(dimensions: int, differentiable: bool = False, primal: bool = False):
    if differentiable:
        if dimensions == 0:
            return f"""\
    [ForceInline]
    [Differentiable]
    [BackwardDerivative(_store_bwd_indices)]
    public void store(T value)
    {{
        this._write_primal_each(value);
    }}
    [ForceInline]
    void _store_bwd_indices(inout DifferentialPair<T> grad)
    {{
        grad = diffPair(grad.p, this._read_grad_each());
    }}
"""
        else:
            args = ", ".join([f"int i{i}" for i in range(dimensions)])
            idx_args = ", ".join([f"i{i}" for i in range(dimensions)])

            return f"""\
    [ForceInline]
    [Differentiable]
    [BackwardDerivative(_store_bwd_indices)]
    public void store({args}, T value)
    {{
        this._write_primal_each(value, {idx_args});
    }}
    [ForceInline]
    void _store_bwd_indices({args}, inout DifferentialPair<T> grad)
    {{
        grad = diffPair(grad.p, this._read_grad_each({idx_args}));
    }}
"""
    elif primal:
        if dimensions == 0:
            return f"""\
    [ForceInline]
    [TreatAsDifferentiable]
    public void store(T value)
    {{
        this._write_primal_each(value);
    }}
"""
        else:
            args = ", ".join([f"int i{i}" for i in range(dimensions)])
            idx_args = ", ".join([f"i{i}" for i in range(dimensions)])

            return f"""\
    [ForceInline]
    [TreatAsDifferentiable]
    public void store({args}, T value)
    {{
        this._write_primal_each(value, {idx_args});
    }}
"""
    else:
        if dimensions == 0:
            return f"""\
    [ForceInline]
    public void store(T value)
    {{
        this._write_each(value);
    }}
"""
        else:
            args = ", ".join([f"int i{i}" for i in range(dimensions)])
            idx_args = ", ".join([f"i{i}" for i in range(dimensions)])

            return f"""\
    [ForceInline]
    public void store({args}, T value)
    {{
        this._write_each(value, {idx_args});
    }}
"""


def cg_atomic_add(dimensions: int):
    if dimensions == 0:
        return f"""\
    [ForceInline]
    public void add(T value)
    {{
        this._accumulate_each(value);
    }}
"""
    else:
        args = ", ".join([f"int i{i}" for i in range(dimensions)])
        idx_args = ", ".join([f"i{i}" for i in range(dimensions)])

        return f"""\
    [ForceInline]
    public void add({args}, T value)
    {{
        this._accumulate_each(value, {idx_args});
    }}
"""


def cg_subscript_getter(dimensions: int, differentiable: bool = False):
    diff = "[Differentiable] " if differentiable else ""
    if dimensions == 0:
        return f"{diff}get {{ return this.load(); }}"
    else:
        args = ", ".join([f"i{i}" for i in range(dimensions)])
        return f"{diff}get {{ return this.load({args}); }}"


def cg_subscript_setter(dimensions: int, differentiable: bool = False):
    diff = "[Differentiable] " if differentiable else ""
    if dimensions == 0:
        return f"{diff}set {{ this.store(newValue); }}"
    else:
        args = ", ".join([f"i{i}" for i in range(dimensions)])
        return f"{diff}set {{ this.store({args}, newValue); }}"


def cg_subscript_extension(
    getter: bool, setter: bool, dimensions: int, differentiable: bool = False
):
    if not getter and not setter:
        return ""

    getter_str = (
        ("        " + cg_subscript_getter(dimensions, differentiable) + "\n") if getter else ""
    )
    setter_str = (
        ("        " + cg_subscript_setter(dimensions, differentiable) + "\n") if setter else ""
    )

    if dimensions == 0:
        return f"""\
    public __subscript()->T
    {{
{getter_str}{setter_str}    }}
"""
    else:
        args = ", ".join([f"int i{i}" for i in range(dimensions)])
        return f"""\
    public __subscript({args})->T
    {{
{getter_str}{setter_str}    }}
"""


def cg_tensor_name(
    tensor_type: str, dimensions: int, differentiable: bool = False, primal: bool = False
):
    if differentiable:
        diff = "Diff"
    elif primal:
        diff = "Primal"
    else:
        diff = ""
    return f"{tensor_type}{diff}Tensor<T, {dimensions}>"


def cg_interface_extension_header(
    tensor_type: str, dimensions: int, differentiable: bool = False, primal: bool = False
):
    tensor_name = cg_tensor_name(tensor_type, dimensions, differentiable, primal)
    diff_constraint = ": IDifferentiable" if differentiable else ""
    code = f"public extension<T{diff_constraint}, TensorType : I{tensor_name}> TensorType"
    if differentiable:
        code += " where T.Differential : IAtomicAddable"
    return code


def cg_struct_extension_header(
    tensor_type: str, dimensions: int, differentiable: bool = False, primal: bool = False
):
    tensor_name = cg_tensor_name(tensor_type, dimensions, differentiable, primal)
    diff_constraint = ": IDifferentiable" if differentiable or primal else ""
    code = f"public extension<T{diff_constraint}> {tensor_name}"
    if differentiable or primal:
        code += " where T.Differential : IAtomicAddable"
    return code


def cg_atomic_extension_header(tensor_type: str, dimensions: int):
    tensor_name = cg_tensor_name(tensor_type, dimensions, False)
    code = f"public extension<T> {tensor_name} where T : IAtomicAddable"
    return code


def generate_tensor_extensions():
    tensor_types = ["", "W", "RW"]
    dimensions = range(0, 9)
    code = []

    code.append("// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n")
    code.append("// Tensor index extensions generated by tools/generate_tensors.py\n")
    code.append("implementing slangpy;\n\n")

    differentiable = False

    # for dim in dimensions:
    #     code.append(cg_idx(dim))

    for differentiable in [False, True]:
        for tensor_type in tensor_types:
            for dim in dimensions:
                getter = False
                setter = False
                if not "W" in tensor_type or "RW" in tensor_type:
                    getter = True
                if "W" in tensor_type or "RW" in tensor_type:
                    setter = True

                # Struct extensions
                # code.append(cg_struct_extension_header(tensor_type, dim, differentiable))
                # code.append("\n{\n")
                # code.append("}\n")
                # code.append("\n")

                # Interface extensions
                code.append(cg_interface_extension_header(tensor_type, dim, differentiable))
                code.append("\n{\n")
                # if not (getter and setter):
                if getter:
                    code.append(cg_load(dim, differentiable))
                if setter:
                    code.append(cg_store(dim, differentiable))
                code.append(cg_subscript_extension(getter, setter, dim, differentiable))
                code.append("}\n")
                code.append("\n")

    if True:
        for tensor_type in tensor_types:
            for dim in dimensions:
                getter = False
                setter = False
                if not "W" in tensor_type or "RW" in tensor_type:
                    getter = True
                if "W" in tensor_type or "RW" in tensor_type:
                    setter = True

                # Struct extensions
                # code.append(cg_struct_extension_header(tensor_type, dim, primal=True))
                # code.append("\n{\n")
                # code.append(cg_subscript_extension(getter, setter, dim))
                # code.append("}\n")
                # code.append("\n")

    for dim in dimensions:
        # Struct extensions
        tensor_type = "Atomic"
        getter = True
        setter = True
        code.append(cg_atomic_extension_header(tensor_type, dim))
        code.append("\n{\n")
        code.append(cg_load(dim))
        code.append(cg_store(dim))
        code.append(cg_atomic_add(dim))
        # code.append(cg_subscript_extension(True, True, dim))
        code.append("}\n")
        code.append("\n")

    return "".join(code)


if __name__ == "__main__":
    tensor_code = generate_tensor_extensions()
    path = Path(__file__).parent.parent / "slangpy" / "slang" / "tensor_indices_generated.slang"
    with open(path, "w") as f:
        f.write(tensor_code)
