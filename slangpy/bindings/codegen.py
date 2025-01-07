# SPDX-License-Identifier: Apache-2.0

from typing import Optional


def diff_pair(primal: str, derivative: str = "0"):
    return f"diffPair({primal}, {derivative})"


def declare(type_name: str, variable_name: str):
    return f"{type_name} {variable_name}"


def assign(target: str, value: str):
    return f"{target} = {value}"


def declarevar(variable_name: str, value: str):
    return f"var {variable_name} = {value}"


def attribute(object_name: str, attribute_name: str):
    return f"{object_name}.{attribute_name}"


def statement(statement: str, indent: int = 0):
    return "    " * indent + statement + ";"


class CodeGenBlock:
    def __init__(self, gen: "CodeGen"):
        super().__init__()
        self.gen = gen
        self.code: list[str] = []
        self.indent = ""

    def add_import(self, import_name: str):
        self.gen.add_import(import_name)

    def inc_indent(self):
        self.indent += "    "

    def dec_indent(self):
        self.indent = self.indent[:-4]

    def append_indent(self):
        self.append_code(self.indent)

    def append_code(self, code: str):
        self.code.append(code)

    def append_code_indented(self, code: str):
        lines = code.splitlines()
        for line in lines:
            self.append_line(line)

    def empty_line(self):
        self.append_code("\n")

    def append_line(self, func_line: str):
        self.append_indent()
        self.append_code(func_line)
        self.append_code("\n")

    def append_statement(self, func_line: str):
        self.append_indent()
        self.append_code(func_line)
        self.append_code(";\n")

    def begin_block(self):
        self.append_line("{")
        self.inc_indent()

    def end_block(self):
        self.dec_indent()
        self.append_line("}")

    def begin_struct(self, struct_name: str):
        self.append_line(f"struct {struct_name}")
        self.begin_block()

    def end_struct(self):
        self.end_block()

    def type_alias(self, alias_name: str, type_name: Optional[str]):
        if type_name is not None:
            return self.append_statement(f"typealias {alias_name} = {type_name}")

    def diff_pair(self, primal: str, derivative: str = "0"):
        return self.append_statement(diff_pair(primal, derivative))

    def declare(self, type_name: str, variable_name: str):
        return self.append_statement(declare(type_name, variable_name))

    def assign(self, target: str, value: str):
        return self.append_statement(assign(target, value))

    def declarevar(self, variable_name: str, value: str):
        return self.append_statement(declarevar(variable_name, value))

    def statement(self, statement: str):
        return self.append_statement(statement)

    def add_snippet(self, name: str, code: str):
        return self.gen.add_snippet(name, code)

    def finish(self):
        return "".join(self.code)


class CodeGen:
    """
    Tool for generating the code for a SlangPy kernel. Contains a set of
    different code blocks that can be filled in and then combined to
    generate the final code.
    """

    def __init__(self):
        super().__init__()

        #: Structs that contain code for loading/storing call data.
        self.call_data_structs = CodeGenBlock(self)

        #: The main call data uniforms struct.
        self.call_data = CodeGenBlock(self)
        self.call_data.append_line("struct CallData")
        self.call_data.begin_block()

        # legacy
        self.input_load_store = CodeGenBlock(self)

        #: File header
        self.header = ""

        #: Main kernel code
        self.kernel = CodeGenBlock(self)

        #: Imports list
        self.imports: set[str] = set()

        #: Trampoline function
        self.trampoline = CodeGenBlock(self)

        #: Context struct
        self.context = CodeGenBlock(self)

        #: Link time constants
        self.constants = CodeGenBlock(self)

        #: Code snippets.
        self.snippets: dict[str, str] = {}

    def add_snippet(self, name: str, code: str):
        """
        Add an arbitrary snippet of code to the kernel, typically
        used for adding utility functions or other code that doesn't
        fit into the other code blocks. Use 'name' parameter to
        deduplicate.
        """
        if not name in self.snippets:
            self.snippets[name] = code

    def add_import(self, import_name: str):
        """
        Add an import to the kernel.
        """
        self.imports.add(import_name)

    def finish(self,
               header: bool = False,
               call_data: bool = False,
               input_load_store: bool = False,
               kernel: bool = False,
               imports: bool = False,
               trampoline: bool = False,
               context: bool = False,
               snippets: bool = False,
               call_data_structs: bool = False,
               constants: bool = False):
        """
        Generate the final code for the kernel.
        """

        self.call_data.end_block()
        self.call_data.append_statement("ParameterBlock<CallData> call_data")

        all_code: list[str] = []
        if header:
            all_code = [self.header] + all_code
            all_code.append("\n")
        if imports:
            all_code = all_code + [f'import "{x}";\n' for x in self.imports]
            all_code.append("\n")
        if constants:
            all_code = all_code + self.constants.code
            all_code.append("\n")
        if context:
            all_code = all_code + self.context.code
            all_code.append("\n")
        if call_data_structs:
            all_code = all_code + self.call_data_structs.code
            all_code.append("\n")
        if call_data:
            all_code = all_code + self.call_data.code
            all_code.append("\n")
        if snippets:
            all_code = all_code + list(self.snippets.values())
            all_code.append("\n")
        if input_load_store:
            all_code = all_code + self.input_load_store.code
            all_code.append("\n")
        if trampoline:
            all_code = all_code + self.trampoline.code
            all_code.append("\n")
        if kernel:
            all_code = all_code + self.kernel.code
            all_code.append("\n")

        return "".join(all_code)
