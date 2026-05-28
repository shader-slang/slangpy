# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import re
import importlib
import json
from inspect import isbuiltin, isclass, ismodule
from pathlib import Path

DIR = Path(__file__).parent
INDENT = "    "

API_ORDER = json.load(open(DIR / "api_order.json"))


def parse_signature(signature: str):
    # print(f"old signature: {signature}")

    name, rest = signature.split("(", maxsplit=1)
    args, return_type = rest.split(") -> ")

    slash_arg = False
    if len(args) > 3 and args[-1] == "/":
        slash_arg = True
        args = args[:-3]

    new_signature = f"{name}("

    # split items by "arg: ""
    items = re.split(r"([a-zA-Z\_0-9]+): ", args)

    # handle first argument which is either "" or "self"
    first_arg = True
    if len(items) > 0:
        if "self" in items[0]:
            new_signature += "self"
            first_arg = False
        items.pop(0)

    args = []

    for i in range(len(items) // 2):
        arg_name = items[i * 2].strip()
        arg_type = items[i * 2 + 1].strip()
        arg_default = None
        if "=" in arg_type:
            arg_type, arg_default = [s.strip() for s in arg_type.rsplit("=", maxsplit=1)]
            if arg_default[-1] == ",":
                arg_default = arg_default[:-1]

        args.append((arg_name, arg_type, arg_default))

        if not first_arg:
            new_signature += ", "
        new_signature += arg_name
        if arg_default:
            new_signature += f"={arg_default}"
        first_arg = False

    if slash_arg:
        new_signature += ", /"

    new_signature += f") -> {return_type}"

    # print(args)
    # print(f"new signature: {new_signature}")

    return signature
    return new_signature


def split_signature_doc(doc: str | None):
    lines = doc.split("\n") if doc else []
    signatures: list[str] = []
    docs: list[str] = []
    signature_index = 0
    state = 0
    for line in lines:
        # Read signatures
        if state == 0:
            if line == "":
                state = 1
            else:
                signatures.append(line)
                docs.append("")
        # Parse docs
        elif state == 1:
            if line == "Overloaded function.":
                continue
            if any(s in line for s in signatures):
                for i in range(len(signatures)):
                    if signatures[i] in line:
                        signature_index = i
                        break
                continue
            docs[signature_index] += line + "\n"

    signatures = [parse_signature(signature) for signature in signatures]
    docs = [doc.strip() for doc in docs]
    return list(zip(signatures, docs))


class Context:
    def __init__(self):
        super().__init__()
        self.level = 0
        self.prefix = ""
        self.stack = []
        self.enable_write = True
        self.output = ""
        self.entries = {"*": ""}
        self.current_entry = "*"
        self.visited_modules = set()

    def write(self, text: str):
        lines = text.split("\n")
        for line in lines:
            self.output += f"{INDENT * self.level}{line}\n"
            self.entries[self.current_entry] += f"{INDENT * self.level}{line}\n"

    def push(self, name: str, indent: bool = True):
        self.stack.append((self.level, self.prefix))
        if indent:
            self.level += 1
        self.prefix = name if self.prefix == "" else f"{self.prefix}.{name}"

    def pop(self):
        self.level, self.prefix = self.stack.pop()

    def new_entry(self, name: str):
        self.current_entry = f"{self.prefix}.{name}"
        if self.current_entry in self.entries:
            raise RuntimeError("duplicate entry")
        self.entries[self.current_entry] = ""


def process_method(obj: object, name: str, ctx: Context):
    first = True
    for signature, doc in split_signature_doc(obj.__doc__):
        ctx.write(f".. py:method:: {signature}")
        if not first:
            ctx.write(f"{INDENT}:no-index:")
        ctx.write("")
        ctx.push(name)
        if doc:
            ctx.write(doc + "\n")
        ctx.pop()
        first = False


def process_static_method(obj: object, name: str, ctx: Context):
    first = True
    for signature, doc in split_signature_doc(obj.__doc__):
        ctx.write(f".. py:staticmethod:: {signature}")
        if not first:
            ctx.write(f"{INDENT}:no-index:")
        ctx.write("")
        ctx.push(name)
        if doc:
            ctx.write(doc + "\n")
        ctx.pop()
        first = False


def process_property(obj: object, name: str, ctx: Context):
    # TODO(docs) this fails on properties not defined in nanobind
    try:
        read_write = obj.fset != None  # type: ignore
        doc = obj.fget.__doc__.split("\n")  # type: ignore
        type = doc[0].split("->")[1].strip()
        doc = "\n".join(doc[1:]).strip()
    except:
        print(f"Error processing property {name}")
        return
    ctx.write(f".. py:property:: {name}")
    ctx.write(f"{INDENT}:type: {type}\n")
    ctx.push(name)
    if doc != "":
        ctx.write(doc + "\n")
    ctx.pop()


def process_attribute(obj: object, name: str, ctx: Context):
    ctx.write(f".. py:attribute:: {ctx.prefix}.{name}")
    ctx.write(f"{INDENT}:type: {type(obj).__name__}")
    value = f'"{obj}"' if isinstance(obj, str) else str(obj)
    ctx.write(f"{INDENT}:value: {value}")
    ctx.write("")


def process_class(obj: object, name: str, ctx: Context):
    ctx.write(f".. py:class:: {ctx.prefix}.{name}")

    is_alias = f"{ctx.prefix}.{name}" != f"{obj.__module__}.{obj.__qualname__}"
    is_enum = hasattr(obj, "@entries")

    # Check if this is a type alias
    if is_alias:
        # print(f"CLASS ALIAS: {ctx.prefix}.{name}")
        ctx.push(name)
        # ctx.write(f":noindex:")
        ctx.write(f":canonical: {obj.__module__}.{obj.__qualname__}\n")
        ctx.write(f"Alias class: :py:class:`{obj.__module__}.{obj.__qualname__}`\n")
        ctx.pop()
        return

    ctx.write("")
    ctx.push(name)
    base = obj.__base__  # type: ignore
    if base.__name__ != "object":
        ctx.write(f"Base class: :py:class:`{base.__module__}.{base.__name__}`\n")

    if isinstance(obj.__doc__, str):
        ctx.write(obj.__doc__ + "\n")

    for cn in obj.__dict__:
        # Skip properties
        if re.match(r"__[a-zA-Z\_0-9]+__", cn) and cn != "__init__":
            continue
        # Skip private attributes
        if re.match(r"_[a-zA-Z\_0-9]+", cn) and cn != "__init__":
            continue

        co = getattr(obj, cn)
        if "nanobind.nb_method" in str(co):
            process_method(co, cn, ctx)
        elif "nanobind.nb_func" in str(co):
            process_static_method(co, cn, ctx)
        elif isinstance(co, property):
            process_property(co, cn, ctx)
        elif isclass(co):
            # TODO(docs) skip classes not defined in nanobind
            if not is_extension(co):
                print(f"Skipping class {cn} ({type(co)})")
                continue
            process_class(co, cn, ctx)
        else:
            if is_enum or cn == "__init__":
                continue
            process_attribute(co, cn, ctx)

    # Handle enum values
    if is_enum:
        entries = getattr(obj, "@entries")
        for value, info in entries.items():
            ctx.write(f".. py:attribute:: {info[0]}")
            ctx.write(f"{INDENT}:value: {value}")
            ctx.write("")
            if info[1]:
                ctx.write(info[1] + "\n")

    ctx.pop()


def process_function(obj: object, name: str, ctx: Context):
    first = True
    for signature, doc in split_signature_doc(obj.__doc__):
        ctx.write(f".. py:function:: {ctx.prefix}.{signature}")
        if not first:
            ctx.write(f"{INDENT}:no-index:")
        ctx.write("")
        ctx.push(name)
        if doc:
            ctx.write(doc + "\n")
        ctx.pop()
        first = False


def process_data(obj: object, name: str, ctx: Context):
    ctx.write(f".. py:data:: {ctx.prefix}.{name}")
    ctx.write(f"{INDENT}:type: {type(obj).__name__}")
    value = f'"{obj}"' if isinstance(obj, str) else str(obj)
    ctx.write(f"{INDENT}:value: {value}")
    ctx.write("")


def is_extension(obj: object):
    return isbuiltin(obj) or (isclass(obj) and not hasattr(obj, "__code__"))


def process_module(obj: object, name: str, ctx: Context):
    if obj in ctx.visited_modules:
        return
    ctx.visited_modules.add(obj)

    ctx.push(name, indent=False)

    for cn in obj.__dict__:
        # Skip properties
        if re.match(r"__[a-zA-Z\_0-9]+__", cn) and cn != "__init__":
            continue
        # Skip private attributes
        if re.match(r"_[a-zA-Z\_0-9]+", cn) and cn != "__init__":
            continue

        co = getattr(obj, cn)

        if ismodule(co):
            if not co.__name__.startswith("slangpy"):
                continue

            process_module(co, cn, ctx)
        elif isclass(co):
            # TODO(docs) skip classes not defined in nanobind
            if not is_extension(co):
                print(f"Skipping class {cn} ({type(co)})")
                continue

            ctx.new_entry(cn)
            process_class(co, cn, ctx)
        # elif isfunction(co):
        #     process_function(co, cn, ctx)
        elif "nanobind.nb_func" in str(co):
            ctx.new_entry(cn)
            process_function(co, cn, ctx)
        else:
            # TODO(docs) skip classes not defined in nanobind
            if not co.__class__ == int and not co.__class__ == str and not is_extension(co):
                print(f"Skipping data {cn} ({type(co)})")
                continue

            ctx.new_entry(cn)
            process_data(co, cn, ctx)

    ctx.pop()


def generate_api():
    ctx = Context()
    module = importlib.import_module("slangpy")
    process_module(module, "slangpy", ctx)
    # print(ctx.output)

    out = ""
    entries = ctx.entries
    added_entries = set()
    visited_api_order = set()
    for section_name, patterns in API_ORDER.items():
        out += f"{section_name}\n"
        out += "-" * len(section_name) + "\n\n"

        for pattern in patterns:
            for entry in entries:
                if entry in added_entries:
                    continue
                if re.fullmatch(pattern, entry):
                    out += entries[entry] + "\n"
                    out += "\n----\n\n"
                    added_entries.add(entry)
                    visited_api_order.add((section_name, pattern))

    out += "Miscellaneous\n"
    out += "-------------\n\n"
    for entry in entries:
        if entry in added_entries:
            continue
        out += entries[entry] + "\n"
        out += "\n----\n\n"
        print(f"Unassigned entry {entry} with content:")
        print(entries[entry])

    for section_name, patterns in API_ORDER.items():
        for pattern in patterns:
            if (section_name, pattern) not in visited_api_order:
                print(f"Unvisited api order pattern {section_name} / {pattern}")

    # Write file if it changed.
    api_path = Path(__file__).parent / "generated" / "api.rst"
    api_path.parent.mkdir(exist_ok=True)
    if not api_path.exists() or api_path.read_text() != out:
        api_path.write_text(out)


if __name__ == "__main__":
    # For testing, allow loading slangpy module from root
    import sys

    sys.path.append(str(DIR.parent))

    generate_api()
