# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import argparse
from typing import Any, Optional, cast
import libcst as cst
import libcst.matchers as m

# List of classes we expect to discover that can be constructed from
# a dictionary. If 'True', a corresponding TypedDict and Union type
# will be generated. If 'False', it will be ignored. If a type
# is discovered but not in this list it will be treated as an error.
DESCRIPTOR_CONVERT_TYPES = {
    "AccelerationStructureBuildDesc": True,
    "AccelerationStructureBuildInputInstances": True,
    "AccelerationStructureBuildInputMotionOptions": True,
    "AccelerationStructureBuildInputProceduralPrimitives": True,
    "AccelerationStructureBuildInputTriangles": True,
    "AccelerationStructureDesc": True,
    "AccelerationStructureInstanceDesc": True,
    "AccelerationStructureQueryDesc": True,
    "AppDesc": True,
    "AppWindowDesc": True,
    "AspectBlendDesc": True,
    "BufferDesc": True,
    "BufferOffsetPair": True,
    "ColorTargetDesc": True,
    "ComputePipelineDesc": True,
    "DepthStencilDesc": True,
    "DepthStencilOpDesc": True,
    "DeviceDesc": True,
    "DrawArguments": True,
    "FenceDesc": True,
    "HitGroupDesc": True,
    "InputElementDesc": True,
    "InputLayoutDesc": True,
    "MultisampleDesc": True,
    "QueryPoolDesc": True,
    "RasterizerDesc": True,
    "RayTracingPipelineDesc": True,
    "RenderPassColorAttachment": True,
    "RenderPassDepthStencilAttachment": True,
    "RenderPassDesc": True,
    "RenderPipelineDesc": True,
    "RenderState": True,
    "SamplerDesc": True,
    "ScissorRect": True,
    "ShaderTableDesc": True,
    "SlangCompilerOptions": True,
    "SlangLinkOptions": True,
    "SlangSessionDesc": True,
    "SubresourceRange": True,
    "SurfaceConfig": True,
    "SurfaceConfig": True,
    "TextureDesc": True,
    "TextureLoader.Options": True,
    "TextureViewDesc": True,
    "VertexStreamDesc": True,
    "Viewport": True,
}

ADDITIONAL_DESCRIPTOR_CONVERTIONS = {"BufferOffsetPair": ("Buffer",)}

QUIET = False


# Helper to print verbose output.
def print_verbose(*args: Any):
    if not QUIET:
        print(*args)


# Field from a descriptor class that needs to be represented in the
# corresponding dictionary type.
class FCDFieldInfo:
    def __init__(self, name: cst.Name, annotation: cst.BaseExpression):
        super().__init__()
        self.name = name
        self.annotation = annotation


# A class discovered by FindConvertableToDictionaryTypes that
# needs to have a corresponding dictionary type.
class FCDStackInfo:
    def __init__(
        self,
        parent_classs_type: cst.ClassDef | None,
        class_type: cst.ClassDef,
        full_name: str,
    ):
        super().__init__()
        self.parent_class_type = parent_classs_type
        self.class_type = class_type
        self.valid = False
        self.fields: list[FCDFieldInfo] = []
        self.full_name = full_name


# This visitor will find all classes that have a constructor that
# takes a single dictionary and extracts its fields by
# looking for setter functions.
class FindConvertableToDictionaryTypes(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.stack: list[FCDStackInfo] = []
        self.results: list[FCDStackInfo] = []
        self.read_class = False
        self.name_stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        parent = self.stack[-1].class_type if len(self.stack) > 0 else None
        self.name_stack.append(node.name.value)
        self.stack.append(FCDStackInfo(parent, node, ".".join(self.name_stack)))

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.stack.pop()
        self.name_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        if node.name.value == "__init__":
            # If this is an init function, use to check if this is a class
            # we're interested in. Function shuld have 2 positional parameters:
            func_params = node.params
            if (
                len(func_params.params) != 0
                or len(func_params.kwonly_params) != 0
                or len(func_params.posonly_params) != 2
            ):
                return False
            if func_params.posonly_params[0].name.value != "self":
                return False
            if func_params.posonly_params[1].name.value != "arg":
                return False

            # 2nd parameter should be a dictionary.
            if not self._annotation_name_equals(func_params.posonly_params[1].annotation, "dict"):
                return False

            # Store the type
            # print(self.stack[-1].class_type.name.value)
            self.stack[-1].valid = True
            self.results.append(self.stack[-1])
        elif len(self.stack) > 0 and self.stack[-1].valid:
            # Not an init function but we're in a valid descriptor class, so
            # so we want to record any setter types.
            if self._is_setter(node):
                self.stack[-1].fields.append(
                    FCDFieldInfo(
                        node.name, node.params.posonly_params[1].annotation.annotation  # type: ignore (bad libcst typing info)
                    )
                )

    def _annotation_name_equals(self, annotation: Optional[cst.Annotation], name: str):
        if annotation is not None:
            return m.matches(annotation, m.Annotation(annotation=m.Name(name)))
        return False

    def _is_setter(self, node: cst.FunctionDef):
        for decorator in node.decorators:
            if m.matches(decorator, m.Decorator(decorator=m.Attribute(attr=m.Name("setter")))):
                return True
        return False


# This transformer will insert the TypedDict and Union types
# for a given list of descriptor classes.
class InsertTypesTransformer(cst.CSTTransformer):
    def __init__(self, discovered_descriptor_types: list[FCDStackInfo]):
        super().__init__()
        self.discovered_descriptor_types = discovered_descriptor_types

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # On leaving a module, insert the dictionary and union types for any global descriptor classes.
        new_body: list[cst.CSTNode] = list(updated_node.body)
        changed = self._insert_descriptor_nodes(None, new_body)
        if changed:
            return updated_node.with_changes(body=new_body)
        return updated_node

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        # On leaving a class, insert the dictionary and union types for any child descriptor classes.
        new_body: list[cst.CSTNode] = list(updated_node.body.body)
        changed = self._insert_descriptor_nodes(original_node, new_body)
        if changed:
            class_body = updated_node.body
            class_body = class_body.with_changes(body=new_body)
            return updated_node.with_changes(body=class_body)

        return updated_node

    def _insert_nodes_after_class_by_name(
        self,
        body_nodes: list[cst.CSTNode],
        class_name: str,
        nodes_to_insert: list[cst.CSTNode],
    ):
        # Find the class definition in the body_nodes list and inserts nodes after it.
        insert_idx = 0
        while insert_idx < len(body_nodes):
            bn = body_nodes[insert_idx]
            if m.matches(bn, m.ClassDef(name=m.Name(class_name))):
                break
            insert_idx += 1
        insert_idx += 1
        for x in nodes_to_insert:
            body_nodes.insert(insert_idx, x)
            insert_idx += 1

    def _insert_descriptor_nodes(
        self, original_parent: cst.ClassDef | None, updated_body: list[cst.CSTNode]
    ) -> bool:
        # Finds any descriptor classes that are children of the specified parent (or None for module)
        # and inserts the dictionary and union types after the corresponding class definition.
        changed = False
        for desc_type in self.discovered_descriptor_types:
            if desc_type.parent_class_type == original_parent:
                dict_type, union_type = self._build_types_for_descriptor(desc_type)
                self._insert_nodes_after_class_by_name(
                    updated_body,
                    desc_type.class_type.name.value,
                    [cst.Newline(), dict_type, cst.Newline(), union_type],
                )
                changed = True
        return changed

    def _build_types_for_descriptor(self, result: FCDStackInfo):
        # Generates the dictionary and union types for a descriptor class.

        # Generate the elements for the dictionary that defines the
        # new TypedDict. This is really just an element with name:annotation
        # for each entry, but looks more complex due to insertion of correct
        # whitespace (each element other than the last adds a newline and
        # 4 spaces after the comma).
        dict_elements: list[cst.DictElement] = []
        for idx in range(len(result.fields)):
            field = result.fields[idx]
            key = cst.SimpleString(f'"{field.name.value}"')
            value = field.annotation
            if idx < len(result.fields) - 1:
                dict_elements.append(
                    cst.DictElement(
                        key=key,
                        value=value,
                        comma=cst.Comma(
                            whitespace_after=cst.ParenthesizedWhitespace(
                                last_line=cst.SimpleWhitespace("    "), indent=True
                            )
                        ),
                    )
                )
            else:
                dict_elements.append(
                    cst.DictElement(
                        key=key,
                        value=value,
                    )
                )

        # Generate the dictionary definition. Formatting adds a new line
        # and 4 spaces after the opening brace and a new line before the closing
        # brace.
        dict_def = cst.Dict(
            elements=dict_elements,
            lbrace=cst.LeftCurlyBrace(
                whitespace_after=cst.ParenthesizedWhitespace(
                    last_line=cst.SimpleWhitespace("    "), indent=True
                )
            ),
            rbrace=cst.RightCurlyBrace(whitespace_before=cst.ParenthesizedWhitespace(indent=True)),
        )

        # Wrap the dictionary in a call to 'TypedDict' to create the new type.
        typed_dict_annotation = cst.Call(
            func=cst.Name("TypedDict"),
            args=[
                cst.Arg(
                    value=cst.SimpleString(f'"{result.class_type.name.value}Dict"'),
                ),
                cst.Arg(value=dict_def),
                cst.Arg(keyword=cst.Name("total"), value=cst.Name("False")),
            ],
        )

        # Store the new TypedDict in a type alias.
        type_alias = cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[
                        cst.AssignTarget(target=cst.Name(f"{result.class_type.name.value}Dict"))
                    ],
                    value=typed_dict_annotation,
                )
            ]
        )

        # Check for additional types to union with the descriptor
        # dictionary. This is used for types like BufferOffsetPair which
        # are a union of Buffer and BufferOffsetPair.
        if result.class_type.name.value in ADDITIONAL_DESCRIPTOR_CONVERTIONS:
            additional = [
                cst.SubscriptElement(slice=cst.Index(value=cst.Name(x)))
                for x in ADDITIONAL_DESCRIPTOR_CONVERTIONS[result.class_type.name.value]
            ]
        else:
            additional = []

        # Also generate a corresponding union that combines the source type with the new TypedDict.
        union_type_alias = cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[
                        cst.AssignTarget(target=cst.Name(f"{result.class_type.name.value}Param"))
                    ],
                    value=cst.Subscript(
                        value=cst.Name("Union"),
                        slice=[
                            cst.SubscriptElement(
                                slice=cst.Index(value=cst.Name(result.class_type.name.value))
                            ),
                            cst.SubscriptElement(
                                slice=cst.Index(
                                    value=cst.Name(f"{result.class_type.name.value}Dict")
                                )
                            ),
                        ]
                        + additional,
                    ),
                )
            ]
        )

        return type_alias, union_type_alias


# Helper transformer base class that tracks whether we're within a parameter annotation
# or a TypedDict call. This is used to determine whether we should be updating type names.
class BaseParamAnnotationAndTypeDictTransformer(cst.CSTTransformer):
    def __init__(self):
        super().__init__()
        self.in_param = 0
        self.in_annotation = 0
        self.in_typed_dict_call = 0

    def is_in_scope(self):
        return (self.in_param == 1 and self.in_annotation == 1) or (self.in_typed_dict_call == 1)

    def visit_Param(self, node: cst.Param) -> Optional[bool]:
        self.in_param += 1

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        self.in_param -= 1
        return updated_node

    def visit_Annotation(self, node: cst.Annotation) -> Optional[bool]:
        self.in_annotation += 1

    def leave_Annotation(
        self, original_node: cst.Annotation, updated_node: cst.Annotation
    ) -> cst.Annotation:
        self.in_annotation -= 1
        return updated_node

    def visit_Call(self, node: cst.Call) -> Optional[bool]:
        if m.matches(
            node,
            m.Call(
                func=m.Name("TypedDict"),
            ),
        ):
            self.in_typed_dict_call += 1

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if m.matches(
            original_node,
            m.Call(
                func=m.Name("TypedDict"),
            ),
        ):
            self.in_typed_dict_call -= 1
        return updated_node


# Helper to build fully qualified attribute name from an attribute tree.
def build_attribute_name(node: cst.Attribute):
    if isinstance(node.value, cst.Attribute):
        return f"{build_attribute_name(node.value)}.{node.attr.value}"
    elif isinstance(node.value, cst.Name):
        return node.value.value + "." + node.attr.value
    else:
        raise Exception("Unexpected node type in _build_attribute_name")


# Helper to build an attribute tree from a fully qualified name.
def build_attribute_tree(full_name: str):
    parts = full_name.split(".")
    parts.reverse()
    return _build_attribute_tree_recurse(parts, 0)


# Internal recursive helper for build_attribute_tree.
def _build_attribute_tree_recurse(parts: list[str], idx: int):
    if idx == len(parts) - 1:
        return cst.Name(parts[idx])
    return cst.Attribute(
        value=_build_attribute_tree_recurse(parts, idx + 1),
        attr=cst.Name(parts[idx]),
    )


# This transformer will replace parameter annotations or typed dict entries
# with a new type that appends 'extension'.
class ReplaceTypesTransformer(BaseParamAnnotationAndTypeDictTransformer):
    def __init__(self, types_by_full_name: set[str], extension: str):
        super().__init__()
        self.types_by_full_name = types_by_full_name
        self.extension = extension
        self.replacements = 0

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        # If within a parameter annotation or TypedDict call, update type names.
        if self.is_in_scope():
            if updated_node.value in self.types_by_full_name:
                self.replacements += 1
                return updated_node.with_changes(value=f"{updated_node.value}{self.extension}")
        return updated_node

    def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
        return False  # don't step into attributes - we just want to treat as fully qualified names.

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute:
        # If within a parameter annotation or TypedDict call, update type names. Attribute version
        # has to be a bit smarter than leave_Name as it needs to reconstruct the full name,
        # match it, and on a match construct a new attribute tree to replace it.
        if self.is_in_scope():
            full_name = build_attribute_name(updated_node)
            if full_name in self.types_by_full_name:
                self.replacements += 1
                return cast(cst.Attribute, build_attribute_tree(f"{full_name}{self.extension}"))

        return updated_node


# Creates a set of unions for vector types like uint3 with their implicit sequence type,
# e.g. Union[uint3,Sequence[int]] and relaces references to the original type with the union.
class ExtendVectorTypeArgs(BaseParamAnnotationAndTypeDictTransformer):
    def __init__(self):
        super().__init__()
        self.CONVERSIONS = [
            ("bool1", "bool"),
            ("bool2", "bool"),
            ("bool3", "bool"),
            ("bool4", "bool"),
            ("float1", "float"),
            ("float16_t", "float"),
            ("float16_t1", "float16_t"),
            ("float16_t2", "float16_t"),
            ("float16_t3", "float16_t"),
            ("float16_t4", "float16_t"),
            ("float2", "float"),
            ("float2x2", "float"),
            ("float2x3", "float"),
            ("float2x4", "float"),
            ("float3", "float"),
            ("float3x2", "float"),
            ("float3x3", "float"),
            ("float3x4", "float"),
            ("float4", "float"),
            ("float4x2", "float"),
            ("float4x3", "float"),
            ("float4x4", "float"),
            ("int1", "int"),
            ("int2", "int"),
            ("int3", "int"),
            ("int4", "int"),
            ("quatf", "float"),
            ("uint1", "int"),
            ("uint2", "int"),
            ("uint3", "int"),
            ("uint4", "int"),
        ]
        self.full_names = set([f"math.{x[0]}" for x in self.CONVERSIONS])
        self.new_types = 0
        self.replacements = 0

    def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
        return False  # don't step into attributes - we just want to treat as fully qualified names

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute:
        # If within a parameter annotation or TypedDict call, update type names. Attribute version
        # has to be a bit smarter than leave_Name as it needs to reconstruct the full name,
        # match it, and on a match construct a new attribute tree to replace it.
        if self.is_in_scope():
            full_name = build_attribute_name(updated_node)
            if full_name in self.full_names:
                self.replacements += 1
                return cast(cst.Attribute, cst.Name(f"{updated_node.attr.value}param"))

        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        new_union_types: list[cst.CSTNode] = []
        for conversion in self.CONVERSIONS:
            type_name = conversion[0]
            if type_name == "float16_t":
                type_union = self._union_with_name(
                    build_attribute_tree(f"math.{type_name}"), conversion[1]
                )
            else:
                type_union = self._union_with_sequence(
                    build_attribute_tree(f"math.{type_name}"), conversion[1]
                )
            new_union_types.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[cst.AssignTarget(target=cst.Name(type_name + "param"))],
                            value=type_union,
                        )
                    ]
                )
            )

        # Store number added
        self.new_types = len(new_union_types)

        # To keep things neat, find the existing typing import.
        insert_idx = 0
        while insert_idx < len(tree.body):
            bn = tree.body[insert_idx]
            if isinstance(bn, cst.ClassDef):
                break
            insert_idx += 1

        # Now add new typing imports after it
        return updated_node.with_changes(
            body=list(updated_node.body[:insert_idx])
            + new_union_types
            + list(updated_node.body[insert_idx:])
        )

    def _union_with_name(self, node: cst.Attribute | cst.Name, name: str):
        return cst.Subscript(
            value=cst.Name("Union"),
            slice=[
                cst.SubscriptElement(slice=cst.Index(value=node)),
                cst.SubscriptElement(slice=cst.Index(value=cst.Name(name))),
            ],
        )

    def _union_with_sequence(self, node: cst.Attribute | cst.Name, sequence_type: str):
        return cst.Subscript(
            value=cst.Name("Union"),
            slice=[
                cst.SubscriptElement(slice=cst.Index(value=node)),
                cst.SubscriptElement(
                    slice=cst.Index(
                        value=cst.Subscript(
                            value=cst.Name("Sequence"),
                            slice=[
                                cst.SubscriptElement(slice=cst.Index(value=cst.Name(sequence_type)))
                            ],
                        )
                    )
                ),
            ],
        )


# Replaces the 'ArrayLike' type for 'NDArray' for to_numpy functions.
class FixNumpyArrays(cst.CSTTransformer):
    def __init__(self):
        super().__init__()
        self.in_numpy_function = 0
        self.replacements = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        if node.name.value == "to_numpy":
            self.in_numpy_function += 1

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if original_node.name.value == "to_numpy":
            self.in_numpy_function -= 1
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if self.in_numpy_function > 0 and updated_node.value == "ArrayLike":
            self.replacements += 1
            return updated_node.with_changes(value="NDArray")
        return updated_node


# Loads a parsed file.
def load_file(file_path: str) -> cst.Module:
    code = open(file_path).read()
    return cst.parse_module(code)


# Gets convertible descriptor types from tree.
def find_convertable_descriptors(tree: cst.Module) -> list[FCDStackInfo]:
    # Find all descriptor classes that can be constructed from a dictionary.
    find_convertable_types_visitor = FindConvertableToDictionaryTypes()
    tree.visit(find_convertable_types_visitor)

    # Sanity check against list of descriptors we expect to convert.
    for result in find_convertable_types_visitor.results:
        if not result.full_name in DESCRIPTOR_CONVERT_TYPES:
            raise Exception(
                f"Discovered descriptor class {result.full_name}, but not in list of types to convert. Add it to the list in postprocess_stub.py."
            )
    for conv_type in DESCRIPTOR_CONVERT_TYPES:
        if not any(
            result.full_name == conv_type for result in find_convertable_types_visitor.results
        ):
            raise Exception(
                f"Type {conv_type} is in list of types to convert but no corresponding descriptor class was discovered. Remove it from the list in postprocess_stub.py."
            )

    # Filter only results for which DESCRIPTOR_CONVERT_TYPES is true.
    convertable_types = [
        x for x in find_convertable_types_visitor.results if DESCRIPTOR_CONVERT_TYPES[x.full_name]
    ]
    print_verbose(f"  Found {len(convertable_types)} descriptor types convertable to dictionaries.")
    return convertable_types


# Inserts the TypedDict and Union types for the discovered descriptor classes.
def insert_converted_descriptors(
    tree: cst.Module, convertable_types: list[FCDStackInfo]
) -> cst.Module:
    transformer = InsertTypesTransformer(convertable_types)
    res = tree.visit(transformer)
    print_verbose(f"  Inserted descriptor dictionaries.")
    return res


# Insert typing imports
def insert_typing_imports(tree: cst.Module) -> cst.Module:

    # To keep things neat, find the existing typing import.
    insert_idx = 0
    while insert_idx < len(tree.body):
        bn = tree.body[insert_idx]
        if m.matches(bn, m.SimpleStatementLine(body=(m.ImportFrom(module=m.Name("typing")),))):
            break
        insert_idx += 1

    # Now add new typing imports after it.
    new_imports = [
        cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Name("typing"),
                    names=[
                        cst.ImportAlias(name=cst.Name("TypedDict"), asname=None),
                        cst.ImportAlias(name=cst.Name("Union"), asname=None),
                    ],
                )
            ]
        ),
        cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Attribute(value=cst.Name("numpy"), attr=cst.Name("typing")),
                    names=[
                        cst.ImportAlias(name=cst.Name("NDArray"), asname=None),
                    ],
                )
            ]
        ),
    ]

    # Construct new body
    new_body = list(tree.body[:insert_idx]) + new_imports + list(tree.body[insert_idx:])
    print_verbose(
        f"  Inserting {len(new_body)-len(tree.body)} import statements at index {insert_idx}."
    )

    return tree.with_changes(body=new_body)


# Replaces argument and dictionary references to descriptor classes with their new types.
def replace_types(tree: cst.Module, convertable_types: list[FCDStackInfo]) -> cst.Module:
    transformer = ReplaceTypesTransformer(set([x.full_name for x in convertable_types]), "Param")
    res = tree.visit(transformer)
    print_verbose(f"  Replaced {transformer.replacements} type references to descriptor classes.")
    return res


# Replaces vector types like uint3 with Union[uint3,Sequence[int]].
def extend_vector_types(tree: cst.Module) -> cst.Module:
    transformer = ExtendVectorTypeArgs()
    res = tree.visit(transformer)
    print_verbose(
        f"  Added {transformer.new_types} vector types and replaced {transformer.replacements} references to them."
    )
    return res


# Makes to_numpy functions return NDArray.
def fix_numpy_types(tree: cst.Module) -> cst.Module:
    transformer = FixNumpyArrays()
    res = tree.visit(transformer)
    print_verbose(
        f"  Replaced {transformer.replacements} references to ArrayLike with NDArray in to_numpy functions."
    )
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, action="store", help="Input file")
    parser.add_argument(
        "--out",
        type=str,
        action="store",
        help="Out filename (defaults to overwriting input)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output to console")
    parser.add_argument(
        "--submodule",
        action="store_true",
        help="Mark as a submodule with reduced post processing.",
    )
    args = vars(parser.parse_args())
    input_filename = args["file"]
    output_filename = args.get("out")
    submodule = args.get("submodule", False)
    if output_filename is None:
        output_filename = input_filename
    QUIET = args.get("quiet", False)

    # Enable these for testing.
    # input_filename = str(Path(__file__).parent / "../build/windows-vs2022/bin/Debug/python/sgl/__init__.pyi")
    # output_filename = input_filename.replace(".pyi", "_2.pyi")

    # Single message in quiet mode
    if QUIET:
        print(f"Post processing {input_filename}")

    print_verbose("Post processing python stub:")
    print_verbose(f"  Input: {input_filename}")
    print_verbose(f"  Output: {output_filename}")

    tree = load_file(input_filename)

    if not submodule:
        convertable_types = find_convertable_descriptors(tree)

        tree = insert_converted_descriptors(tree, convertable_types)

    tree = insert_typing_imports(tree)

    if not submodule:
        tree = replace_types(tree, convertable_types)

        tree = extend_vector_types(tree)

    tree = fix_numpy_types(tree)

    open(output_filename, "w").write(tree.code)
