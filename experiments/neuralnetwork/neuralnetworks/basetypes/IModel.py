# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from .TypeLike import TypeLike

from slangpy import Module
from slangpy.reflection import SlangType
from slangpy.types import Tensor
from typing import Optional, Any


class ModelError(Exception):
    pass


# Root interface representing a slang type that implements the IModel interface
class IModel:
    def __init__(self):
        super().__init__()

        self._initialized = False
        self.parent: Optional[IModel] = None

    def check_initialized(self):
        if not self._initialized:
            raise self.model_error("Model is uninitialized. Make sure to "
                                   "call .initialize() before using the model")

    def model_error(self, msg: str):
        segments: list[str] = []
        child = self
        while child:
            child_name = type(child).__name__
            if child.parent:
                readable_name = child.parent.child_name(child)
                if readable_name is not None:
                    child_name = f"{readable_name}: {child_name}"
            segments = [child_name] + segments
            child = child.parent

        component_name = type(self).__name__
        component_path = ' -> '.join(segments)
        raise ModelError("Encountered an error while handling model component "
                         f"{component_name} (with path {component_path}): {msg}")

    def child_name(self, child: IModel) -> Optional[str]:
        return None

    def set_parent(self, parent: IModel):
        self.parent = parent

    @property
    def type_name(self) -> str:
        raise NotImplementedError()

    # Returns a dictionary containing the data for the slang struct
    def get_this(self) -> dict[str, Any]:
        raise NotImplementedError()

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        pass

    def validate(self, module: Module):
        if self.input_type is None:
            self.model_error("input_type is None after initialization")
        if self.output_type is None:
            self.model_error("output_type is None after initialization")

        self.lookup_mandatory_type(module, self.type_name)

    def lookup_mandatory_type(self, module: Module, name: str) -> SlangType:
        lookup = module.layout.find_type_by_name(name)

        if lookup is None:
            self.model_error("Looking up slang type failed. This might be because of a missing import, or "
                             "because of a type error. Try pasting the type name into the slang "
                             f"file and check the compilation errors to help diagnose: {name}")

        return lookup

    # Returns a list of this module and all child modules
    def modules(self) -> list[IModel]:
        return [self]

    def parameters(self) -> list[Tensor]:
        return []
