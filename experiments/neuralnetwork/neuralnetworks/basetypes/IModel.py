# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from slangpy import Module, Struct, Tensor
from slangpy.reflection import SlangType

from .RealArray import RealArray

from typing import Optional, Union, Any


TypeLike = Union[str, SlangType, Struct, RealArray]


class ModelError(Exception):
    pass


# Root interface representing a slang type that implements the IModel interface
class IModel:
    def __init__(self):
        super().__init__()

        self._initialized = False
        self.parent: Optional[IModel] = None
        self._input_type: SlangType
        self._output_type: SlangType

    @property
    def input_type(self) -> SlangType:
        self.check_initialized()
        return self._input_type

    @property
    def output_type(self) -> SlangType:
        self.check_initialized()
        return self._output_type

    def components(self) -> list[IModel]:
        self.check_initialized()
        return list(self._component_iter())

    def parameters(self) -> list[Tensor]:
        self.check_initialized()
        result = []
        for c in self._component_iter():
            result += c.model_params()
        return result

    def initialize(self, module: Module, input_type: Optional[TypeLike]):
        if input_type is None:
            input_type = self.resolve_input_type(module)
        if isinstance(input_type, RealArray):
            input_type = input_type.name()
        if isinstance(input_type, str):
            input_type = self._lookup_mandatory_type(module, input_type)
        if isinstance(input_type, Struct):
            input_type = input_type.struct
        if input_type is None:
            self.model_error("initialize() cannot proceed: No input_type was provided, and "
                             "the model can't resolve it by itself, either because the model "
                             "does not implement it or because some parameters are set to Auto.")

        try:
            self.model_init(module, input_type)
        except ModelError as e:
            raise
        except Exception as e:
            self.model_error(f"{type(e).__name__}: {e}")

        self._input_type = input_type
        self._initialized = True

        type_name = self.type_name
        model_type = self._lookup_mandatory_type(module, type_name)

        if len(type_name) > 50:
            short_type_name = type_name[:47] + "..."
            full_type_msg = f". The full type name was {type_name}"
        else:
            short_type_name = type_name
            full_type_msg = ""

        forward = module.layout.find_function_by_name_in_type(model_type, "forward")
        if forward is None:
            self.model_error(f"Looking up method {short_type_name}::forward() failed. Make sure the type "
                             f"implements the IModel interface{full_type_msg}")

        # The correct solution to looking up the return type of forward given the input type
        # is to always lookup forward() and specialize it. However, currently this can crash
        # in some circumstances when forward() is overloaded due to a slang reflection bug,
        # and we have to work around it by looping through the overloads and checking if a
        # matching IModel implementation exists.
        # This should go away and be replaced by the curent else: branch always
        if forward.is_overloaded:
            return_types = {
                f.return_type.full_name for f in forward.overloads if f.return_type is not None}
            candidates = []
            for candidate in return_types:
                witness_name = f"impl::returnTypeWitness<{input_type.full_name}, {candidate}, {type_name}>"
                witness = module.layout.find_function_by_name(witness_name)
                if witness is not None:
                    candidates.append(candidate)
            if len(candidates) > 1:
                self.model_error(f"Found multiple matching overloads for {short_type_name}::forward({input_type.full_name}), "
                                 f"and the return type is ambiguous (found {candidates}). Make sure there is only one forward() "
                                 f"implementation for each input type.{full_type_msg}")
            elif len(candidates) == 0:
                self.model_error(f"Could not find a matching overload for {short_type_name}::forward({input_type.full_name}). "
                                 "The most common cause is that the output of the previous model is not compatible "
                                 f"with the input expected by the next model, e.g. due to mismatched dimensions "
                                 f"or element precision{full_type_msg}")
            else:
                self._output_type = self._lookup_mandatory_type(module, candidates[0])
        else:
            specialized = forward.specialize_with_arg_types([input_type])
            if specialized is None:
                self.model_error(f"Could not find a matching overload for {short_type_name}::forward({input_type.full_name}). "
                                 "The most common cause is that the output of the previous model is not compatible "
                                 f"with the input expected by the next model, e.g. due to mismatched dimensions "
                                 f"or element precision{full_type_msg}")
            if specialized.return_type is None:
                self.model_error(f"The method {short_type_name}::forward({input_type.full_name}) does not return a value. "
                                 f"Make sure the model conforms to the IModel interface{full_type_msg}")

            self._output_type = specialized.return_type

    @property
    def type_name(self) -> str:
        raise NotImplementedError()

    def model_init(self, module: Module, input_type: Optional[Union[str, SlangType, Struct]]):
        pass

    def model_params(self) -> list[Tensor]:
        return []

    def children(self) -> list[IModel]:
        return []

    def child_name(self, child: IModel) -> Optional[str]:
        return None

    # Returns a dictionary containing the data for the slang struct
    def get_this(self) -> dict[str, Any]:
        return {'_type': self.type_name}

    def resolve_input_type(self, module: Module) -> Optional[SlangType]:
        return None

    def set_parent(self, parent: IModel):
        self.parent = parent

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

    def _lookup_mandatory_type(self, module: Module, name: str) -> SlangType:
        lookup = module.layout.find_type_by_name(name)

        if lookup is None:
            self.model_error("Looking up slang type failed. This might be because of a missing import, or "
                             "because of a type error. Try pasting the type name into the slang "
                             f"module and check for compilation errors to help diagnose: {name}")

        return lookup

    def _component_iter(self):
        yield self
        for c in self.children():
            yield from c._component_iter()
