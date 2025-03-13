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
    """
    This is the root class of trainable models. Instances of this class map to a slang type implementing the IModel interface.

    Creating a model is done in two phases:
    1) By calling the constructor of the model (e.g. LinearLayer, ModelChain, etc.)
       and setting its parameters (e.g. num_inputs/num_outputs for LinearLayer).
       This is generally light-weight and does not do much work yet.
    2) By calling .initialize(module), passing in the slang module containing
       the required slang types. This will perform allocation and type checking work.

    Most methods on IModel can only be called after the model is initialized, and will throw otherwise.

    To make networks more flexible and the user experience less verbose, many arguments of
    network components can be inferred automatically from the input type passed
    to the component. Such arguments are signalled to the user by the AutoSettable type hint.
    """

    def __init__(self):
        super().__init__()

        self._initialized = False
        self.parent: Optional[IModel] = None
        self._input_type: SlangType
        self._output_type: SlangType

    @property
    def input_type(self) -> SlangType:
        """Returns the SlangType expected by the forward method of this model's Slang implementation"""
        self.check_initialized()
        return self._input_type

    @property
    def output_type(self) -> SlangType:
        """Returns the SlangType returned by the forward method of this model's Slang implementation"""
        self.check_initialized()
        return self._output_type

    def components(self) -> list[IModel]:
        """
        Returns a list of all components in this model.

        This is equivalent to following .children() recursively.
        """
        self.check_initialized()
        return list(self._component_iter())

    def parameters(self) -> list[Tensor]:
        """Returns a list of all trainable parameters in this model."""
        self.check_initialized()
        result = []
        for c in self._component_iter():
            result += c.model_params()
        return result

    def initialize(self, module: Module, input_type: Optional[TypeLike] = None):
        """
        Initializes the model, performs parameter allocation and does type checking.

        Module should be a loaded Slang module that contains all types used by this model.
        Usually, this requires at least an `import NeuralModules;` in the slang file.

        input_type is the type of the input that will be passed to the model and is used
        to perform type checking and resolving Auto parameters. This can be either a string
        (which will be looked up in the module with reflection), a SlangType or slangpy.Struct,
        or an instance of RealArray.

        If the model can resolve its input_type by itself (e.g. if no parameters are set to Auto),
        input_type may be omitted.
        """
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
            self.model_error(f"Looking up method forward() in type {short_type_name} failed. Make sure the type "
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
                self.model_error(f"Found multiple matching overloads for method forward({input_type.full_name}) in type {short_type_name}, "
                                 f"and the return type is ambiguous (found {candidates}). Make sure there is only one forward() "
                                 f"implementation for each input type.{full_type_msg}")
            elif len(candidates) == 0:
                self.model_error(f"Could not find a matching overload for method forward({input_type.full_name}) in type {short_type_name}. "
                                 "The most common cause is that the output of the previous model is not compatible "
                                 f"with the input expected by the next model, e.g. due to mismatched dimensions "
                                 f"or element precision{full_type_msg}")
            else:
                self._output_type = self._lookup_mandatory_type(module, candidates[0])
        else:
            specialized = forward.specialize_with_arg_types([input_type])
            if specialized is None:
                self.model_error(f"Could not find a matching overload for method forward({input_type.full_name}) in type {short_type_name}. "
                                 "The most common cause is that the output of the previous model is not compatible "
                                 f"with the input expected by the next model, e.g. due to mismatched dimensions "
                                 f"or element precision{full_type_msg}")
            if specialized.return_type is None:
                self.model_error(f"The method forward({input_type.full_name}) in type {short_type_name} does not return a value. "
                                 f"Make sure the model conforms to the IModel interface{full_type_msg}")

            self._output_type = specialized.return_type

    @property
    def type_name(self) -> str:
        """Returns the name of the Slang type implementing this model."""
        self.model_error("type_name is not implemented")

    def model_init(self, module: Module, input_type: SlangType):
        """Internally called by IModel during .initialize()."""
        pass

    def model_params(self) -> list[Tensor]:
        """Returns a list of parameters used by this model (not any of its children)"""
        return []

    def children(self) -> list[IModel]:
        """Returns a list of immediate child models, if any"""
        return []

    def child_name(self, child: IModel) -> Optional[str]:
        """
        Returns a human-readable name of an immediate child of this model.

        This is called inside .model_error() to produce a useful string pointing
        to where in the model the error occurred.
        """
        return None

    def get_this(self) -> dict[str, Any]:
        """Returns a SlangPy-compatible object to be passed to Slang during a function call"""
        return {'_type': self.type_name}

    def resolve_input_type(self, module: Module) -> Optional[TypeLike]:
        """
        Called by initialize() if no input_type is provided.

        This may be optionally implemented by a model if it can resolve its input_type
        by itself, e.g. if it is not generic or if none of its parameters set to Auto.
        """
        return None

    def set_parent(self, parent: IModel):
        """Establish the parent of this model. Used during .model_error() to provide useful debugging info."""
        self.parent = parent

    def check_initialized(self):
        """May be called at the beginning of a method that is only valid if the model is initialized. Throws if not."""
        if not self._initialized:
            raise self.model_error("Model is uninitialized. Make sure to "
                                   "call .initialize() before using the model")

    def model_error(self, msg: str):
        """Throws a ModelError exception with the given message, and some extra info to help debug the issue."""
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
