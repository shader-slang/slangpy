from typing import Any, Callable, Optional

from kernelfunctions.core import SlangFunction

from kernelfunctions.backend import SlangModule, DeclReflection
from kernelfunctions.shapes import TConcreteShape


class FunctionChainBase:
    def __init__(self, parent: Optional["FunctionChainBase"]) -> None:
        super().__init__()
        self.parent = parent

    def call(self, *args: Any, **kwargs: Any) -> Any:
        calldata = self._build_call_data(False, *args, **kwargs)
        return calldata.call(*args, **kwargs)

    def backwards(self, *args: Any, **kwargs: Any) -> Any:
        calldata = self._build_call_data(True, *args, **kwargs)
        return calldata.call(*args, **kwargs)

    def set(self, *args: Any, **kwargs: Any):
        return FunctionChainSet(self, *args, **kwargs)

    def transform_input(self, transforms: dict[str, TConcreteShape]):
        return FunctionChainInputTransform(self, transforms)

    def transform_output(self, transforms: dict[str, TConcreteShape]):
        return FunctionChainOutputTransform(self, transforms)

    def debug_build_call_data(self, backwards: bool, *args: Any, **kwargs: Any):
        return self._build_call_data(backwards, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.call(*args, **kwargs)

    def _build_call_data(self, backwards: bool, *args: Any, **kwargs: Any):
        from .calldata import CallData

        chain = []
        current = self
        while current is not None:
            chain.append(current)
            current = current.parent
        chain.reverse()
        return CallData(chain, backwards, *args, **kwargs)


class FunctionChainSet(FunctionChainBase):
    def __init__(self, parent: FunctionChainBase, *args: Any, **kwargs: Any) -> None:
        super().__init__(parent)
        self.props: Optional[dict[str, Any]] = None
        # type: ignore (not decided on arguments yet)
        self.callback: Optional[Callable] = None

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError(
                "Set accepts either positional or keyword arguments, not both"
            )
        if len(args) > 1:
            raise ValueError(
                "Set accepts only one positional argument (a dictionary or callback)"
            )

        if len(kwargs) > 0:
            self.props = kwargs
        elif len(args) > 0:
            if callable(args[0]):
                self.callback = args[0]
            elif isinstance(args[0], dict):
                self.props = args[0]
            else:
                raise ValueError(
                    "Set requires a dictionary or callback as a single positional argument"
                )
        else:
            raise ValueError("Set requires at least one argument")


class FunctionChainInputTransform(FunctionChainBase):
    def __init__(
        self, parent: FunctionChainBase, transforms: dict[str, TConcreteShape]
    ) -> None:
        super().__init__(parent)
        self.transforms = transforms


class FunctionChainOutputTransform(FunctionChainBase):
    def __init__(
        self, parent: FunctionChainBase, transforms: dict[str, TConcreteShape]
    ) -> None:
        super().__init__(parent)
        self.transforms = transforms


# A callable kernel function. This assumes the function is in the root
# of the module, however a parent in the abstract syntax tree can be provided
# to search for the function in a specific scope.
class Function(FunctionChainBase):
    def __init__(
        self,
        module: SlangModule,
        name: str,
        ast_parent: Optional[DeclReflection] = None,
    ) -> None:
        super().__init__(None)
        self.module = module
        self.name = name
        if ast_parent is None:
            ast_parent = module.module_decl
        ast_functions = ast_parent.find_children_of_kind(
            DeclReflection.Kind.func, name
        )
        if len(ast_functions) == 0:
            raise ValueError(f"Function '{name}' not found in module {module.name}")
        self.overloads = [SlangFunction(x.as_function()) for x in ast_functions]
