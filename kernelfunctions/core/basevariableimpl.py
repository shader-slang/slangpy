from .basevariable import BaseVariable


class BaseVariableImpl(BaseVariable):
    def __init__(self):
        super().__init__()

    @property
    def writable(self):
        return self.primal.is_writable

    def _recurse_str(self, depth: int) -> str:
        if self.fields is not None:
            child_strs = [
                f"{'  ' * depth}{name}: {child._recurse_str(depth + 1)}" for name, child in self.fields.items()]
            return "\n" + "\n".join(child_strs)
        else:
            return f"{self.primal.slang_type.full_name}"
