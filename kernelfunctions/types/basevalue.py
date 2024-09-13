class BaseValue:
    def __init__(self):
        super().__init__()
        self.param_index = -1

    def is_compatible(self, other: 'BaseValue') -> bool:
        return True
        raise NotImplementedError()
