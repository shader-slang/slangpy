from kernelfunctions.callsignature import BasePythonTypeMarshal, register_python_type


class IntMarshal(BasePythonTypeMarshal):
    def __init__(self):
        super().__init__(int)


class FloatMarshal(BasePythonTypeMarshal):
    def __init__(self):
        super().__init__(float)


register_python_type(int, IntMarshal(), None)
register_python_type(int, FloatMarshal(), None)
