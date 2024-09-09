

class RandFloatArg:
    """
    Request random floats from a wang hash function. eg
    void myfunc(float3 input) { }
    """

    def __init__(self, min: float, max: float, dim: int):
        super().__init__()
        self.min = float(min)
        self.max = float(max)
        self.dim = int(dim)
