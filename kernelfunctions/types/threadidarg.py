

class ThreadIdArg:
    """
    Dummy class to allow for passing thread id kernel functions.
    """

    def __init__(self, dims: int = 3):
        super().__init__()
        self.dims = dims
