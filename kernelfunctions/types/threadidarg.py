

class ThreadIdArg:
    """
    Request the thread id. eg
    void myfunc(int3 input) { }
    """

    def __init__(self, dims: int = 3):
        super().__init__()
        self.dims = dims
