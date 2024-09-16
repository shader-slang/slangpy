class SlangCodeGen:
    def __init__(self, indent: int = 0):
        super().__init__()
        self.cur_indent = indent
        self.partial_line = ""
        self.output = []

    def blank_line(self):
        self.output += [""]

    def write(self, text: str):
        self.partial_line += text

    def emit(self, text: str):
        self.output.append(("    " * self.cur_indent) + self.partial_line + text)
        self.partial_line = ""

    def indent(self):
        self.cur_indent += 1

    def dedent(self):
        self.cur_indent -= 1

    def append(self, text: str):
        assert self.output
        self.output[-1] += text

    def comment(self, text: str):
        self.emit(f"// {text}")

    def begin_block(self, text: str = ""):
        if text:
            self.emit(text)
        self.indent()

    def end_block(self, text: str = ""):
        self.dedent()
        if text:
            self.emit(text)

    def namespace(self, name: str):
        self.begin_block(f"namespace {name} {{")

    def code(self):
        return "\n".join(self.output)
