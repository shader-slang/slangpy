def diff_pair(primal: str, derivative: str = "0"):
    return f"diffPair({primal}, {derivative})"


def declare(type_name: str, variable_name: str):
    return f"{type_name} {variable_name}"


def assign(target: str, value: str):
    return f"{target} = {value}"


def declarevar(variable_name: str, value: str):
    return f"var {variable_name} = {value}"


def attribute(object_name: str, attribute_name: str):
    return f"{object_name}.{attribute_name}"


def statement(statement: str, indent: int = 0):
    return "    " * indent + statement + ";"
