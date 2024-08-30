from enum import Enum
from typing import Callable, Sequence, Union


class NodeType(Enum):
    assign = (1,)
    call = (2,)
    index = (3,)
    declare = 4


TNodeOrValue = Union[str, "TNode"]
TNodeArgs = tuple[TNodeOrValue, ...]
TNode = tuple[NodeType, TNodeArgs]

TGenerator = Callable[[*TNodeArgs], str]


def make_node(op: NodeType, *args: TNodeOrValue) -> TNode:
    return (op, args)


def eval(node: TNodeOrValue):
    if isinstance(node, str):
        return node
    else:
        return GENERATORS[node[0]](*node[1])


def gen_assign(*args: TNodeOrValue):
    return f"{eval(args[0])} = {eval(args[1])}"


def gen_declare(*args: TNodeOrValue):
    return f"{eval(args[0])} {eval(args[1])}"


def gen_index(*args: TNodeOrValue):
    return f"{eval(args[0])}[{eval(args[1])}]"


GENERATORS: dict[NodeType, TGenerator] = {
    NodeType.assign: gen_assign,
    NodeType.declare: gen_declare,
    NodeType.index: gen_index,
}

print(
    eval(
        make_node(
            NodeType.assign,
            make_node(NodeType.declare, "int", "a"),
            make_node(NodeType.index, "b", "0"),
        )
    )
)
