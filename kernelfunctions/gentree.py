from enum import Enum
from typing import Callable, Union


class NodeType(Enum):
    assign = 1
    call = 2
    index = 3
    declare = 4


TNodeOrValue = Union[str, "TNode"]
TNodeArgs = list[TNodeOrValue]
TNode = tuple[NodeType, TNodeArgs]

TGenerator = Callable[[*tuple[TNodeOrValue]], str]


def make_node(op: NodeType, *args: TNodeOrValue) -> TNode:
    return (op, list(args))


def declare(typename: TNodeOrValue, varname: TNodeOrValue) -> TNode:
    return make_node(NodeType.declare, typename, varname)


def gen_declare(*args: TNodeOrValue):
    return f"{eval(args[0])} {eval(args[1])}"


def assign(lhs: TNode, rhs: TNode) -> TNode:
    return make_node(NodeType.assign, lhs, rhs)


def gen_assign(*args: TNodeOrValue):
    return f"{eval(args[0])} = {eval(args[1])}"


def index(array: TNode, idx: TNode) -> TNode:
    return make_node(NodeType.index, array, idx)


def gen_index(*args: TNodeOrValue):
    return f"{eval(args[0])}[{eval(args[1])}]"


def eval(node: TNodeOrValue):
    if isinstance(node, str):
        return node
    else:
        return GENERATORS[node[0]](*node[1])


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
