# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Demo script to show the new Fuser functions."""

import slangpy as spy
import slangpy.testing.helpers as helpers
from slangpy.experimental.fuse import FuseNode, Fuser

# Setup
device = helpers.get_device(spy.DeviceType.d3d12)
module = spy.Module.load_from_file(device, "slangpy/tests/slangpy_tests/fusetest.slang")

# Create a simple graph with children
mul_node = FuseNode.from_function(module.require_function("ft_mul"))
add_node = FuseNode.from_function(module.require_function("ft_add"))
root = FuseNode("compute", ["p", "q", "r"], ["final"])

root.children.append(mul_node)
root.children.append(add_node)

mul_node.get_input("a").source = (None, "p")
mul_node.get_input("b").source = (None, "q")
add_node.get_input("a").source = (mul_node, "_result")
add_node.get_input("b").source = (None, "r")
root.get_output("final").source = (add_node, "_result")

# Create fuser
fuser = Fuser(root)

print("=" * 80)
print("BEFORE TYPE INFERENCE:")
print("=" * 80)
print(fuser.dump_graph())
print()

# Run type inference
fuser._infer_types(root)

print("=" * 80)
print("AFTER TYPE INFERENCE:")
print("=" * 80)
print(fuser.dump_graph())
print()

# Clear type info
fuser.clear_type_info()

print("=" * 80)
print("AFTER CLEARING TYPE INFO:")
print("=" * 80)
print(fuser.dump_graph())
