# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Kernel generation test.

These tests exercise different code paths for kernel generation, to exercise different kernel types, such as:
- passing arguments directly vs via call data
- passing read-only arguments that don't need storing directly rather than via marshalls
- handling the semantic 'dispatch thread id' etc and calling kernels directly
"""

from typing import Any

import pytest
import os

import slangpy as spy
from slangpy.testing import helpers

PRINT_TEST_KERNEL_GEN = os.getenv("PRINT_TEST_KERNEL_GEN", "0") == "1"


def generate_code(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> str:
    """
    Generate code for the given function and arguments, and return the generated code as a string.
    """
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.debug_build_call_data(*args, **kwargs)
    if PRINT_TEST_KERNEL_GEN:
        print(cd.code)
    return cd.code


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_kernel_gen_basic(device_type: spy.DeviceType):
    """
    Test basic kernel generation with a simple function that adds two numbers.
    """
    src = """
int add(int a, int b) {
    return a + b;
}
"""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", src, 1, 2)
    print(code)
    assert "add" in code


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
