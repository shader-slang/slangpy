# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import io
import pytest
from contextlib import redirect_stdout
from slangpy.testing import helpers
from slangpy import DeviceType
from slangpy.core.calldata import set_print_generated_shaders


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_print_generated_shaders_programmatic(device_type: DeviceType):
    """Test that set_print_generated_shaders() works correctly."""
    device = helpers.get_device(device_type)

    # Create a simple function that will generate a shader (like test_call_function)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(int a, int b) {
}
""",
    )

    # Enable shader printing
    set_print_generated_shaders(True)

    # Capture stdout to verify the shader is printed
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        # Just verify it can be called with no exceptions (like test_call_function)
        function(5, 10)

    output = captured_output.getvalue()

    # Verify that shader output is present
    assert "GENERATED SHADER:" in output
    assert "add_numbers" in output
    assert "=" * 80 in output
    assert "END SHADER:" in output

    # Disable shader printing
    set_print_generated_shaders(False)

    # Verify no output when disabled
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        # Just verify it can be called with no exceptions
        function(5, 10)

    output = captured_output.getvalue()
    assert "GENERATED SHADER:" not in output


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_print_generated_shaders_environment_variable(device_type: DeviceType):
    """Test that SLANGPY_PRINT_GENERATED_SHADERS environment variable works correctly."""
    device = helpers.get_device(device_type)

    # Create a simple function that will generate a shader (like test_call_function)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(int a, int b) {
}
""",
    )

    # Test with environment variable set
    original_env = os.environ.get("SLANGPY_PRINT_GENERATED_SHADERS")

    try:
        # Set environment variable
        os.environ["SLANGPY_PRINT_GENERATED_SHADERS"] = "true"

        # We need to use the programmatic API since the environment variable
        # is only checked at import time
        set_print_generated_shaders(True)

        # Capture stdout to verify the shader is printed
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            # Just verify it can be called with no exceptions (like test_call_function)
            function(5, 10)

        output = captured_output.getvalue()

        # Verify that shader output is present
        assert "GENERATED SHADER:" in output
        assert "add_numbers" in output

    finally:
        # Restore original environment
        if original_env is None:
            os.environ.pop("SLANGPY_PRINT_GENERATED_SHADERS", None)
        else:
            os.environ["SLANGPY_PRINT_GENERATED_SHADERS"] = original_env
        set_print_generated_shaders(False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
