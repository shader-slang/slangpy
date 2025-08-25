# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from slangpy import DeviceType
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_to_field_conversion(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "scalar_to_field_test",
        f"""
struct Uniform {{
    float x;
}}

int scalar_to_field_test(Uniform u) {{
    return int(u.x);
}}
""",
    )

    assert kernel_output_values({"x": 512}) == 512


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_to_parameter_conversion(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "scalar_to_parameter_test",
        f"""
int scalar_to_parameter_test(float x) {{
    return int(x);
}}
""",
    )

    assert kernel_output_values(512) == 512


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_array_conversion(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "array_test",
        f"""
int array_test(Array<float, 5> x) {{
    return x[0] + x[1] + x[2] + x[3] + x[4];
}}
""",
    )

    assert kernel_output_values([1, 2, 3, 4, 5]) == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
