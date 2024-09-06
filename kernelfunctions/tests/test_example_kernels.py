from pathlib import Path
import numpy as np
import pytest
import sgl
import kernelfunctions as kf
from kernelfunctions.tests import helpers
from helpers import test_id  # type: ignore (pytest fixture)


def rand_array_of_floats(size: int):
    return np.random.rand(size).astype(np.float32)


# Verify a 'hard coded' example of a generated kernel compiles and runs
# correctly.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffered_scalar_function(test_id: str, device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)

    user_func_module = device.load_module_from_source(
        f"user_module_{test_id}",
        r"""
[Differentiable]
void user_func(float a, float b, out float c) {
    c = a*a + b + 1;
}
""",
    )

    # Load the example shader, with the custom user function at the top.
    generated_module = device.load_module_from_source(
        f"generated_module_{test_id}",
        f'import "user_module_{test_id}";\n'
        + open(Path(__file__).parent / "test_example_kernel_scalar.slang").read(),
    )

    # Create the forward and backward kernels.
    ep = generated_module.entry_point("main")
    program = device.link_program([generated_module, user_func_module], [ep])
    kernel = device.create_compute_kernel(program)
    backwards_ep = generated_module.entry_point("main_backwards")
    backwards_program = device.link_program(
        [generated_module, user_func_module], [backwards_ep]
    )
    backwards_kernel = device.create_compute_kernel(backwards_program)

    # Create input buffer 0 with random numbers and an empty gradient buffer (ignored).
    in_buffer_0 = kf.StructuredBuffer(
        element_count=64, device=device, element_type=float, requires_grad=True
    )
    in_buffer_0.buffer.from_numpy(rand_array_of_floats(in_buffer_0.element_count))
    in_buffer_0.grad_buffer.from_numpy(
        np.zeros(in_buffer_0.element_count, dtype=np.float32))  # type: ignore

    # Same with input buffer 1.
    in_buffer_1 = kf.StructuredBuffer(
        element_count=64, device=device, element_type=float, requires_grad=True
    )
    in_buffer_1.buffer.from_numpy(rand_array_of_floats(in_buffer_1.element_count))
    in_buffer_1.grad_buffer.from_numpy(
        np.zeros(in_buffer_1.element_count, dtype=np.float32))  # type: ignore

    # Create empty output buffer with gradients initialized to 1 (as there is 1-1 correspondence between
    # output of user function and output of kernel)
    out_buffer = kf.StructuredBuffer(
        element_count=64, device=device, element_type=float, requires_grad=True
    )
    out_buffer.buffer.from_numpy(np.zeros(out_buffer.element_count, dtype=np.float32))
    out_buffer.grad_buffer.from_numpy(
        np.ones(out_buffer.element_count, dtype=np.float32))  # type: ignore

    # Dispatch the forward kernel.
    kernel.dispatch(
        sgl.uint3(64, 1, 1),
        {
            "call_data": {
                "a": in_buffer_0.buffer,
                "b": in_buffer_1.buffer,
                "c": out_buffer.buffer,
            }
        },
    )

    # Read and validate forward kernel results (expecting c = a*a + b + 1)
    in_data_0 = in_buffer_0.buffer.to_numpy().view(np.float32)
    in_data_1 = in_buffer_1.buffer.to_numpy().view(np.float32)
    out_data = out_buffer.buffer.to_numpy().view(np.float32)
    eval_data = in_data_0 * in_data_0 + in_data_1 + 1
    assert np.allclose(out_data, eval_data)

    # Dispatch the backward kernel.
    backwards_kernel.dispatch(
        sgl.uint3(64, 1, 1),
        {
            "call_data": {
                "a": in_buffer_0.buffer,
                "a_grad": in_buffer_0.grad_buffer,
                "b": in_buffer_1.buffer,
                "b_grad": in_buffer_1.grad_buffer,
                "c": out_buffer.buffer,
                "c_grad": out_buffer.grad_buffer,
            }
        },
    )

    # Read and validate backward kernel results (expecting a_grad = 2*a, b_grad = 1)
    in_grad_0 = in_buffer_0.grad_buffer.to_numpy().view(np.float32)  # type: ignore
    in_grad_1 = in_buffer_1.grad_buffer.to_numpy().view(np.float32)  # type: ignore
    eval_grad_0 = 2 * in_data_0
    eval_grad_1 = np.ones(in_data_1.shape)
    assert np.allclose(in_grad_0, eval_grad_0)
    assert np.allclose(in_grad_1, eval_grad_1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
