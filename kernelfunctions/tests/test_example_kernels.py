from pathlib import Path
import numpy as np
import pytest
import sgl
import kernelfunctions as kf
from kernelfunctions.buffer import StructuredBuffer
from kernelfunctions.tests import helpers
from helpers import test_id
# type: ignore (pytest fixture)
from kernelfunctions.tests.test_differential_function_call import python_eval_polynomial, python_eval_polynomial_a_deriv, python_eval_polynomial_b_deriv


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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vec3_call_with_buffers_soa(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    program = device.load_program(
        str(Path(__file__).parent / "generated_tests/polynomial_soa.slang"), ["main"])

    kernel_eval_polynomial = device.create_compute_kernel(program)

    a_x = StructuredBuffer(
        element_count=32,
        device=device,
        element_type=sgl.float1,
        requires_grad=True,
    )
    a_x.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    a_y = StructuredBuffer(
        element_count=32,
        device=device,
        element_type=sgl.float1,
        requires_grad=True,
    )
    a_y.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    a_z = StructuredBuffer(
        element_count=32,
        device=device,
        element_type=sgl.float1,
        requires_grad=True,
    )
    a_z.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    b = StructuredBuffer(
        element_count=32,
        device=device,
        element_type=sgl.float3,
        requires_grad=True,
    )
    b.buffer.from_numpy(np.random.rand(32*3).astype(np.float32))

    res = StructuredBuffer(
        element_count=32,
        device=device,
        element_type=sgl.float3,
        requires_grad=True,
    )

    call_data = {
        'x': a_x.buffer,
        'y': a_y.buffer,
        'z': a_z.buffer,
        'b': b.buffer,
        'res': res.buffer
    }

    call_data["_call_stride"] = [1]
    call_data["_call_dim"] = [32]
    call_data["_thread_count"] = uint3(total_threads, 1, 1)

    # Dispatch the kernel.
    self.kernel.dispatch(uint3(total_threads, 1, 1), {"call_data": call_data})

    kernel_eval_polynomial.dispatch(thread_count=sgl.uint3(32, 1, 1), )

    a_x_data = a_x.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_y_data = a_y.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_z_data = a_z.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_data = np.column_stack((a_x_data, a_y_data, a_z_data))
    b_data = b.buffer.to_numpy().view(np.float32).reshape(-1, 3)
    expected = python_eval_polynomial(a_data, b_data)
    res_data = res.buffer.to_numpy().view(np.float32).reshape(-1, 3)

    assert np.allclose(res_data, expected)

    res.grad_buffer.from_numpy(np.ones(32*3, dtype=np.float32))

    kernel_eval_polynomial.backwards({
        'x': a_x,
        'y': a_y,
        'z': a_z
    }, b, res)
    a_x_grad_data = a_x.grad_buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_y_grad_data = a_y.grad_buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_z_grad_data = a_z.grad_buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_grad_data = np.column_stack((a_x_grad_data, a_y_grad_data, a_z_grad_data))
    b_grad_data = b.grad_buffer.to_numpy().view(np.float32).reshape(-1, 3)

    exprected_grad = python_eval_polynomial_a_deriv(a_data, b_data)
    assert np.allclose(a_grad_data, exprected_grad)

    exprected_grad = python_eval_polynomial_b_deriv(a_data, b_data)
    assert np.allclose(b_grad_data, exprected_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
