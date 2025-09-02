# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import time
import logging
import sys
from pathlib import Path

import slangpy as spy
from slangpy.testing import helpers

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)


def setup_logging():
    """Setup logging to both console and file"""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"torch_race_test_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")
    return logger


logger = None


def run_tensor_race_condition_tests(
    share_context: bool = False, custom_stream: bool = False, share_stream: bool = False
):
    # Use global logger if nothing specified
    global logger
    if logger is None:
        logger = logging.getLogger()

    # Ensure CUDA is initialized
    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()

    initial_handles = spy.get_cuda_current_context_native_handles()
    handles = spy.get_cuda_current_context_native_handles()

    if share_context:
        # Access torch device+stream once to ensure cuda context is initialized,
        # then request the current context handles from slangpy and init device with
        # those handles. This ensures we are using the same context as torch.
        device = helpers.get_device(
            spy.DeviceType.cuda, use_cache=False, existing_device_handles=handles
        )
        logger.info(f"Using device '{device.info.adapter_name}' with shared context")
    else:
        # Create a new device without sharing context
        handles[1] = spy.NativeHandle()  # clear context so only device handle is shared
        device = helpers.get_device(
            spy.DeviceType.cuda, use_cache=False, existing_device_handles=handles
        )
        logger.info(f"Using device '{device.info.adapter_name}' with new context")

    # Create a nice big tensor to make gpu jobs take long enough to see race conditions.
    size = 100_000_000
    torch_tensor = torch.zeros((size,), dtype=torch.float32, device=torch.device("cuda"))
    dp = torch_tensor.data_ptr()

    # Create tensor of 1s to add to the torch tensor each iteration (this is slower than adding constant)
    ones = torch.ones((size,), dtype=torch.float32, device=torch.device("cuda"))

    # Create slangpy function that copies from input to output buffer by pointers
    copy_buffers = helpers.create_function_from_module(
        device,
        "copy_buffers",
        r"""
void copy_buffers(int call_id, float* in_buffer, float* out_buffer) {
    out_buffer[call_id] = in_buffer[call_id];
}
""",
    )

    # Create output buffer
    out_buffer = spy.NDBuffer.empty(device, (size,), dtype="float")

    # Run the test function once and wait for device to be idle, to avoid compile
    # times interfering
    copy_buffers(range(size), dp, out_buffer.storage)
    device.wait_for_idle()

    # Run torch either on a custom stream or the default stream
    if custom_stream:
        stream = torch.cuda.Stream()

    iterations = 3

    for i in range(iterations):
        if custom_stream:
            with torch.cuda.stream(stream):  # type: ignore
                for i in range(0, 20):
                    torch_tensor.add_(ones)
            stream_handle = spy.NativeHandle.from_cuda_stream(stream.cuda_stream)
        else:
            for i in range(0, 20):
                torch_tensor.add_(ones)
            stream_handle = spy.NativeHandle.from_cuda_stream(
                torch.cuda.current_stream().cuda_stream
            )

        # Call the function
        if share_stream:
            # If sharing stream, build command encoder, populate it, then do
            # device submit with the stream handle
            enc = device.create_command_encoder()
            for j in range(0, 10):
                copy_buffers(range(size), dp, out_buffer.storage, _append_to=enc)
            device.submit_command_buffers([enc.finish()], cuda_stream=stream_handle)
        else:
            # If not sharing stream, we can just call the function directly
            copy_buffers(range(size), dp, out_buffer.storage)
        time.sleep(0.1)  # Sleep to simulate work

    # Ensure all operations complete before checking results
    torch.cuda.synchronize()
    device.wait_for_idle()

    # Pause to give a nice readable gap in the profile
    time.sleep(0.1)

    # Get outputs
    result = out_buffer.to_numpy()
    expected_per_element = 20 * iterations

    # Shut down device as its owned by torch
    device.close()
    device = None

    final_handles = spy.get_cuda_current_context_native_handles()
    assert initial_handles[1].type == spy.NativeHandleType.CUcontext
    assert initial_handles[1].value == final_handles[1].value

    # Check for inconsistency (race condition detected)
    if not (result.min() == result.max() == expected_per_element):
        logger.info(
            f"RACE CONDITION DETECTED! Expected {expected_per_element}, values range from {result.min()} to {result.max()}"
        )
        return True
    else:
        logger.info(f"No race condition detected")
        return False


# Pytest for our most common default cuda-interop case, in which we've configured pytorch
# and slangpy to share the same context and stream.
@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_shared_context_and_stream(device_type: spy.DeviceType):
    assert (
        run_tensor_race_condition_tests(share_context=True, custom_stream=False, share_stream=True)
        == False
    )


# Pytest for none-shared context case, which appears to avoid race conditions through some level
# of synchronization in the default streams of separate contexts. For now this has shown not
# to cause race conditions, so testing for that behaviour.
@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_non_shared_context(device_type: spy.DeviceType):
    assert run_tensor_race_condition_tests(share_context=False) == False


# Pytest for known race condition case, where we use a custom stream in torch but not sharing it with slangpy.
@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_custom_stream_no_share(device_type: spy.DeviceType):
    pytest.skip("Race condition doesn't reproduce reliably on CI machines of varying specs")
    assert (
        run_tensor_race_condition_tests(share_context=True, custom_stream=True, share_stream=False)
        == True
    )


# Pytest that removes the race condition by sharing the custom stream
@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_custom_stream_share(device_type: spy.DeviceType):
    assert (
        run_tensor_race_condition_tests(share_context=True, custom_stream=True, share_stream=True)
        == False
    )


# When running this file as main, trigger a sequence of race condition repros to prove
# out expected behaviour
if __name__ == "__main__":
    logger = setup_logging()

    pytest.main(["-v", "-s", __file__])
    exit(0)

    try:
        # In theory, not sharing context is great race condition situation, however from NSight it appears
        # that the default streams of separate contexts on the same device have at least some form of
        # synchronization, or even are fully shared. Hard to tell, but no race condition occurs here.
        run_tensor_race_condition_tests(share_context=False)

        # Expect no race condition when sharing context and not using custom stream, because
        # both torch and slangpy will choose the default stream.
        run_tensor_race_condition_tests(share_context=True, custom_stream=False, share_stream=True)

        # Should be identical when sharing a stream, as we're just being explicit about sharing
        # the default stream.
        run_tensor_race_condition_tests(share_context=True, custom_stream=False, share_stream=True)

        # Expect race condition if switching torch to a custom stream but not sharing it
        run_tensor_race_condition_tests(share_context=True, custom_stream=True, share_stream=False)

        # Expect no race condition when sharing the custom stream
        run_tensor_race_condition_tests(share_context=True, custom_stream=True, share_stream=True)

    except Exception as e:
        logger.exception(f"Test failed with exception: {e}")
        sys.exit(1)  # Exit with different error code for exceptions
