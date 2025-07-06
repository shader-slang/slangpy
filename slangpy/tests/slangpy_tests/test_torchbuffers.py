# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import time
import threading
import logging
from pathlib import Path
import pytest
import slangpy as spy

import sys

sys.path.append(str(Path(__file__).parent))
import helpers

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)


torch.cuda.init()  # Ensure CUDA is initialized


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


logger = setup_logging()


def run_tensor_race_condition_tests(
    share_context: bool = False, custom_stream: bool = False, share_stream: bool = False
):

    if share_context:
        # Access torch device+stream once to ensure cuda context is initialized,
        # then request the current context handles from slangpy and init device with
        # those handles. This ensures we are using the same context as torch.
        torch.cuda.current_device()
        torch.cuda.current_stream()
        handles = spy.get_cuda_current_context_native_handles()
        device = helpers.get_device(
            spy.DeviceType.cuda, use_cache=False, existing_device_handles=handles
        )
        logger.info(f"Using device '{device.info.adapter_name}' with shared context")
    else:
        # Create a new device without sharing context
        device = helpers.get_device(spy.DeviceType.cuda, use_cache=False)
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
        with torch.cuda.stream(stream):
            for i in range(0, 100):
                torch_tensor.add_(ones)
        stream_handle = spy.NativeHandle.from_cuda_stream(stream.cuda_stream)
    else:
        for i in range(0, 100):
            torch_tensor.add_(ones)
        stream_handle = spy.NativeHandle.from_cuda_stream(torch.cuda.current_stream().cuda_stream)

    # Call the function
    if share_stream:
        # If sharing stream, build command encoder, populate it, then do
        # device submit with the stream handle
        enc = device.create_command_encoder()
        copy_buffers(range(size), dp, out_buffer.storage, _append_to=enc)
        device.submit_command_buffers([enc.finish()], cuda_stream=stream_handle)
    else:
        # If not sharing stream, we can just call the function directly
        copy_buffers(range(size), dp, out_buffer.storage)

    # Ensure all operations complete before checking results
    torch.cuda.synchronize()
    device.wait_for_idle()

    # Pause to give a nice readable gap in the profile
    time.sleep(0.1)

    # Get outputs
    result = out_buffer.to_numpy()
    expected_per_element = 100

    # Check for inconsistency (race condition detected)
    if not (result.min() == result.max() == expected_per_element):
        logger.error(
            f"RACE CONDITION DETECTED! Expected {expected_per_element}, values range from {result.min()} to {result.max()}"
        )
    else:
        logger.info(f"No race condition detected")


if __name__ == "__main__":
    try:
        # In theory, not sharing context is great race condition situation, however from NSight it appears
        # that the default streams of separate contexts on the same device have at least some form of
        # synchronization, or even are fully shared. Hard to tell, but no race condition occurs here.
        run_tensor_race_condition_tests(share_context=False)

        # Expect no race condition when sharing context and not using custom stream, because
        # both torch and slangpy will choose the default stream.
        run_tensor_race_condition_tests(share_context=True, custom_stream=False, share_stream=False)

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
