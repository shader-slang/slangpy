# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Measures how many host-to-device copies the CUDA by-reference descriptor-table
# ABI (shader-slang/slang#11939) adds to the command stream, by dispatching the
# converting shape (TensorList - a struct carrying a fixed-size tensor array)
# under Nsight Systems and counting cuMemcpyHtoDAsync in the API summary.
#
# The claim under test: the payload rides slang-rhi's pooled global upload,
# which runs ONCE PER COMMAND BUFFER (cuda-command.cpp: CommandExecutor::execute
# calls m_constantBufferPool.upload() once before replaying commands) - not once
# per parameter, and not via a per-dispatch __constant__ symbol copy.
#
# Usage (on a CUDA machine, run from slangpy/benchmarks/ so the .slang module
# is on the search path):
#
#   nsys profile -t cuda --stats=true -o byref-separate \
#       python profile_byref_pool_upload.py --mode separate --dispatches 100
#   nsys profile -t cuda --stats=true -o byref-batched \
#       python profile_byref_pool_upload.py --mode batched --dispatches 100
#
# Read the `cuda_api_sum` table of each report:
#   - separate: ~100 cuMemcpyHtoDAsync (one command buffer per call -> one pooled
#     upload per dispatch; slangpy's default calling pattern) + fixed setup copies
#   - batched:  a small flat count (~1 per used global-pool page, e.g. 3-5) for
#     all 100 dispatches - upload() issues one copy per pool page, so the count
#     tracks payload volume, not dispatch count, and stays flat as --dispatches
#     grows + the same fixed setup copies
#
# The separate-vs-batched delta isolates the pooled upload from setup traffic
# (tensor creation, module init), and batched staying at ~1 while --dispatches
# grows is the direct demonstration that the upload is per-command-buffer.
# For an A/B against the legacy ABI, rerun with a Slang built without #11939 or
# with the kernel compiled under -cuda-entry-point-params-by-value: the pooled
# upload disappears and the same payload travels inside the kernel-argument blob.

import argparse

import numpy as np

import slangpy as spy
from slangpy.testing import helpers

COUNT = 8  # tensors in the descriptor table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["separate", "batched"], default="separate")
    parser.add_argument("--dispatches", type=int, default=100)
    args = parser.parse_args()

    device = helpers.get_device(spy.DeviceType.cuda)

    inputs = [np.random.rand(256, 256).astype(np.float32) for _ in range(COUNT)]
    input_tensors = [spy.Tensor.from_numpy(device, x) for x in inputs]
    result_tensor = spy.Tensor.empty(device, shape=(256, 256), dtype=float)

    module = spy.Module(device.load_module("test_benchmark_tensor.slang"))
    func = module.require_function(f"sum_indirect<{COUNT}>")

    kwargs = dict(
        tid=spy.call_id(),
        tensor_list={"tensors": input_tensors},
        tensor_indices=list(range(COUNT)),
        _result=result_tensor,
    )

    # Warmup: compile the kernel and touch all pools outside the measured region.
    func(**kwargs)
    device.wait_for_idle()

    if args.mode == "separate":
        # slangpy's default pattern: one command buffer (and thus one pooled
        # upload) per call.
        for _ in range(args.dispatches):
            func(**kwargs)
    else:
        # One command buffer for all dispatches: the pooled upload should appear
        # once, no matter how many dispatches are encoded.
        encoder = device.create_command_encoder()
        for _ in range(args.dispatches):
            func.append_to(encoder, **kwargs)
        device.submit_command_buffer(encoder.finish())

    device.wait_for_idle()

    assert np.allclose(result_tensor.to_numpy(), sum(inputs), atol=1e-3)
    print(f"mode={args.mode} dispatches={args.dispatches} ok")
    print("Now read the cuda_api_sum table of the nsys report for cuMemcpyHtoDAsync.")


if __name__ == "__main__":
    main()
