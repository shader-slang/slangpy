# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Slangtorch wrapper for PPISP pipeline.

Based on github.com/nv-tlabs/ppisp (Apache 2.0).
Uses [CUDAKernel] + DiffTensorView with explicit CUDA parallelism.
"""

import os
import time

import torch
import torch.nn as nn

from slangpy.benchmarks.ppisp.ppisp_slangpy import _dispatch_times

PPISP_DEFINES = {"NUM_VIGNETTING_ALPHA_TERMS": "3"}

_native_module = None  # Raw C++ extension module, same as NRE's libppisp_slang_cc


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_native_module():
    """Load the slangtorch PPISP kernel as a raw C++ extension module.

    Tries to load the pre-compiled ppisp_slangtorch_cc.so shipped alongside
    this file (same as NRE's bazel-built libppisp_slang_cc). Falls back to
    JIT compilation via slangtorch if the pre-compiled .so can't be loaded
    (e.g. different platform, Python version, or CUDA toolkit).

        ppisp_slang = _get_native_module()
        ppisp_slang.ppisp(block, grid, ...)           # ~30us
        ppisp_slang.ppisp_bwd_diff(block, grid, ...)  # ~30us
    """
    global _native_module
    if _native_module is not None:
        return _native_module

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try pre-compiled .so first (committed to repo)
    so_path = os.path.join(current_dir, "ppisp_slangtorch_cc.so")
    if os.path.exists(so_path):
        try:
            import importlib.util
            # Find the PyInit_ symbol name baked into the .so
            with open(so_path, "rb") as f:
                data = f.read()
            marker = b"PyInit_"
            idx = data.find(marker)
            if idx == -1:
                raise ImportError("No PyInit_ symbol in .so")
            end = idx + len(marker)
            while end < len(data) and (data[end:end+1].isalnum() or data[end:end+1] == b"_"):
                end += 1
            mod_name = data[idx + len(marker):end].decode("ascii")
            spec = importlib.util.spec_from_file_location(mod_name, so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Verify it has the expected functions
            if hasattr(mod, "ppisp") and hasattr(mod, "ppisp_bwd_diff"):
                _native_module = mod
                return _native_module
        except (ImportError, OSError):
            pass  # Fall through to JIT compilation

    # Fallback: JIT compile via slangtorch and return raw native module
    import slangtorch.slangtorch as st

    slang_path = os.path.join(current_dir, "ppisp_slangtorch.slang")

    options_hash = st.getHash([PPISP_DEFINES, ["--use_fast_math", "--generate-line-info"], [], current_dir], truncate_at=16)
    base_name = "ppisp_slangtorch"
    base_output_folder = os.path.join(current_dir, ".slangtorch_cache", base_name)
    output_folder = os.path.join(base_output_folder, options_hash)
    module_name = f"_slangtorch_{base_name}_{options_hash}"

    lock_file = os.path.join(current_dir, f"ppisp_slangtorch.slang{options_hash}.lock")
    from filelock import FileLock
    with FileLock(lock_file):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        options = st.makeOptionsList(PPISP_DEFINES)
        build_dir, build_id = st.getLatestDir(output_folder, output_folder)

        if build_dir is not None:
            needs_recompile = st._loadModule(
                slang_path, f"{module_name}_{build_id}", build_dir, options,
                sourceDir=output_folder, verbose=False, includePaths=[], dryRun=True,
                skipNinjaCheck=False, extraCudaFlags=["--use_fast_math", "--generate-line-info"],
                extraSlangFlags=[],
            )
        else:
            needs_recompile = True

        if needs_recompile:
            build_dir, build_id = st.getOrCreateUniqueDir(output_folder, output_folder)

        _native_module = st._loadModule(
            slang_path, f"{module_name}_{build_id}", build_dir, options,
            sourceDir=output_folder, verbose=False, includePaths=[], dryRun=False,
            skipNinjaCheck=False, extraCudaFlags=["--use_fast_math", "--generate-line-info"],
            extraSlangFlags=[],
        )
        st.addLoadedDirectoryEntry(output_folder, build_dir)

    return _native_module


class PPISPSlangtorchFunction(torch.autograd.Function):
    """Custom autograd for PPISP slangtorch backend."""

    @staticmethod
    def forward(
        ctx,
        batch_size,
        num_cameras,
        num_frames,
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb,
        pixel_coords,
        camera_idcs,
        frame_idcs,
        resolution_w,
        resolution_h,
    ):
        ppisp_slang = _get_native_module()
        rgb_out = torch.empty_like(rgb)
        block = (32, 1, 1)
        grid = (div_up(batch_size, 32), 1, 1)

        t0 = time.perf_counter()
        ppisp_slang.ppisp(
            block, grid,
            batch_size, num_cameras, num_frames,
            (exposure_params, (exposure_params,)),
            (vignetting_params, (vignetting_params,)),
            (color_params, (color_params,)),
            (crf_params, (crf_params,)),
            (rgb, (rgb,)),
            (rgb_out, (rgb_out,)),
            pixel_coords,
            camera_idcs,
            frame_idcs,
            float(resolution_w),
            float(resolution_h),
        )
        _dispatch_times["slangtorch.fwd"].append((time.perf_counter() - t0) * 1e6)

        ctx.save_for_backward(
            exposure_params, vignetting_params, color_params, crf_params, rgb, rgb_out
        )
        ctx.batch_size = batch_size
        ctx.num_cameras = num_cameras
        ctx.num_frames = num_frames
        ctx.pixel_coords = pixel_coords
        ctx.camera_idcs = camera_idcs
        ctx.frame_idcs = frame_idcs
        ctx.resolution_w = resolution_w
        ctx.resolution_h = resolution_h

        return rgb_out

    @staticmethod
    def backward(ctx, grad_output):
        (exposure_params, vignetting_params, color_params, crf_params, rgb, rgb_out) = (
            ctx.saved_tensors
        )

        grad_exposure = torch.zeros_like(exposure_params)
        grad_vignetting = torch.zeros_like(vignetting_params)
        grad_color = torch.zeros_like(color_params)
        grad_crf = torch.zeros_like(crf_params)
        grad_rgb_in = torch.empty_like(rgb)
        grad_output = grad_output.contiguous()

        ppisp_slang = _get_native_module()
        block = (32, 1, 1)
        grid = (div_up(ctx.batch_size, 32), 1, 1)

        t0 = time.perf_counter()
        ppisp_slang.ppisp_bwd_diff(
            block, grid,
            ctx.batch_size, ctx.num_cameras, ctx.num_frames,
            (exposure_params, (grad_exposure,)),
            (vignetting_params, (grad_vignetting,)),
            (color_params, (grad_color,)),
            (crf_params, (grad_crf,)),
            (rgb, (grad_rgb_in,)),
            (rgb_out, (grad_output,)),
            ctx.pixel_coords,
            ctx.camera_idcs,
            ctx.frame_idcs,
            float(ctx.resolution_w),
            float(ctx.resolution_h),
        )
        _dispatch_times["slangtorch.bwd"].append((time.perf_counter() - t0) * 1e6)

        return (
            None,
            None,
            None,  # batch_size, num_cameras, num_frames
            grad_exposure,
            grad_vignetting,
            grad_color,
            grad_crf,
            grad_rgb_in,
            None,
            None,
            None,  # pixel_coords, camera_idcs, frame_idcs
            None,
            None,  # resolution_w, resolution_h
        )


class PPISPSlangtorch(nn.Module):
    """PPISP pipeline using slangtorch backend."""

    def __init__(
        self,
        num_cameras: int,
        num_frames: int,
        resolution_w: int = 1920,
        resolution_h: int = 1080,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h

        self.exposure_params = nn.Parameter(torch.zeros(num_frames, device=device))
        self.vignetting_params = nn.Parameter(torch.zeros(num_cameras, 3, 5, device=device))
        self.color_params = nn.Parameter(torch.zeros(num_frames, 8, device=device))

        def _sp_inv(x: float, min_val: float) -> float:
            return float(torch.log(torch.expm1(torch.tensor(max(1e-5, x - min_val)))))

        crf_raw = torch.zeros(4, device=device)
        crf_raw[0] = _sp_inv(1.0, 0.3)
        crf_raw[1] = _sp_inv(1.0, 0.3)
        crf_raw[2] = _sp_inv(1.0, 0.1)
        crf_raw[3] = 0.0
        self.crf_params = nn.Parameter(crf_raw.view(1, 1, 4).repeat(num_cameras, 3, 1).contiguous())

        self.to(device)

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        camera_idcs: torch.Tensor,
        frame_idcs: torch.Tensor,
    ) -> torch.Tensor:
        return PPISPSlangtorchFunction.apply(
            rgb.shape[0],
            self.num_cameras,
            self.num_frames,
            self.exposure_params,
            self.vignetting_params,
            self.color_params,
            self.crf_params,
            rgb,
            pixel_coords,
            camera_idcs,
            frame_idcs,
            self.resolution_w,
            self.resolution_h,
        )
