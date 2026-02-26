# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Slangtorch wrapper for PPISP pipeline.

Based on github.com/nv-tlabs/ppisp (Apache 2.0).
Uses [CUDAKernel] + DiffTensorView with explicit CUDA parallelism.
"""

import os

import torch
import torch.nn as nn

PPISP_DEFINES = {"NUM_VIGNETTING_ALPHA_TERMS": "3"}

_slang_module = None


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_slang_module():
    global _slang_module
    if _slang_module is not None:
        return _slang_module

    import slangtorch

    current_dir = os.path.dirname(os.path.abspath(__file__))
    slang_path = os.path.join(current_dir, "ppisp_slangtorch.slang")

    _slang_module = slangtorch.loadModule(
        slang_path,
        verbose=False,
        defines=PPISP_DEFINES,
    )
    return _slang_module


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
        module = _get_slang_module()
        rgb_out = torch.empty_like(rgb)

        module.ppisp(
            batch_size=batch_size,
            num_cameras=num_cameras,
            num_frames=num_frames,
            exposure_params=exposure_params,
            vignetting_params=vignetting_params,
            color_params=color_params,
            crf_params=crf_params,
            rgb_in=rgb,
            rgb_out=rgb_out,
            pixel_coords=pixel_coords,
            camera_idcs=camera_idcs,
            frame_idcs=frame_idcs,
            resolution_w=float(resolution_w),
            resolution_h=float(resolution_h),
        ).launchRaw(blockSize=(32, 1, 1), gridSize=(div_up(batch_size, 32), 1, 1))

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

        module = _get_slang_module()

        module.ppisp.bwd(
            batch_size=ctx.batch_size,
            num_cameras=ctx.num_cameras,
            num_frames=ctx.num_frames,
            exposure_params=(exposure_params, grad_exposure),
            vignetting_params=(vignetting_params, grad_vignetting),
            color_params=(color_params, grad_color),
            crf_params=(crf_params, grad_crf),
            rgb_in=(rgb, grad_rgb_in),
            rgb_out=(rgb_out, grad_output),
            pixel_coords=ctx.pixel_coords,
            camera_idcs=ctx.camera_idcs,
            frame_idcs=ctx.frame_idcs,
            resolution_w=float(ctx.resolution_w),
            resolution_h=float(ctx.resolution_h),
        ).launchRaw(blockSize=(32, 1, 1), gridSize=(div_up(ctx.batch_size, 32), 1, 1))

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
