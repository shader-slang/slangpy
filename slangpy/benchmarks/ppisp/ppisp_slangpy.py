# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
SlangPy wrapper for PPISP pipeline.

Based on github.com/nv-tlabs/ppisp (Apache 2.0).

ISP parameters use DiffTensorView with loadUniform/loadVecUniform for
wave-level gradient reduction in backward (same kernel as slangtorch).
Per-pixel inputs (rgb, pixel_coord, camera/frame idx) are auto-vectorized.

SlangPy's TorchAutoGradHook handles autograd integration automatically.

IMPORTANT: The first call (warmup) MUST pass requires_grad=True on ALL
differentiable tensors so SlangPy initializes its CallData with the
correct execution path for DiffTensorView params.
"""

import os
from typing import Optional

import torch
import torch.nn as nn

PPISP_DEFINES = {"NUM_VIGNETTING_ALPHA_TERMS": "3"}

# Cached SlangPy module
_slang_module = None
_warmed_up = False


def _get_slang_module(spy_device: Optional["spy.Device"] = None):  # noqa: F821
    global _slang_module
    if _slang_module is not None:
        return _slang_module

    import slangpy as spy

    current_dir = os.path.dirname(os.path.abspath(__file__))
    slang_path = os.path.join(current_dir, "ppisp_slangpy.slang").replace("\\", "/")

    if spy_device is None:
        from slangpy.core.utils import create_torch_device

        slangpy_path = os.path.dirname(spy.__file__)
        slangpy_include = os.path.join(slangpy_path, "slang").replace("\\", "/")

        spy_device = create_torch_device(
            type=spy.DeviceType.cuda,
            include_paths=[slangpy_include],
            enable_print=False,
            enable_hot_reload=False,
        )

    # Pass defines via a custom slang session (create_torch_device doesn't accept compiler_options)
    session = spy_device.create_slang_session({
        "include_paths": spy_device.slang_session.desc.compiler_options.include_paths,
        "defines": PPISP_DEFINES,
    })
    raw_module = session.load_module(slang_path)
    _slang_module = spy.Module(raw_module)
    return _slang_module


def _warmup(device: torch.device, spy_device: Optional["spy.Device"] = None):  # noqa: F821
    """Warmup SlangPy module with requires_grad=True to init autograd state.

    The first call must use requires_grad=True on ALL differentiable tensors
    so SlangPy's TorchAutoGradHook initializes the backward kernel path
    with GradOutTensor wrappers for ITensor params. Without this, later
    calls won't produce grad_fn for those tensors.
    """
    global _warmed_up
    if _warmed_up:
        return
    module = _get_slang_module(spy_device)
    module.ppisp(
        batch_size=1,
        num_cameras=1,
        num_frames=1,
        exposure_params=torch.zeros(1, device=device, requires_grad=True),
        vignetting_params=torch.zeros(1, 3, 5, device=device, requires_grad=True),
        color_params=torch.zeros(1, 8, device=device, requires_grad=True),
        crf_params=torch.zeros(1, 3, 4, device=device, requires_grad=True),
        rgb_pixel=torch.zeros(1, 3, device=device, requires_grad=True),
        pixel_coord=torch.zeros(1, 2, device=device),
        camera_idx=torch.zeros(1, device=device, dtype=torch.int16),
        frame_idx=torch.zeros(1, device=device, dtype=torch.int32),
        resolution_w=1920.0,
        resolution_h=1080.0,
    )
    _warmed_up = True


class PPISPSlangPy(nn.Module):
    """PPISP pipeline using SlangPy backend with DiffTensorView.

    ISP parameters use DiffTensorView with wave-level gradient reduction
    (loadUniform/loadVecUniform), matching slangtorch's backward kernel.
    Per-pixel inputs are auto-vectorized by SlangPy for batch dispatch.
    """

    def __init__(self, num_cameras: int, num_frames: int,
                 resolution_w: int = 1920, resolution_h: int = 1080,
                 device: torch.device | str = "cuda",
                 spy_device: Optional["spy.Device"] = None):  # noqa: F821
        super().__init__()
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h

        self.exposure_params = nn.Parameter(torch.zeros(num_frames, device=device))
        self.vignetting_params = nn.Parameter(torch.zeros(num_cameras, 3, 5, device=device))
        self.color_params = nn.Parameter(torch.zeros(num_frames, 8, device=device))

        # CRF init matching OSS
        def _sp_inv(x: float, min_val: float) -> float:
            return float(torch.log(torch.expm1(torch.tensor(max(1e-5, x - min_val)))))

        crf_raw = torch.zeros(4, device=device)
        crf_raw[0] = _sp_inv(1.0, 0.3)
        crf_raw[1] = _sp_inv(1.0, 0.3)
        crf_raw[2] = _sp_inv(1.0, 0.1)
        crf_raw[3] = 0.0
        self.crf_params = nn.Parameter(
            crf_raw.view(1, 1, 4).repeat(num_cameras, 3, 1).contiguous())

        _warmup(device, spy_device)

    def forward(self, rgb: torch.Tensor, pixel_coords: torch.Tensor,
                camera_idcs: torch.Tensor, frame_idcs: torch.Tensor) -> torch.Tensor:
        module = _get_slang_module()
        # Pass all differentiable tensors directly (with requires_grad intact).
        # SlangPy's TorchAutoGradHook handles autograd for DiffTensorView
        # params automatically.
        return module.ppisp(
            batch_size=rgb.shape[0],
            num_cameras=self.num_cameras,
            num_frames=self.num_frames,
            exposure_params=self.exposure_params,
            vignetting_params=self.vignetting_params,
            color_params=self.color_params,
            crf_params=self.crf_params,
            rgb_pixel=rgb,
            pixel_coord=pixel_coords,
            camera_idx=camera_idcs,
            frame_idx=frame_idcs,
            resolution_w=float(self.resolution_w),
            resolution_h=float(self.resolution_h),
        )
