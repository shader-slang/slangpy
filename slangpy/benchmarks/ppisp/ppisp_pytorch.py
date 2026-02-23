# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pure PyTorch reference implementation of the PPISP pipeline.

Based on github.com/nv-tlabs/ppisp (Apache 2.0).
See: Deutsch et al., "Physically-Plausible Compensation and Control of
     Photometric Variations in Radiance Field Reconstruction",
     arXiv:2601.18336, 2026.

Pipeline stages:
  1. Exposure compensation  - per-frame EV offset        (Debevec & Malik, SIGGRAPH 1997)
  2. Vignetting             - per-camera radial falloff   (Goldman, TPAMI 2010)
  3. Color correction       - per-frame RGI homography    (Finlayson et al., TPAMI 2019)
  4. Camera response (CRF)  - per-camera toe-shoulder     (Grossberg & Nayar, TPAMI 2004)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral]
# Matches ppisp/__init__.py and tests/torch_reference.py from OSS repo
_COLOR_PINV_BLOCK_DIAG = torch.block_diag(
    torch.tensor([[0.0480542, -0.0043631], [-0.0043631, 0.0481283]]),   # Blue
    torch.tensor([[0.0580570, -0.0179872], [-0.0179872, 0.0431061]]),   # Red
    torch.tensor([[0.0433336, -0.0180537], [-0.0180537, 0.0580500]]),   # Green
    torch.tensor([[0.0128369, -0.0034654], [-0.0034654, 0.0128158]]),   # Neutral
).to(torch.float32)

NUM_VIGNETTING_ALPHA_TERMS = 3
CRF_PARAMS_PER_CHANNEL = 4


# -- Helpers ------------------------------------------------------------------

def _get_homography(color_params: Tensor, frame_idx: int) -> Tensor:
    """Compute color correction homography from latent params.

    Port of _get_homography_torch() from ppisp OSS tests/torch_reference.py.
    """
    cp = color_params[frame_idx]  # [8]
    device = cp.device

    block_diag = _COLOR_PINV_BLOCK_DIAG.to(device)
    offsets = cp @ block_diag  # [8]

    bd = offsets[0:2]  # blue
    rd = offsets[2:4]  # red
    gd = offsets[4:6]  # green
    nd = offsets[6:8]  # neutral

    # Fixed source chromaticities (r, g, 1)
    s_b = torch.tensor([0.0, 0.0, 1.0], device=device)
    s_r = torch.tensor([1.0, 0.0, 1.0], device=device)
    s_g = torch.tensor([0.0, 1.0, 1.0], device=device)
    s_gray = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0], device=device)

    # Target = source + offset
    t_b = torch.stack([s_b[0] + bd[0], s_b[1] + bd[1], torch.ones_like(bd[0])])
    t_r = torch.stack([s_r[0] + rd[0], s_r[1] + rd[1], torch.ones_like(rd[0])])
    t_g = torch.stack([s_g[0] + gd[0], s_g[1] + gd[1], torch.ones_like(gd[0])])
    t_gray = torch.stack([s_gray[0] + nd[0], s_gray[1] + nd[1], torch.ones_like(nd[0])])

    T = torch.stack([t_b, t_r, t_g], dim=1)  # [3, 3]

    # Skew-symmetric matrix of t_gray
    skew = torch.stack([
        torch.stack([torch.zeros_like(t_gray[0]), -t_gray[2], t_gray[1]]),
        torch.stack([t_gray[2], torch.zeros_like(t_gray[0]), -t_gray[0]]),
        torch.stack([-t_gray[1], t_gray[0], torch.zeros_like(t_gray[0])]),
    ])

    M = skew @ T

    # Nullspace vector lambda via cross of two independent rows
    r0, r1, r2 = M[0], M[1], M[2]
    lam01 = torch.linalg.cross(r0, r1)
    lam02 = torch.linalg.cross(r0, r2)
    lam12 = torch.linalg.cross(r1, r2)

    n01 = (lam01 * lam01).sum()
    n02 = (lam02 * lam02).sum()
    n12 = (lam12 * lam12).sum()

    lam = torch.where(n01 >= n02,
                      torch.where(n01 >= n12, lam01, lam12),
                      torch.where(n02 >= n12, lam02, lam12))

    S_inv = torch.tensor([[-1.0, -1.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], device=device)

    D = torch.diag(lam)
    H = T @ D @ S_inv
    H = H / (H[2, 2] + 1e-10)
    return H


# -- Pipeline stages ----------------------------------------------------------

def apply_exposure(rgb: Tensor, exposure_params: Tensor, frame_idx: int) -> Tensor:
    """Stage 1: Exposure compensation.  rgb * 2^offset."""
    return rgb * torch.pow(torch.tensor(2.0, device=rgb.device), exposure_params[frame_idx])


def apply_vignetting(
    rgb: Tensor,
    vignetting_params: Tensor,
    pixel_coords: Tensor,
    resolution_w: int,
    resolution_h: int,
    camera_idx: int,
) -> Tensor:
    """Stage 2: Polynomial radial vignetting falloff."""
    device = rgb.device
    res_f = torch.tensor([resolution_w, resolution_h], device=device, dtype=rgb.dtype)
    uv = (pixel_coords - res_f * 0.5) / res_f.max()

    channels = []
    for ch in range(3):
        vig = vignetting_params[camera_idx, ch]  # [5]
        center = vig[:2]
        alphas = vig[2:]
        delta = uv - center.unsqueeze(0)
        r2 = (delta * delta).sum(dim=-1)
        falloff = torch.ones_like(r2)
        r2_pow = r2.clone()
        for alpha in alphas:
            falloff = falloff + alpha * r2_pow
            r2_pow = r2_pow * r2
        falloff = falloff.clamp(0.0, 1.0)
        channels.append(rgb[:, ch] * falloff)
    return torch.stack(channels, dim=-1)


def apply_color_correction(
    rgb: Tensor,
    color_params: Tensor,
    frame_idx: int,
) -> Tensor:
    """Stage 3: Homography-based color correction in RGI space."""
    H = _get_homography(color_params, frame_idx)  # [3, 3]
    intensity = rgb.sum(dim=-1, keepdim=True)
    rgi = torch.cat([rgb[:, 0:1], rgb[:, 1:2], intensity], dim=-1)
    rgi = (H @ rgi.T).T
    rgi = rgi * (intensity / (rgi[:, 2:3] + 1e-5))
    r_out = rgi[:, 0]
    g_out = rgi[:, 1]
    b_out = rgi[:, 2] - r_out - g_out
    return torch.stack([r_out, g_out, b_out], dim=-1)


def apply_crf(
    rgb: Tensor,
    crf_params: Tensor,
    camera_idx: int,
) -> Tensor:
    """Stage 4: Parametric toe-shoulder CRF."""
    rgb = rgb.clamp(0.0, 1.0)
    eps = 1e-6
    channels = []
    for ch in range(3):
        crf = crf_params[camera_idx, ch]  # [4]: toe_raw, shoulder_raw, gamma_raw, center_raw
        toe = 0.3 + F.softplus(crf[0])
        shoulder = 0.3 + F.softplus(crf[1])
        gamma = 0.1 + F.softplus(crf[2])
        center = torch.sigmoid(crf[3])

        lerp_val = toe + center * (shoulder - toe)
        a = (shoulder * center) / lerp_val
        b = 1.0 - a

        x = rgb[:, ch]
        mask_low = x <= center
        y_low = a * torch.pow((x / center).clamp(min=eps), toe)
        y_high = 1.0 - b * torch.pow(((1.0 - x) / (1.0 - center)).clamp(min=eps), shoulder)
        y = torch.where(mask_low, y_low, y_high)
        channels.append(torch.pow(y.clamp(min=eps), gamma))
    return torch.stack(channels, dim=-1)


# -- Full pipeline wrapper ----------------------------------------------------

def ppisp_apply_torch(
    exposure_params: Tensor,
    vignetting_params: Tensor,
    color_params: Tensor,
    crf_params: Tensor,
    rgb_in: Tensor,
    pixel_coords: Tensor,
    resolution_w: int,
    resolution_h: int,
    camera_idx: int,
    frame_idx: int,
) -> Tensor:
    """Apply the full 4-stage PPISP pipeline (PyTorch reference).

    Args:
        exposure_params: [num_frames]
        vignetting_params: [num_cameras, 3, 5]
        color_params: [num_frames, 8]
        crf_params: [num_cameras, 3, 4]
        rgb_in: [N, 3]
        pixel_coords: [N, 2] pixel coordinates
        resolution_w: image width
        resolution_h: image height
        camera_idx: camera index (-1 to skip per-camera stages)
        frame_idx: frame index (-1 to skip per-frame stages)

    Returns:
        Processed RGB [N, 3]
    """
    rgb = rgb_in.clone()

    if frame_idx != -1:
        rgb = apply_exposure(rgb, exposure_params, frame_idx)

    if camera_idx != -1:
        rgb = apply_vignetting(rgb, vignetting_params, pixel_coords,
                               resolution_w, resolution_h, camera_idx)

    if frame_idx != -1:
        rgb = apply_color_correction(rgb, color_params, frame_idx)

    if camera_idx != -1:
        rgb = apply_crf(rgb, crf_params, camera_idx)

    return rgb


# -- nn.Module wrapper for benchmark convenience ------------------------------

class PPISPPyTorch(nn.Module):
    """Complete 4-stage PPISP pipeline as nn.Module (matching OSS ppisp API)."""

    def __init__(self, num_cameras: int, num_frames: int,
                 resolution_w: int = 1920, resolution_h: int = 1080,
                 device: torch.device | str = "cuda"):
        super().__init__()
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h

        self.exposure_params = nn.Parameter(torch.zeros(num_frames, device=device))
        self.vignetting_params = nn.Parameter(
            torch.zeros(num_cameras, 3, 2 + NUM_VIGNETTING_ALPHA_TERMS, device=device))

        self.color_params = nn.Parameter(
            torch.zeros(num_frames, 8, device=device))

        # CRF init matching OSS: softplus_inverse(1.0 - min_value) for toe/shoulder/gamma
        def _sp_inv(x: float, min_val: float) -> float:
            return float(torch.log(torch.expm1(torch.tensor(max(1e-5, x - min_val)))))

        crf_raw = torch.zeros(CRF_PARAMS_PER_CHANNEL, device=device)
        crf_raw[0] = _sp_inv(1.0, 0.3)   # toe
        crf_raw[1] = _sp_inv(1.0, 0.3)   # shoulder
        crf_raw[2] = _sp_inv(1.0, 0.1)   # gamma
        crf_raw[3] = 0.0                  # center -> sigmoid(0) = 0.5
        self.crf_params = nn.Parameter(
            crf_raw.view(1, 1, CRF_PARAMS_PER_CHANNEL)
            .repeat(num_cameras, 3, 1).contiguous())

    def forward(
        self,
        rgb: Tensor,
        pixel_coords: Tensor,
        camera_idx: int = 0,
        frame_idx: int = 0,
    ) -> Tensor:
        return ppisp_apply_torch(
            self.exposure_params, self.vignetting_params,
            self.color_params, self.crf_params,
            rgb, pixel_coords,
            self.resolution_w, self.resolution_h,
            camera_idx, frame_idx,
        )
