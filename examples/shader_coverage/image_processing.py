# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Image generation, GPU processing, and numerical validation; no coverage logic."""

from pathlib import Path
import numpy as np
import slangpy as spy

SOURCE = Path(__file__).with_name("postprocess.slang").resolve()
SOURCES = tuple(sorted(SOURCE.parent.glob("*.slang")))
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def make_inputs() -> list[tuple[str, np.ndarray, bool, int]]:
    """Generate reproducible edges, gradients, and noise without an image asset."""
    y, x = np.mgrid[:240, :320].astype(np.float32)
    x /= 319
    y /= 239
    rgb = np.stack((0.1 + 0.5 * x, 0.15 + 0.4 * y, 0.65 - 0.4 * x), axis=-1)
    rgb[(x - 0.65) ** 2 + (y - 0.35) ** 2 < 0.18**2] = [0.9, 0.25, 0.08]
    rgb[(x < 0.3) & (y > 0.35) & (y < 0.8)] = [0.05, 0.15, 0.75]
    noise = np.random.default_rng(739).normal(0, 0.025, rgb.shape).astype(np.float32)
    rgba = np.ones((240, 320, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb + noise, 0.02, 0.95)
    stress = rgba.copy()
    stress[..., :3] *= 12.0
    stress[30:100, 20:100, :3] = 0
    stress[140:220, 220:300, 3] = 0
    return [
        ("ordinary", rgba, True, 0),
        ("hdr_alpha", stress, True, 0),
        ("filter_off", stress, False, 1),
    ]


def reference(image: np.ndarray, denoise: bool, mapper_id: int = 0) -> np.ndarray:
    """Independent filter, analytic exposure gradient, and tone-mapping reference."""
    color = image[..., :3].copy()
    height, width = color.shape[:2]
    if denoise:
        light = color @ LUMA
        total = np.zeros_like(color)
        weights = np.zeros((height, width), dtype=np.float32)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y = slice(max(0, -dy), min(height, height - dy))
                x = slice(max(0, -dx), min(width, width - dx))
                neighbor = color[
                    max(0, dy) : min(height, height + dy), max(0, dx) : min(width, width + dx)
                ]
                accepted = np.abs(neighbor @ LUMA - light[y, x]) <= 0.12
                weight = np.float32(1.0 / (1 + dx * dx + dy * dy))
                total[y, x] += neighbor * (accepted * weight)[..., None]
                weights[y, x] += accepted * weight
        color = total / weights[..., None]
    light = color @ LUMA
    # Analytic derivative of 0.5 * (exposure * light - 0.45)**2 at exposure=1.
    gradient = (light - np.float32(0.45)) * light
    fitted_exposure = np.clip(1.0 - np.float32(0.25) * gradient, 0.25, 4.0)
    color *= fitted_exposure[..., None]
    if mapper_id == 0:
        light = color @ LUMA
        color = color / np.maximum(light, 1.0)[..., None]
        color = np.sqrt(np.clip(color, 0, 1))
        color[light <= 1e-6] = 0
    else:
        color = np.sqrt(np.clip(color / (1.0 + color), 0, 1))
    output = image.copy()
    output[..., :3] = color
    output[image[..., 3] == 0] = image[image[..., 3] == 0]
    return output


class ImageProcessor:
    """Run the image shader on the supplied device, independently of instrumentation."""

    def __init__(self, device: spy.Device) -> None:
        self.device = device
        self.module = spy.Module.load_from_file(device, str(SOURCE))
        self.function = self.module.denoiseToneMap.type_conformances(
            [
                spy.TypeConformance("IToneMapper", "NormalizeToneMapper", 0),
                spy.TypeConformance("IToneMapper", "ReinhardToneMapper", 1),
            ]
        )

    def process(self, image: np.ndarray, denoise: bool = True, mapper_id: int = 0) -> np.ndarray:
        if mapper_id not in (0, 1):
            raise ValueError("mapper_id must be 0 (normalize) or 1 (Reinhard)")
        height, width = image.shape[:2]
        texture = self.device.create_texture(
            width=width,
            height=height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource,
            data=image,
        )
        output = self.device.create_texture(
            width=width,
            height=height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )
        self.function(
            spy.grid((height, width)), texture, 1.0, 0.12, denoise, mapper_id, _result=output
        )
        return output.to_numpy()


def save_preview(path: Path, rgba: np.ndarray, linear: bool = False) -> None:
    rgb = np.clip(rgba[..., :3], 0, 1)
    if linear:
        rgb = rgb ** (1.0 / 2.2)
    y, x = np.indices(rgb.shape[:2])
    checker = np.where((x // 12 + y // 12) % 2, 0.65, 0.85)[..., None]
    alpha = rgba[..., 3:4]
    rgb = rgb * alpha + checker * (1 - alpha)
    spy.Bitmap(np.rint(rgb * 255).astype(np.uint8)).write(path)


def validate_outputs(
    device_type: spy.DeviceType,
    scenarios: list[tuple[str, np.ndarray, bool, int]],
    outputs: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compare outputs against NumPy and the same workload on an ordinary device."""
    results = {}
    with spy.Device(type=device_type, enable_hot_reload=False) as device:
        processor = ImageProcessor(device)
        for name, image, denoise, mapper_id in scenarios:
            expected = reference(image, denoise, mapper_id)
            plain = processor.process(image, denoise, mapper_id)
            np.testing.assert_allclose(outputs[name], expected, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(outputs[name], plain, rtol=2e-5, atol=2e-5)
            results[name] = {
                "numpy_max_error": float(np.max(np.abs(outputs[name] - expected))),
                "uninstrumented_max_error": float(np.max(np.abs(outputs[name] - plain))),
            }
    return results
