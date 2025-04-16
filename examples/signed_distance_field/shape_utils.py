# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np


def draw_circle(image: np.ndarray,
                center_x: float,
                center_y: float,
                radius: float,
                color: list[int],
                antialiased: bool = False):
    """Draw a filled circle"""
    height, width = image.shape[:2]
    y, x = np.ogrid[-center_y:height - center_y, -center_x:width - center_x]

    if antialiased:
        # Not strictly correct coverage, but this is just an example
        dist = np.sqrt(x * x + y * y) - radius
        coverage = np.clip(1.0 - dist, 0.0, 1.0)
        for i in range(4):
            current = image[:, :, i]
            new_coverage = np.minimum(current + coverage, 1.0)
            image[:, :, i] = np.where(dist <= 1.0, color[i] * new_coverage,
                                      image[:, :, i])
    else:
        mask = x * x + y * y <= radius * radius
        image[mask] = color


def draw_line(image: np.ndarray,
              x1: float,
              y1: float,
              x2: float,
              y2: float,
              thickness: float,
              color: list[int],
              antialiased: bool = False):
    """Draw a thick line"""
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx * dx + dy * dy)
    if line_length == 0:
        return
    nx = dx / line_length
    ny = dy / line_length
    px = x - x1
    py = y - y1
    proj = (px * nx + py * ny)
    proj = np.clip(proj, 0, line_length)
    closest_x = x1 + proj * nx
    closest_y = y1 + proj * ny
    dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
    if antialiased:
        # Calculate coverage using smooth falloff
        # Half thickness plus 1 pixel for antialiasing
        coverage = np.clip(thickness / 2 + 1 - dist, 0.0, 1.0)
        for i in range(4):
            current = image[:, :, i] / color[i] if color[i] != 0 else 0
            new_coverage = np.minimum(current + coverage, 1.0)
            image[:, :, i] = np.where(dist <= thickness / 2 + 1,
                                      color[i] * new_coverage, image[:, :, i])
    else:
        mask = dist <= thickness / 2
        image[mask] = color


def draw_rotated_rect(image: np.ndarray,
                      center_x: float,
                      center_y: float,
                      width: float,
                      height: float,
                      angle: float,
                      thickness: float,
                      color: list[int],
                      antialiased: bool = False):
    """Draw a non-filled rectangle"""
    angle = np.radians(angle)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    half_w = width / 2
    half_h = height / 2
    corners = np.array([[-half_w, -half_h], [half_w, -half_h],
                        [half_w, half_h], [-half_w, half_h]])

    rotated = np.zeros_like(corners)
    for i in range(4):
        rotated[i,
                0] = corners[i, 0] * cos_a - corners[i, 1] * sin_a + center_x
        rotated[i,
                1] = corners[i, 0] * sin_a + corners[i, 1] * cos_a + center_y

    for i in range(4):
        j = (i + 1) % 4
        draw_line(image, rotated[i, 0], rotated[i, 1], rotated[j, 0],
                  rotated[j, 1], thickness, color, antialiased)
