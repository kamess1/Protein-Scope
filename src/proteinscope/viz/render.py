"""Software renderer for space-filling molecular images.

A small z-buffered sphere rasteriser with Lambert shading and depth cueing, so that reports
contain real 3D views on any machine, with no PyMOL or OpenGL installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_rgb

from .style import SURFACE


@dataclass
class Camera:
    """Orthographic camera. `rotation` rows are the screen x, y and towards-viewer axes."""

    rotation: np.ndarray
    center: np.ndarray
    half_width: float


def look_along(view_direction: np.ndarray, center: np.ndarray, half_width: float) -> Camera:
    """Camera placed on the +view_direction side of `center`, looking back at it."""
    z = np.asarray(view_direction, dtype=float)
    z = z / max(np.linalg.norm(z), 1e-9)
    helper = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Camera(np.vstack([x, y, z]), np.asarray(center, dtype=float), half_width)


@dataclass
class Layer:
    rgb: np.ndarray
    depth: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.depth)


def render_spheres(
    coords: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    camera: Camera,
    size: int = 900,
    clip_front: float | None = None,
) -> Layer:
    """Rasterise spheres back to front through a z-buffer.

    clip_front drops spheres whose centers lie closer to the viewer than that plane, measured
    along the view axis from the camera center. It gives cut-away views into buried pockets.
    """
    camera_coords = (coords - camera.center) @ camera.rotation.T
    scale = size / (2.0 * camera.half_width)
    pixel_x = (camera_coords[:, 0] + camera.half_width) * scale
    pixel_y = (camera.half_width - camera_coords[:, 1]) * scale

    rgb = np.ones((size, size, 3)) * np.array(to_rgb(SURFACE))
    depth = np.full((size, size), -np.inf)
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)

    depth_range = max(np.ptp(camera_coords[:, 2]), 1.0)
    depth_min = camera_coords[:, 2].min()

    for index in np.argsort(camera_coords[:, 2]):
        if clip_front is not None and camera_coords[index, 2] > clip_front:
            continue
        radius_px = radii[index] * scale
        left = max(int(pixel_x[index] - radius_px), 0)
        right = min(int(pixel_x[index] + radius_px) + 2, size)
        top = max(int(pixel_y[index] - radius_px), 0)
        bottom = min(int(pixel_y[index] + radius_px) + 2, size)
        if left >= right or top >= bottom:
            continue
        ys, xs = np.mgrid[top:bottom, left:right]
        nx = (xs - pixel_x[index]) / radius_px
        ny = (pixel_y[index] - ys) / radius_px
        squared = nx**2 + ny**2
        inside = squared <= 1.0
        nz = np.sqrt(np.maximum(1.0 - squared, 0.0))
        z = camera_coords[index, 2] + nz * radii[index]
        visible = inside & (z > depth[top:bottom, left:right])
        if not visible.any():
            continue
        lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
        specular = np.clip(lambert, 0.0, 1.0) ** 24 * 0.12
        cue = 0.72 + 0.28 * np.clip((z - depth_min) / depth_range, 0.0, 1.0)
        shade = (0.45 + 0.55 * lambert) * cue
        window_rgb = rgb[top:bottom, left:right]
        window_rgb[visible] = np.clip(
            colors[index][None, :] * shade[visible][:, None] + specular[visible][:, None], 0.0, 1.0
        )
        depth[top:bottom, left:right][visible] = z[visible]
    return Layer(rgb, np.where(np.isneginf(depth), np.nan, depth))


def add_outlines(layer: Layer, strength: float = 0.55, threshold: float = 1.2) -> np.ndarray:
    """Darken pixels at depth discontinuities, which makes shape readable in flat colors."""
    depth = np.nan_to_num(layer.depth, nan=np.nanmin(layer.depth) - 20.0 if layer.mask.any() else 0.0)
    jump = np.zeros(depth.shape, dtype=bool)
    jump[:-1, :] |= np.abs(np.diff(depth, axis=0)) > threshold
    jump[:, :-1] |= np.abs(np.diff(depth, axis=1)) > threshold
    result = layer.rgb.copy()
    result[jump] *= 1.0 - strength
    return result


def composite_xray(base: Layer, overlay: Layer, hidden_alpha: float = 0.6) -> np.ndarray:
    """Draw overlay over base; parts hidden behind the base stay visible at reduced alpha."""
    image = add_outlines(base)
    mask = overlay.mask
    behind = mask & base.mask & (np.nan_to_num(overlay.depth, nan=-np.inf) < np.nan_to_num(base.depth, nan=-np.inf))
    alpha = np.where(behind, hidden_alpha, 1.0)[..., None]
    image[mask] = (overlay.rgb * alpha + image * (1 - alpha))[mask]
    return image


def diverging_colors(values: np.ndarray, limit: float, low: str, mid: str, high: str) -> np.ndarray:
    """Map values in [-limit, limit] through a two-hue scale with a neutral midpoint."""
    low_rgb, mid_rgb, high_rgb = (np.array(to_rgb(c)) for c in (low, mid, high))
    t = np.clip(values / limit, -1.0, 1.0)[:, None]
    return np.where(t < 0, mid_rgb + (low_rgb - mid_rgb) * -t, mid_rgb + (high_rgb - mid_rgb) * t)
