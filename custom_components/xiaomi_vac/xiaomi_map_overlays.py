"""Carpet and travelled-path overlays for Xiaomi JSON maps."""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable

from PIL import Image, ImageDraw

_LOGGER = logging.getLogger(__name__)


def parse_carpets(unpacked: str | bytes) -> list[list[float]]:
    """Extract carpet quadrilaterals from a Xiaomi JSON map."""
    try:
        carpets = json.loads(unpacked).get("carpets") or []
    except (AttributeError, TypeError, UnicodeDecodeError, ValueError):
        return []
    return [
        carpet["p"]
        for carpet in carpets
        if isinstance(carpet, dict)
        and isinstance(carpet.get("p"), list)
        and len(carpet["p"]) == 8
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in carpet["p"]
        )
    ]


def parse_path(unpacked: str | bytes) -> list[list[tuple[float, float]]]:
    """Extract independent travelled-path segments from a Xiaomi JSON map."""
    try:
        payload = json.loads(unpacked)
        encoded = payload.get("paths", {}).get("points") or "[]"
        points = json.loads(encoded) if isinstance(encoded, str) else encoded
    except (AttributeError, TypeError, UnicodeDecodeError, ValueError):
        return []
    if not isinstance(points, list):
        return []

    runs: list[list[dict]] = []
    current: list[dict] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        x, y = point.get("x"), point.get("y")
        if (
            not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(x)
            or not math.isfinite(y)
        ):
            continue
        if x == 0 and y == 0:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(point)
    if current:
        runs.append(current)
    if not runs:
        return []

    segments: list[list[tuple[float, float]]] = []
    segment: list[tuple[float, float]] = []
    for point in max(runs, key=len):
        if point.get("type") == 0 and segment:
            if len(segment) >= 2:
                segments.append(segment)
            segment = []
        segment.append((point["x"], point["y"]))
    if len(segment) >= 2:
        segments.append(segment)
    return segments


def calibration_transform(
    calibration: list[dict],
) -> Callable[[float, float], tuple[float, float]] | None:
    """Return a vacuum-coordinate to image-pixel transform."""
    if len(calibration) < 3:
        return None
    p0, p1, p2 = calibration[:3]
    try:
        dx = p1["vacuum"]["x"] - p0["vacuum"]["x"]
        dy = p2["vacuum"]["y"] - p0["vacuum"]["y"]
        if not dx or not dy:
            return None
        scale_x = (p1["map"]["x"] - p0["map"]["x"]) / dx
        scale_y = (p2["map"]["y"] - p0["map"]["y"]) / dy
        offset_x = p0["map"]["x"] - p0["vacuum"]["x"] * scale_x
        offset_y = p0["map"]["y"] - p0["vacuum"]["y"] * scale_y
    except (KeyError, TypeError):
        return None
    return lambda x, y: (offset_x + x * scale_x, offset_y + y * scale_y)


def draw_vacuum_dot(
    image: Image.Image,
    vac_x: float,
    vac_y: float,
    angle: float | None,
    calibration: list[dict],
    offset_x: int,
    offset_y: int,
) -> Image.Image:
    """Draw the vacuum position dot on a cropped map image.

    Mirrors vacuum_map_parser_base's ImageGenerator._draw_vacuum (same sizes,
    colours and geometry) so a dot stamped by the fast P-frame path is
    pixel-identical to one the parser bakes in during a full render. The
    parser no longer draws it (VACUUM_POSITION was removed from _DRAWABLES) —
    this is now the only dot painter, used by both render paths.

    ``vacuum_map_parser_base`` is imported lazily: this module must stay
    importable for brands whose maps carry no vacuum position.
    """
    from vacuum_map_parser_base.config.color import ColorsPalette, SupportedColor
    from vacuum_map_parser_base.config.size import Size, Sizes

    transform = calibration_transform(calibration)
    if transform is None:
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    px, py = transform(vac_x, vac_y)
    px -= offset_x
    py -= offset_y
    r = Sizes().get_size(Size.VACUUM_RADIUS)
    outline = ColorsPalette().get_color(SupportedColor.ROBO_OUTLINE)
    fill = ColorsPalette().get_color(SupportedColor.ROBO)
    use_layer = len(outline) > 3 or len(fill) > 3

    def draw_func(draw: ImageDraw.ImageDraw) -> None:
        a = angle if angle is not None else 0
        r_scaled = r / 16
        # main outline
        draw.ellipse([px - r, py - r, px + r, py + r], outline=outline, fill=fill)
        if r >= 8:
            # secondary outline
            r2 = r_scaled * 14
            draw.ellipse([px - r2, py - r2, px + r2, py + r2], outline=outline)
        # bin cover
        a1 = (a + 104) / 180 * math.pi
        a2 = (a - 104) / 180 * math.pi
        r2 = r_scaled * 13
        draw.line(
            [px - r2 * math.cos(a1), py + r2 * math.sin(a1),
             px - r2 * math.cos(a2), py + r2 * math.sin(a2)],
            width=1, fill=outline,
        )
        # lidar
        ang = a / 180 * math.pi
        r2 = r_scaled * 3
        x = px + r2 * math.cos(ang)
        y = py - r2 * math.sin(ang)
        r2 = r_scaled * 4
        draw.ellipse([x - r2, y - r2, x + r2, y + r2], outline=outline, fill=fill)
        # button
        half_color = tuple(
            (outline[i] + fill[i]) // 2 for i in range(3)
        )
        r2 = r_scaled * 10
        x = px + r2 * math.cos(ang)
        y = py - r2 * math.sin(ang)
        r2 = r_scaled * 2
        draw.ellipse([x - r2, y - r2, x + r2, y + r2], outline=half_color, fill=half_color)

    if use_layer:
        # Same compositing as the parser's _draw_on_new_layer for RGBA colours.
        layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw_func(ImageDraw.Draw(layer, "RGBA"))
        return Image.alpha_composite(image, layer)
    draw_func(ImageDraw.Draw(image, "RGBA"))
    return image


def draw_overlays(
    image: Image.Image,
    carpets: list[list[float]],
    path_segments: list[list[tuple[float, float]]],
    calibration: list[dict],
    offset_x: int,
    offset_y: int,
) -> Image.Image:
    """Draw Xiaomi-only overlays without changing the base renderer."""
    transform = calibration_transform(calibration)
    if transform is None or (not carpets and not path_segments):
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    for carpet in carpets:
        try:
            points = [transform(carpet[i], carpet[i + 1]) for i in range(0, 8, 2)]
            points = [(x - offset_x, y - offset_y) for x, y in points]
            draw.polygon(
                points,
                fill=(210, 170, 90, 70),
                outline=(180, 138, 60, 225),
                width=2,
            )
        except (IndexError, OverflowError, TypeError, ValueError):
            _LOGGER.debug("Could not draw Xiaomi carpet", exc_info=True)
    for segment in path_segments:
        try:
            points = [transform(x, y) for x, y in segment]
            points = [(x - offset_x, y - offset_y) for x, y in points]
            draw.line(points, fill=(64, 200, 255, 210), width=2, joint="curve")
        except (OverflowError, TypeError, ValueError):
            _LOGGER.debug("Could not draw Xiaomi path segment", exc_info=True)
    return image
