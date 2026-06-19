from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageOps

DEFAULT_STICKER_SIZE = 1024
DEFAULT_PADDING_RATIO = 0.20
DEFAULT_PRIMITIVE_PADDING_RATIO = 0.34
DEFAULT_BACKGROUND_COLOR = (255, 255, 255, 255)
DEFAULT_SHAPE_COLOR = (47, 43, 92, 255)
PRIMITIVE_SHAPES = ("square", "triangle", "circle", "star", "heart")
ANTIALIAS_SCALE = 4
SQUARE_EXTRA_INSET_RATIO = 0.040
TRIANGLE_Y_OFFSET_RATIO = -0.045
HEART_Y_OFFSET_RATIO = 0.030


def _rgba_color(value):
    """Return a color value as an RGBA tuple.

    Args:
        value: Color string or RGB/RGBA tuple accepted by Pillow.

    Returns:
        Four-channel 8-bit RGBA tuple.
    """
    rgba = ImageColor.getcolor(value, "RGBA") if isinstance(value, str) else value
    if len(rgba) == 3:
        return (rgba[0], rgba[1], rgba[2], 255)
    return (rgba[0], rgba[1], rgba[2], rgba[3])


def _scaled_box(size: int, padding_ratio: float, scale: int) -> tuple[int, int, int, int]:
    """Return the padded content box in a scaled coordinate system.

    Args:
        size: Final sticker size in pixels.
        padding_ratio: Fraction of the final canvas reserved as padding.
        scale: Antialiasing scale factor.

    Returns:
        Inclusive Pillow drawing box in scaled pixels.
    """
    inset = int(round(size * padding_ratio * scale / 2.0))
    edge = size * scale - 1 - inset
    return (inset, inset, edge, edge)


def _inset_box(
    box: tuple[int, int, int, int],
    inset: int,
) -> tuple[int, int, int, int]:
    """Return a box inset by the same amount on every side.

    Args:
        box: Inclusive drawing bounds.
        inset: Pixel inset to apply.

    Returns:
        Inset inclusive drawing bounds.
    """
    left, top, right, bottom = box
    return (left + inset, top + inset, right - inset, bottom - inset)


def _offset_box(
    box: tuple[int, int, int, int],
    x_offset: int = 0,
    y_offset: int = 0,
) -> tuple[int, int, int, int]:
    """Return a box shifted by an x and y offset.

    Args:
        box: Inclusive drawing bounds.
        x_offset: Horizontal offset in pixels.
        y_offset: Vertical offset in pixels.

    Returns:
        Shifted inclusive drawing bounds.
    """
    left, top, right, bottom = box
    return (
        left + x_offset,
        top + y_offset,
        right + x_offset,
        bottom + y_offset,
    )


def _new_sticker_canvas(
    size: int = DEFAULT_STICKER_SIZE,
    background_color: str | tuple[int, int, int] | tuple[int, int, int, int] = DEFAULT_BACKGROUND_COLOR,
) -> Image.Image:
    """Create a transparent square canvas with a white circular background.

    Args:
        size: Square canvas size in pixels.
        background_color: Fill color for the circular sticker backing.

    Returns:
        RGBA Pillow image with transparent corners.
    """
    scaled_size = size * ANTIALIAS_SCALE
    image = Image.new("RGBA", (scaled_size, scaled_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (0, 0, scaled_size - 1, scaled_size - 1),
        fill=_rgba_color(background_color),
    )
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _star_points(box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """Return a five-point star polygon inside a box.

    Args:
        box: Inclusive drawing bounds.

    Returns:
        Polygon points alternating outer and inner radii.
    """
    left, top, right, bottom = box
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    outer_radius = min(right - left, bottom - top) / 2.0
    inner_radius = outer_radius * 0.45
    points = []
    for index in range(10):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        points.append(
            (
                center_x + math.cos(angle) * radius,
                center_y + math.sin(angle) * radius,
            )
        )
    return points


def _heart_points(box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """Return a heart polygon inside a box.

    Args:
        box: Inclusive drawing bounds.

    Returns:
        Polygon points approximating a centered heart.
    """
    left, top, right, bottom = box
    raw_points = []
    for index in range(160):
        t = 2.0 * math.pi * index / 160.0
        x = 16.0 * math.sin(t) ** 3
        y = (
            13.0 * math.cos(t)
            - 5.0 * math.cos(2.0 * t)
            - 2.0 * math.cos(3.0 * t)
            - math.cos(4.0 * t)
        )
        raw_points.append((x, -y))
    min_x = min(point[0] for point in raw_points)
    max_x = max(point[0] for point in raw_points)
    min_y = min(point[1] for point in raw_points)
    max_y = max(point[1] for point in raw_points)
    width = right - left
    height = bottom - top
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    offset_x = (left + right - (min_x + max_x) * scale) / 2.0
    offset_y = (top + bottom - (min_y + max_y) * scale) / 2.0
    return [(offset_x + x * scale, offset_y + y * scale) for x, y in raw_points]


def _draw_primitive(
    draw: ImageDraw.ImageDraw,
    shape: str,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
) -> None:
    """Draw one primitive shape into a content box.

    Args:
        draw: Pillow drawing context.
        shape: Primitive shape name.
        box: Inclusive drawing bounds.
        fill: Shape fill color.

    Raises:
        ValueError: If `shape` is not supported.
    """
    box_size = max(1, box[2] - box[0] + 1)
    left, top, right, bottom = box
    if shape == "square":
        box = _inset_box(box, int(round(box_size * SQUARE_EXTRA_INSET_RATIO)))
        draw.rectangle(box, fill=fill)
    elif shape == "triangle":
        box = _offset_box(box, y_offset=int(round(box_size * TRIANGLE_Y_OFFSET_RATIO)))
        left, top, right, bottom = box
        draw.polygon(
            [
                ((left + right) / 2.0, top),
                (right, bottom),
                (left, bottom),
            ],
            fill=fill,
        )
    elif shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "star":
        draw.polygon(_star_points(box), fill=fill)
    elif shape == "heart":
        box = _offset_box(box, y_offset=int(round(box_size * HEART_Y_OFFSET_RATIO)))
        draw.polygon(_heart_points(box), fill=fill)
    else:
        raise ValueError(f"unsupported primitive shape: {shape}")


def create_primitive_sticker(
    shape: str,
    *,
    size: int = DEFAULT_STICKER_SIZE,
    padding_ratio: float = DEFAULT_PRIMITIVE_PADDING_RATIO,
    background_color: str | tuple[int, int, int] | tuple[int, int, int, int] = DEFAULT_BACKGROUND_COLOR,
    shape_color: str | tuple[int, int, int] | tuple[int, int, int, int] = DEFAULT_SHAPE_COLOR,
) -> Image.Image:
    """Create one primitive sticker image.

    Args:
        shape: Primitive shape name.
        size: Square output size in pixels.
        padding_ratio: Fraction of the canvas reserved around the primitive.
        background_color: Circular backing color.
        shape_color: Primitive fill color.

    Returns:
        RGBA sticker image.
    """
    scaled_size = size * ANTIALIAS_SCALE
    image = _new_sticker_canvas(size, background_color).resize(
        (scaled_size, scaled_size),
        Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image)
    _draw_primitive(
        draw,
        shape,
        _scaled_box(size, padding_ratio, ANTIALIAS_SCALE),
        _rgba_color(shape_color),
    )
    return image.resize((size, size), Image.Resampling.LANCZOS)


def create_sticker_from_image(
    source: Image.Image,
    *,
    size: int = DEFAULT_STICKER_SIZE,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    background_color: str | tuple[int, int, int] | tuple[int, int, int, int] = DEFAULT_BACKGROUND_COLOR,
    preserve_canvas: bool = False,
) -> Image.Image:
    """Create a circular sticker from an existing transparent image.

    Args:
        source: Source artwork image.
        size: Square output size in pixels.
        padding_ratio: Fraction of the canvas reserved around the artwork.
        background_color: Circular backing color.
        preserve_canvas: Whether to preserve the full source canvas instead of
            cropping to non-transparent alpha bounds.

    Returns:
        RGBA sticker image.

    Raises:
        ValueError: If the source image has no visible pixels.
    """
    artwork = source.convert("RGBA")
    if not preserve_canvas:
        alpha_bounds = artwork.getchannel("A").getbbox()
        if alpha_bounds is None:
            raise ValueError("source image has no visible pixels")
        artwork = artwork.crop(alpha_bounds)
    sticker = _new_sticker_canvas(size, background_color)
    content_size = max(1, int(round(size * (1.0 - padding_ratio))))
    artwork = ImageOps.contain(
        artwork,
        (content_size, content_size),
        method=Image.Resampling.LANCZOS,
    )
    left = (size - artwork.width) // 2
    top = (size - artwork.height) // 2
    sticker.alpha_composite(artwork, (left, top))
    return sticker


def save_png(image: Image.Image, path: Path) -> Path:
    """Save an image as a PNG, creating parent directories as needed.

    Args:
        image: Image to save.
        path: Destination PNG path.

    Returns:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def generate_primitive_stickers(
    out_dir: Path,
    *,
    size: int = DEFAULT_STICKER_SIZE,
    padding_ratio: float = DEFAULT_PRIMITIVE_PADDING_RATIO,
    background_color: str | tuple[int, int, int] | tuple[int, int, int, int] = DEFAULT_BACKGROUND_COLOR,
    shape_color: str | tuple[int, int, int] | tuple[int, int, int, int] = DEFAULT_SHAPE_COLOR,
) -> list[Path]:
    """Generate the full primitive sticker set.

    Args:
        out_dir: Directory where primitive PNG files are written.
        size: Square output size in pixels.
        padding_ratio: Fraction of the canvas reserved around each primitive.
        background_color: Circular backing color.
        shape_color: Primitive fill color.

    Returns:
        Paths to the written PNG files.
    """
    outputs = []
    for shape in PRIMITIVE_SHAPES:
        image = create_primitive_sticker(
            shape,
            size=size,
            padding_ratio=padding_ratio,
            background_color=background_color,
            shape_color=shape_color,
        )
        outputs.append(save_png(image, out_dir / f"{shape}.png"))
    return outputs
