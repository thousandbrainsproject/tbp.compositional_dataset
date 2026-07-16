from __future__ import annotations

from collections.abc import Sequence

import pytest

from tbp.compositional_datasets.sticker_disk import (
    DEFAULT_DIAMETER,
    DEFAULT_SEGMENTS,
    DEFAULT_THICKNESS,
    build_sticker_disk_mesh,
)


def _triangle_normal_z(
    vertices: Sequence[tuple[float, float, float]],
    face: tuple[int, ...],
) -> float:
    """Return the unnormalized Z component of a triangle normal.

    Args:
        vertices: Object-space mesh vertices.
        face: Triangle vertex indices.

    Returns:
        Signed Z component of the face normal.
    """
    a, b, c = (vertices[index] for index in face)
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def test_default_disk_dimensions_and_counts() -> None:
    """Default mesh uses the approved dimensions and deterministic counts."""
    mesh = build_sticker_disk_mesh()

    xs = [vertex[0] for vertex in mesh.vertices]
    ys = [vertex[1] for vertex in mesh.vertices]
    zs = [vertex[2] for vertex in mesh.vertices]
    assert DEFAULT_DIAMETER == 0.02
    assert DEFAULT_THICKNESS == 0.001
    assert DEFAULT_SEGMENTS == 64
    assert max(xs) - min(xs) == pytest.approx(0.02)
    assert max(ys) - min(ys) == pytest.approx(0.02)
    assert max(zs) - min(zs) == pytest.approx(0.001)
    assert len(mesh.vertices) == 2 * 64 + 2
    assert len(mesh.faces) == 3 * 64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"diameter": 0.0}, "diameter must be positive"),
        ({"thickness": -0.001}, "thickness must be positive"),
        ({"segments": 2}, "segments must be at least 3"),
    ],
)
def test_disk_rejects_invalid_parameters(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    """Invalid geometry inputs fail before mesh construction.

    Args:
        kwargs: Invalid keyword argument for the mesh builder.
        message: Expected validation error text.
    """
    with pytest.raises(ValueError, match=message):
        build_sticker_disk_mesh(**kwargs)


def test_caps_face_outward() -> None:
    """Front and back triangle winding points away from the disk center."""
    mesh = build_sticker_disk_mesh(segments=8)

    assert all(_triangle_normal_z(mesh.vertices, face) > 0.0 for face in mesh.faces[:8])
    assert all(_triangle_normal_z(mesh.vertices, face) < 0.0 for face in mesh.faces[8:16])


def test_face_uvs_match_loops_and_stay_in_unit_square() -> None:
    """Every polygon loop gets one bounded UV coordinate."""
    mesh = build_sticker_disk_mesh(segments=8)

    assert [len(uvs) for uvs in mesh.face_uvs] == [len(face) for face in mesh.faces]
    assert all(
        0.0 <= value <= 1.0
        for face_uvs in mesh.face_uvs
        for uv in face_uvs
        for value in uv
    )


def test_back_cap_mirrors_u_for_an_upright_direct_view() -> None:
    """Back artwork is not horizontally reversed when viewed from -Z."""
    mesh = build_sticker_disk_mesh(segments=8)

    assert mesh.face_uvs[0][1] == pytest.approx((1.0, 0.5))
    assert mesh.face_uvs[8][2] == pytest.approx((0.0, 0.5))


def test_caps_and_side_use_separate_materials() -> None:
    """Both caps use artwork while only wall quads use white."""
    mesh = build_sticker_disk_mesh(segments=8)

    assert mesh.material_indices == (0,) * 16 + (1,) * 8
