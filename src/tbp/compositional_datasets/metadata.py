from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tbp.compositional_datasets.placements import (
    SurfaceAttachment,
    SurfaceFrame,
    TextureStampResult,
)


def generation_metadata(
    *,
    parent_mesh_path: Path,
    sticker_path: Path,
    attachment: SurfaceAttachment,
    frame: SurfaceFrame,
    quad_corners: Any,
    texture_stamp: TextureStampResult | None = None,
) -> dict[str, Any]:
    """Build generation metadata for a placed sticker.

    Args:
        parent_mesh_path: Path to the input parent mesh.
        sticker_path: Path to the sticker image used for placement.
        attachment: Triangle attachment request for the sticker.
        frame: World-space frame computed for the attachment.
        quad_corners: Four world-space sticker corners.
        texture_stamp: Optional texture-stamp output details when the sticker
            was baked into a parent texture.

    Returns:
        A JSON-ready dictionary describing the input assets, triangle
        attachment, world-space placement frame, sticker quad, and optional
        texture-stamp result.
    """
    metadata = {
        "parent_mesh_path": str(parent_mesh_path),
        "sticker_path": str(sticker_path),
    }
    metadata.update(_sticker_fields(attachment, frame, quad_corners))
    if texture_stamp is not None:
        metadata["decal_mode"] = "texture_stamp"
        metadata.update(_texture_stamp_metadata(texture_stamp))
    return metadata


def _sticker_fields(
    attachment: SurfaceAttachment,
    frame: SurfaceFrame,
    quad_corners: Any,
) -> dict[str, Any]:
    """Build JSON-ready sticker placement metadata fields.

    Args:
        attachment: Triangle attachment request for the sticker.
        frame: World-space frame computed for the attachment.
        quad_corners: Four world-space sticker corners.

    Returns:
        JSON-ready attachment, frame, size, and quad-corner fields.
    """
    return {
        "triangle_id": attachment.triangle_id,
        "barycentric": np.asarray(attachment.barycentric, dtype=float).tolist(),
        "world_space_anchor_point": np.asarray(frame.anchor, dtype=float).tolist(),
        "surface_normal": np.asarray(frame.normal, dtype=float).tolist(),
        "tangent": np.asarray(frame.tangent, dtype=float).tolist(),
        "bitangent": np.asarray(frame.bitangent, dtype=float).tolist(),
        "rotated_tangent": np.asarray(frame.rotated_tangent, dtype=float).tolist(),
        "rotated_bitangent": np.asarray(frame.rotated_bitangent, dtype=float).tolist(),
        "rotation_about_normal_deg": attachment.rotation_about_normal_deg,
        "size": np.asarray(attachment.size, dtype=float).tolist(),
        "sticker_quad_corners": [np.asarray(corner, dtype=float).tolist() for corner in quad_corners],
    }


def _texture_stamp_metadata(texture_stamp: TextureStampResult) -> dict[str, Any]:
    """Build JSON-ready texture stamp metadata fields.

    Args:
        texture_stamp: Texture stamp result to serialize.

    Returns:
        JSON-ready texture stamp fields.
    """
    return {
        "texture_stamp_path": str(texture_stamp.texture_path),
        "texture_stamp_uv_center": np.asarray(texture_stamp.uv_center, dtype=float).tolist(),
        "texture_stamp_uv_corners": [
            np.asarray(corner, dtype=float).tolist() for corner in texture_stamp.uv_corners
        ],
        "texture_stamp_texture_size": np.asarray(texture_stamp.texture_size, dtype=int).tolist(),
        "texture_stamp_target_object_name": texture_stamp.target_object_name,
        "texture_stamp_target_material_index": int(texture_stamp.target_material_index),
        "texture_stamp_target_material_name": texture_stamp.target_material_name,
        "texture_stamp_target_uv_layer": texture_stamp.target_uv_layer,
        "texture_stamp_coverage_ratio": float(texture_stamp.coverage_ratio),
    }


def multi_generation_metadata(
    *,
    parent_mesh_path: Path,
    placements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build generation metadata for multiple texture-stamped stickers.

    Args:
        parent_mesh_path: Path to the input parent mesh.
        placements: Placement records containing slot name, side, sticker path,
            attachment, frame, quad corners, and texture stamp result.

    Returns:
        JSON-ready multi-sticker metadata.
    """
    stickers = []
    for placement in placements:
        sticker = {
            "slot_name": str(placement["slot_name"]),
            "side": str(placement["side"]),
            "rotation_deg": float(placement["rotation_deg"]),
            "sticker_path": str(placement["sticker_path"]),
        }
        sticker.update(
            _sticker_fields(
                placement["attachment"],
                placement["frame"],
                placement["quad_corners"],
            )
        )
        sticker.update(_texture_stamp_metadata(placement["texture_stamp"]))
        stickers.append(sticker)
    return {
        "decal_mode": "multi_texture_stamp",
        "parent_mesh_path": str(parent_mesh_path),
        "stickers": stickers,
    }


def write_metadata(metadata: dict[str, Any], output_glb_path: Path) -> Path:
    """Write metadata next to an output GLB path as a JSON sidecar.

    Args:
        metadata: JSON-serializable metadata dictionary.
        output_glb_path: Output GLB path whose suffix is replaced with .json.

    Returns:
        Path to the written metadata JSON file.
    """
    metadata_path = output_glb_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path
