"""Preview GLB meshes with Vedo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from vedo import Mesh, Plotter, Text2D

DEFAULT_WINDOW_SIZE = (1000, 800)


class MeshRecord:
    """Renderable mesh geometry extracted from a trimesh scene.

    Attributes:
        name: Display name for the mesh record.
        vertices: Mesh vertices with shape (n, 3).
        faces: Mesh triangle indices with shape (m, 3).
        uvs: Per-vertex UV coordinates.
        texture_image: Texture image array associated with the mesh.
    """

    def __init__(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: np.ndarray,
        texture_image: np.ndarray,
    ) -> None:
        self.name = name
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.texture_image = texture_image


def load_scene_mesh_records(path: Path | str) -> MeshRecord:
    """Load a textured mesh object with trimesh."""
    mesh_path = Path(path)
    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        geometry = loaded
        name = mesh_path.stem
    else:
        if not isinstance(loaded, trimesh.Scene):
            raise TypeError(f"Unsupported mesh payload in {mesh_path}: {type(loaded)!r}")

        geometry_name, geometry = next(iter(loaded.geometry.items()))
        name = loaded.graph.geometry_nodes[geometry_name][0]
        transform, _ = loaded.graph.get(name)
        geometry = geometry.copy()
        geometry.apply_transform(transform)

    return MeshRecord(
        name=name,
        vertices=np.asarray(geometry.vertices, dtype=float),
        faces=np.asarray(geometry.faces, dtype=int),
        uvs=np.asarray(geometry.visual.uv, dtype=float),
        texture_image=np.asarray(geometry.visual.material.baseColorTexture),
    )


def build_vedo_actor(record: MeshRecord) -> Mesh:
    """Build a Vedo mesh actor from one mesh record."""
    actor = Mesh([record.vertices, record.faces])
    actor.texture(tname=record.texture_image, tcoords=record.uvs)
    actor.name = record.name
    return actor


def render_record(record: MeshRecord) -> None:
    """Render a mesh record in Vedo."""
    actor = build_vedo_actor(record)
    label = Text2D(record.name, pos="top-left", s=0.7, c="black")
    plotter = Plotter(
        size=DEFAULT_WINDOW_SIZE,
        title="GLB texture preview",
    )
    plotter.add(actor, label)
    plotter.show(axes=1, viewup="z", interactive=True)
    plotter.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View a GLB mesh and embedded/base-color texture with Vedo."
    )
    parser.add_argument("glb", type=Path, help="GLB or other mesh file to load.")
    args = parser.parse_args()
    if not args.glb.exists():
        raise SystemExit(f"Mesh file not found: {args.glb}")
    record = load_scene_mesh_records(args.glb)
    print(f"Loaded {record.name}: {len(record.vertices)} vertices, {len(record.faces)} faces, texture")
    render_record(record)


if __name__ == "__main__":
    main()
