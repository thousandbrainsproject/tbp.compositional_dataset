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


def render_records(records: list[MeshRecord]) -> None:
    """Render one mesh or a source/target pair in Vedo.

    Args:
        records: One or two mesh records to render.

    Raises:
        ValueError: If `records` does not contain one or two models.
    """
    if len(records) not in {1, 2}:
        raise ValueError("visualizer accepts one model or one source/target pair")

    plotter = Plotter(
        shape=(1, len(records)),
        sharecam=True,
        size=(1000 * len(records), 800),
        title="GLB texture comparison" if len(records) == 2 else "GLB texture preview",
        interactive=False,
    )
    for index, record in enumerate(records):
        plotter.show(
            build_vedo_actor(record),
            Text2D(record.name, pos="top-left", s=0.7, c="black"),
            at=index,
            axes=1,
            viewup="z",
            interactive=index == len(records) - 1,
        )
    plotter.close()


def render_record(record: MeshRecord) -> None:
    """Render one mesh record in Vedo.

    Args:
        record: Mesh record to render.
    """
    render_records([record])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View a GLB mesh and embedded/base-color texture with Vedo."
    )
    parser.add_argument(
        "glb",
        nargs="+",
        type=Path,
        help="One or two GLB or other mesh files to load.",
    )
    args = parser.parse_args()
    if len(args.glb) not in {1, 2}:
        parser.error("provide one GLB or one source/target pair")
    for path in args.glb:
        if not path.exists():
            raise SystemExit(f"Mesh file not found: {path}")

    records = [load_scene_mesh_records(path) for path in args.glb]
    for record in records:
        print(
            f"Loaded {record.name}: {len(record.vertices)} vertices, "
            f"{len(record.faces)} faces, texture"
        )
    render_records(records)


if __name__ == "__main__":
    main()
