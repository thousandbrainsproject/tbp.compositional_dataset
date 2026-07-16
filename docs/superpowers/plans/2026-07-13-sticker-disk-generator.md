# Sticker Disk Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic Blender CLI that converts one sticker PNG into one UV-mapped circular thin-disk GLB.

**Architecture:** Pure Python will calculate and validate all disk vertices, faces, per-loop UVs, and material indices in an importable module. A small Blender-only entry point will translate that data into `bpy` data blocks, wire two materials, and export one selected object as GLB. This boundary keeps geometry behavior testable without Blender while retaining one real-Blender integration check.

**Tech Stack:** Python 3.13, Blender `bpy` 4.3+, Pillow, pytest, trimesh, glTF/GLB

## Global Constraints

- The default diameter is exactly `0.02` Blender units.
- The default thickness is exactly `0.001` Blender units.
- The default segment count is exactly `64`, with a minimum of `3`.
- The disk is centered at the origin in the XZ plane, with its front normal along -Y.
- The cap artwork appears upright from direct views of both -Y and +Y faces.
- The cap material uses the PNG; the side wall uses an opaque white material.
- The output is genuine circular geometry, never a square plane using alpha for its silhouette.
- Existing output files are not overwritten.
- Every new Python function has a Google-style docstring.
- No new runtime dependencies are added.

---

## File Structure

- Create `src/tbp/compositional_datasets/sticker_disk.py`: pure parameter validation and deterministic mesh/UV generation.
- Create `scripts/create_sticker_disk.py`: Blender command-line entry point, mesh/material creation, and GLB export.
- Create `tests/test_sticker_disk.py`: pure geometry and validation tests.
- Create `tests/test_create_sticker_disk.py`: script argument and path-validation tests with a minimal fake `bpy` module.
- Modify `README.md`: usage, defaults, orientation, and five-asset generation examples.

### Task 1: Pure Circular Mesh Generator

**Files:**
- Create: `src/tbp/compositional_datasets/sticker_disk.py`
- Create: `tests/test_sticker_disk.py`

**Interfaces:**
- Consumes: `diameter: float`, `thickness: float`, `segments: int`.
- Produces: `StickerDiskMesh(vertices, faces, face_uvs, material_indices)` and `build_sticker_disk_mesh(*, diameter: float = 0.02, thickness: float = 0.001, segments: int = 64) -> StickerDiskMesh`.

- [ ] **Step 1: Write failing tests for defaults and invalid parameters**

```python
import pytest

from tbp.compositional_datasets.sticker_disk import (
    DEFAULT_DIAMETER,
    DEFAULT_SEGMENTS,
    DEFAULT_THICKNESS,
    build_sticker_disk_mesh,
)


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
    assert max(ys) - min(ys) == pytest.approx(0.001)
    assert max(zs) - min(zs) == pytest.approx(0.02)
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
def test_disk_rejects_invalid_parameters(kwargs: dict[str, float | int], message: str) -> None:
    """Invalid geometry inputs fail before mesh construction."""
    with pytest.raises(ValueError, match=message):
        build_sticker_disk_mesh(**kwargs)
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run: `source .venv/bin/activate && pytest tests/test_sticker_disk.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'tbp.compositional_datasets.sticker_disk'`.

- [ ] **Step 3: Implement constants, result type, validation, and vertex/face generation**

```python
"""Deterministic geometry for circular sticker disk assets."""

from __future__ import annotations

from dataclasses import dataclass
import math

DEFAULT_DIAMETER = 0.02
DEFAULT_THICKNESS = 0.001
DEFAULT_SEGMENTS = 64
CAP_MATERIAL_INDEX = 0
SIDE_MATERIAL_INDEX = 1

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Face = tuple[int, ...]


@dataclass(frozen=True)
class StickerDiskMesh:
    """Pure mesh data used to construct one Blender sticker disk.

    Attributes:
        vertices: Object-space XYZ coordinates.
        faces: Vertex indices for cap triangles and side quads.
        face_uvs: Per-face UV coordinates in face-loop order.
        material_indices: Material slot index for every face.
    """

    vertices: tuple[Vector3, ...]
    faces: tuple[Face, ...]
    face_uvs: tuple[tuple[Vector2, ...], ...]
    material_indices: tuple[int, ...]


def _validate_parameters(diameter: float, thickness: float, segments: int) -> None:
    """Validate requested disk geometry.

    Args:
        diameter: Disk diameter in Blender units.
        thickness: Disk thickness in Blender units.
        segments: Number of perimeter segments.

    Raises:
        ValueError: If a dimension is non-positive or segments is below three.
    """
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if segments < 3:
        raise ValueError("segments must be at least 3")


def build_sticker_disk_mesh(
    *,
    diameter: float = DEFAULT_DIAMETER,
    thickness: float = DEFAULT_THICKNESS,
    segments: int = DEFAULT_SEGMENTS,
) -> StickerDiskMesh:
    """Build circular cap and side-wall mesh data.

    Args:
        diameter: Disk diameter in Blender units.
        thickness: Disk thickness in Blender units.
        segments: Number of perimeter segments.

    Returns:
        Immutable mesh, UV, and material-assignment data.

    Raises:
        ValueError: If the requested geometry is invalid.
    """
    _validate_parameters(diameter, thickness, segments)
    radius = diameter / 2.0
    half_thickness = thickness / 2.0
    angles = tuple(2.0 * math.pi * index / segments for index in range(segments))
    front_ring = tuple((radius * math.cos(a), -half_thickness, radius * math.sin(a)) for a in angles)
    back_ring = tuple((x, half_thickness, z) for x, _y, z in front_ring)
    vertices = ((0.0, -half_thickness, 0.0), *front_ring, (0.0, half_thickness, 0.0), *back_ring)

    front_center = 0
    front_start = 1
    back_center = segments + 1
    back_start = segments + 2
    front_faces = tuple(
        (front_center, front_start + index, front_start + (index + 1) % segments)
        for index in range(segments)
    )
    back_faces = tuple(
        (back_center, back_start + (index + 1) % segments, back_start + index)
        for index in range(segments)
    )
    side_faces = tuple(
        (
            front_start + index,
            back_start + index,
            back_start + (index + 1) % segments,
            front_start + (index + 1) % segments,
        )
        for index in range(segments)
    )
    faces = (*front_faces, *back_faces, *side_faces)
    face_uvs = _build_face_uvs(angles)
    material_indices = (
        *(CAP_MATERIAL_INDEX for _ in range(2 * segments)),
        *(SIDE_MATERIAL_INDEX for _ in range(segments)),
    )
    return StickerDiskMesh(vertices, faces, face_uvs, material_indices)
```

- [ ] **Step 4: Add failing tests for winding, UVs, and material assignment**

```python
def _triangle_normal_y(vertices, face) -> float:
    a, b, c = (vertices[index] for index in face)
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    return ab[2] * ac[0] - ab[0] * ac[2]


def test_caps_face_outward() -> None:
    """Front and back triangle winding points away from the disk center."""
    mesh = build_sticker_disk_mesh(segments=8)
    assert all(_triangle_normal_y(mesh.vertices, face) < 0.0 for face in mesh.faces[:8])
    assert all(_triangle_normal_y(mesh.vertices, face) > 0.0 for face in mesh.faces[8:16])


def test_face_uvs_match_loops_and_stay_in_unit_square() -> None:
    """Every polygon loop gets one bounded UV coordinate."""
    mesh = build_sticker_disk_mesh(segments=8)
    assert [len(uvs) for uvs in mesh.face_uvs] == [len(face) for face in mesh.faces]
    assert all(0.0 <= value <= 1.0 for uvs in mesh.face_uvs for uv in uvs for value in uv)


def test_back_cap_mirrors_u_for_an_upright_direct_view() -> None:
    """Back artwork is not horizontally reversed when viewed from -Z."""
    mesh = build_sticker_disk_mesh(segments=8)
    assert mesh.face_uvs[0][1] == pytest.approx((1.0, 0.5))
    assert mesh.face_uvs[8][2] == pytest.approx((0.0, 0.5))


def test_caps_and_side_use_separate_materials() -> None:
    """Both caps use artwork while only wall quads use white."""
    mesh = build_sticker_disk_mesh(segments=8)
    assert mesh.material_indices == (0,) * 16 + (1,) * 8
```

- [ ] **Step 5: Run the new tests and verify `_build_face_uvs` is missing**

Run: `source .venv/bin/activate && pytest tests/test_sticker_disk.py -v`

Expected: FAIL because `_build_face_uvs` is not defined.

- [ ] **Step 6: Implement deterministic front, mirrored-back, and side UVs**

```python
def _cap_uv(angle: float, *, mirror_u: bool = False) -> Vector2:
    """Return a circular cap UV for one perimeter angle.

    Args:
        angle: Perimeter angle in radians.
        mirror_u: Whether to mirror U for an upright back-face view.

    Returns:
        UV coordinate inside the unit square.
    """
    cosine = math.cos(angle)
    u = 0.5 - 0.5 * cosine if mirror_u else 0.5 + 0.5 * cosine
    return (u, 0.5 + 0.5 * math.sin(angle))


def _build_face_uvs(angles: tuple[float, ...]) -> tuple[tuple[Vector2, ...], ...]:
    """Build UVs in the same order as generated polygon loops.

    Args:
        angles: Ordered perimeter angles.

    Returns:
        Per-loop UV coordinates for front caps, back caps, and side quads.
    """
    segments = len(angles)
    front = tuple(
        ((0.5, 0.5), _cap_uv(angles[index]), _cap_uv(angles[(index + 1) % segments]))
        for index in range(segments)
    )
    back = tuple(
        (
            (0.5, 0.5),
            _cap_uv(angles[(index + 1) % segments], mirror_u=True),
            _cap_uv(angles[index], mirror_u=True),
        )
        for index in range(segments)
    )
    side = tuple(
        (
            (index / segments, 1.0),
            (index / segments, 0.0),
            ((index + 1) / segments, 0.0),
            ((index + 1) / segments, 1.0),
        )
        for index in range(segments)
    )
    return (*front, *back, *side)
```

- [ ] **Step 7: Run Task 1 tests and commit**

Run: `source .venv/bin/activate && pytest tests/test_sticker_disk.py -v`

Expected: all Task 1 tests PASS.

```bash
git add src/tbp/compositional_datasets/sticker_disk.py tests/test_sticker_disk.py
git commit -m "feat: add deterministic sticker disk geometry"
```

### Task 2: Blender CLI and GLB Export

**Files:**
- Create: `scripts/create_sticker_disk.py`
- Create: `tests/test_create_sticker_disk.py`

**Interfaces:**
- Consumes: `build_sticker_disk_mesh`, `argv_after_blender_separator`, a PNG path, a new GLB path, and optional geometry values.
- Produces: `parse_args(argv: list[str] | None = None) -> argparse.Namespace`, `validate_asset_paths(image_path: Path, output_path: Path) -> None`, and `create_sticker_disk(args: argparse.Namespace, bpy_module: Any) -> Path`.

- [ ] **Step 1: Write failing tests for CLI defaults and path validation**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


def load_script():
    """Load the Blender script with a minimal importable `bpy` double."""
    sys.modules["bpy"] = SimpleNamespace()
    path = Path(__file__).resolve().parents[1] / "scripts" / "create_sticker_disk.py"
    spec = importlib.util.spec_from_file_location("create_sticker_disk", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_approved_defaults() -> None:
    """Only image and output are required for the standard disk."""
    script = load_script()
    args = script.parse_args(["--image", "square.png", "--out", "square.glb"])
    assert args.image == Path("square.png")
    assert args.out == Path("square.glb")
    assert args.diameter == 0.02
    assert args.thickness == 0.001
    assert args.segments == 64


def test_validate_asset_paths_accepts_new_png_to_glb(tmp_path: Path) -> None:
    """An existing PNG and absent GLB form a valid conversion request."""
    script = load_script()
    image = tmp_path / "square.png"
    image.write_bytes(b"png")
    script.validate_asset_paths(image, tmp_path / "square.glb")


@pytest.mark.parametrize(
    ("image_name", "out_name", "create_out", "message"),
    [
        ("missing.png", "disk.glb", False, "sticker image does not exist"),
        ("sticker.jpg", "disk.glb", False, "sticker image must be a .png"),
        ("sticker.png", "disk.obj", False, "output must be a .glb"),
        ("sticker.png", "disk.glb", True, "output already exists"),
    ],
)
def test_validate_asset_paths_rejects_invalid_requests(
    tmp_path: Path,
    image_name: str,
    out_name: str,
    create_out: bool,
    message: str,
) -> None:
    """Invalid source or destination paths fail before Blender mutation."""
    script = load_script()
    image = tmp_path / image_name
    if image_name != "missing.png":
        image.write_bytes(b"image")
    out = tmp_path / out_name
    if create_out:
        out.write_bytes(b"existing")
    with pytest.raises((FileNotFoundError, ValueError, FileExistsError), match=message):
        script.validate_asset_paths(image, out)
```

- [ ] **Step 2: Run the script tests and verify the missing-file failure**

Run: `source .venv/bin/activate && pytest tests/test_create_sticker_disk.py -v`

Expected: FAIL because `scripts/create_sticker_disk.py` does not exist.

- [ ] **Step 3: Implement argument parsing and pre-mutation validation**

Create `scripts/create_sticker_disk.py` with this import and path setup, then
add the argument and validation functions below it:

```python
"""Create one circular sticker disk GLB from a PNG image."""

from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    site.addsitedir(USER_SITE)

from tbp.compositional_datasets.blender_cli import argv_after_blender_separator
from tbp.compositional_datasets.blender_scene import clear_scene
from tbp.compositional_datasets.sticker_disk import (
    DEFAULT_DIAMETER,
    DEFAULT_SEGMENTS,
    DEFAULT_THICKNESS,
    StickerDiskMesh,
    build_sticker_disk_mesh,
)

import bpy
```

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse sticker-disk generator arguments.

    Args:
        argv: Optional argument list. Uses `sys.argv` when None.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Create a circular sticker disk GLB from one PNG.")
    parser.add_argument("--image", required=True, type=Path, help="Source sticker PNG path.")
    parser.add_argument("--out", required=True, type=Path, help="New output GLB path.")
    parser.add_argument("--diameter", type=float, default=DEFAULT_DIAMETER)
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    return parser.parse_args(argv)


def validate_asset_paths(image_path: Path, output_path: Path) -> None:
    """Validate source and destination paths before changing Blender state.

    Args:
        image_path: Existing sticker PNG.
        output_path: New GLB destination.

    Raises:
        FileNotFoundError: If the source image does not exist.
        ValueError: If either suffix is unsupported.
        FileExistsError: If the destination already exists.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"sticker image does not exist: {image_path}")
    if image_path.suffix.lower() != ".png":
        raise ValueError("sticker image must be a .png")
    if output_path.suffix.lower() != ".glb":
        raise ValueError("output must be a .glb")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
```

- [ ] **Step 4: Run the script tests and verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_create_sticker_disk.py -v`

Expected: all CLI and validation tests PASS.

- [ ] **Step 5: Implement Blender mesh, UV, materials, export, and main entry point**

Add focused helpers to `scripts/create_sticker_disk.py`:

```python
def _create_cap_material(bpy_module: Any, image_path: Path) -> Any:
    """Create the image-textured material used by both disk caps.

    Args:
        bpy_module: Imported Blender Python module.
        image_path: Absolute source sticker PNG path.

    Returns:
        Blender material with the image connected to base color.
    """
    material = bpy_module.data.materials.new(name="StickerCaps")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    texture = nodes.new(type="ShaderNodeTexImage")
    texture.image = bpy_module.data.images.load(str(image_path), check_existing=False)
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    return material


def _create_side_material(bpy_module: Any) -> Any:
    """Create the opaque white material used by the disk side wall.

    Args:
        bpy_module: Imported Blender Python module.

    Returns:
        Blender material with an opaque white base color.
    """
    material = bpy_module.data.materials.new(name="StickerEdge")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    return material


def _create_mesh_object(bpy_module: Any, mesh_data: StickerDiskMesh, image_path: Path) -> Any:
    """Create one Blender object from validated pure mesh data.

    Args:
        bpy_module: Imported Blender Python module.
        mesh_data: Deterministic vertices, faces, UVs, and material indices.
        image_path: Absolute source sticker PNG path.

    Returns:
        Linked Blender mesh object with cap and side materials.
    """
    mesh = bpy_module.data.meshes.new("StickerDiskMesh")
    mesh.from_pydata(mesh_data.vertices, [], mesh_data.faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, face_uvs in zip(mesh.polygons, mesh_data.face_uvs, strict=True):
        polygon.material_index = mesh_data.material_indices[polygon.index]
        for loop_index, uv in zip(polygon.loop_indices, face_uvs, strict=True):
            uv_layer.data[loop_index].uv = uv
    obj = bpy_module.data.objects.new("StickerDisk", mesh)
    bpy_module.context.collection.objects.link(obj)
    obj.data.materials.append(_create_cap_material(bpy_module, image_path))
    obj.data.materials.append(_create_side_material(bpy_module))
    return obj


def create_sticker_disk(args: argparse.Namespace, bpy_module: Any) -> Path:
    """Create and export one circular sticker disk asset.

    Args:
        args: Parsed image, output, and geometry arguments.
        bpy_module: Imported Blender Python module.

    Returns:
        Absolute path to the exported GLB.

    Raises:
        FileNotFoundError: If the source image does not exist.
        FileExistsError: If the output already exists.
        ValueError: If paths or geometry values are invalid.
    """
    validate_asset_paths(args.image, args.out)
    mesh_data = build_sticker_disk_mesh(
        diameter=args.diameter,
        thickness=args.thickness,
        segments=args.segments,
    )
    clear_scene(bpy_module)
    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obj = _create_mesh_object(bpy_module, mesh_data, args.image.resolve())
    obj.select_set(True)
    bpy_module.context.view_layer.objects.active = obj
    bpy_module.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
    )
    return output_path


def main() -> None:
    """Run the Blender sticker-disk generator."""
    args = parse_args(argv_after_blender_separator())
    try:
        output_path = create_sticker_disk(args, bpy)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Wrote {output_path}")
```

- [ ] **Step 6: Run a real Blender export and inspect the generated GLB**

Run:

```bash
tmp_dir="$(mktemp -d)"
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/square.png \
  --out "$tmp_dir/square.glb"
source .venv/bin/activate
python - "$tmp_dir/square.glb" <<'PY'
from pathlib import Path
import sys
import trimesh

scene = trimesh.load(Path(sys.argv[1]), force="scene")
extents = scene.extents
assert abs(float(extents[0]) - 0.02) < 1e-6
assert abs(float(extents[1]) - 0.02) < 1e-6
assert abs(float(extents[2]) - 0.001) < 1e-6
print(extents)
PY
```

Expected: Blender exits zero and Python prints extents equivalent to `[0.02, 0.02, 0.001]`.

- [ ] **Step 7: Run Task 2 tests and commit**

Run: `source .venv/bin/activate && pytest tests/test_sticker_disk.py tests/test_create_sticker_disk.py -v`

Expected: all Task 1 and Task 2 tests PASS.

```bash
git add scripts/create_sticker_disk.py tests/test_create_sticker_disk.py
git commit -m "feat: export circular sticker disk assets"
```

### Task 3: Documentation and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the completed `scripts/create_sticker_disk.py` CLI.
- Produces: copy-pasteable single-asset and five-asset generation instructions, plus correct object-config orientation values.

- [ ] **Step 1: Add the documented single-asset command and defaults**

Add a `## Circular 2D Sticker Objects` section after `## Conceptual Model`
containing the following text and command:

````markdown
## Circular 2D Sticker Objects

Create a standalone circular sticker object in Blender background mode:

```bash
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/square.png \
  --out outputs/square.glb
```

The generated object is actual thin-disk geometry, not a transparent square
plane. Defaults are diameter `0.02`, thickness `0.001`, and 64 perimeter
segments. Override them with `--diameter`, `--thickness`, and `--segments`.
The disk lies in Blender's XZ plane with front normal -Y, preserving the
orientation of the existing standalone 2D assets. Existing dataset object
configs do not need to change.
````

- [ ] **Step 2: Add five explicit reproducible generation commands**

Document these separate commands rather than adding a batch API:

```bash
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/square.png \
  --out outputs/201_square/textured.glb
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/circle.png \
  --out outputs/202_circle/textured.glb
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/triangle.png \
  --out outputs/203_triangle/textured.glb
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/star.png \
  --out outputs/204_star/textured.glb
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/heart.png \
  --out outputs/205_heart/textured.glb
```

- [ ] **Step 3: Run full verification**

Run: `source .venv/bin/activate && pytest -q`

Expected: the complete repository test suite passes with zero failures.

Run: `git diff --check`

Expected: no whitespace errors.

Run the real Blender export and trimesh bounds inspection from Task 2 once more using a fresh temporary directory.

Expected: one circular mesh with extents `[0.02, 0.02, 0.001]` and a successful GLB export.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: document circular sticker asset generation"
```

- [ ] **Step 5: Report external dataset rollout separately**

Do not silently overwrite `/Users/hlee/tbp/data/compositional_objects_1.2`. Report the five commands needed to regenerate `201_square` through `205_heart`; their existing config vectors remain unchanged. Apply external asset changes only with explicit filesystem authorization.
