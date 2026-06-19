# tbp.compositional_datasets

`tbp.compositional_datasets` generates simple compositional 2D-on-3D assets:
a 2D child, such as a sticker, baked into the surface texture of a 3D parent
mesh.

## Installation

### Install `uv`

On macOS, `brew install uv` is sufficient. For other platforms, see the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Install Dependencies

From the repository root, sync the project environment:

```bash
uv sync
```

This creates `.venv/` and installs the local distribution package
`tbp-compositional-datasets`. The Python import package is
`tbp.compositional_datasets`.

For scripts that run in the project environment, prefer `uv run`:

```bash
uv run python scripts/generate_compositional_object_configs.py --help
```

Blender scripts such as `scripts/pick_texture_stamp.py` and
`scripts/stamp_object_from_config.py` run inside Blender's Python process when called
with `blender --python`. That Python environment must also be able to import
`tbp.compositional_datasets`. The exact Blender Python path depends on your
Blender install, but the setup pattern is:

```bash
"/path/to/blender/python" -m pip install -e .
```

## Conceptual Model

The parent object is a 3D triangle mesh. The child object is a 2D sticker whose
surface lives in a local tangent/bitangent frame on the parent surface. The
canonical placement record is:

```python
SurfaceAttachment(
    triangle_id=0,
    barycentric=(0.2, 0.3, 0.5),
    rotation_about_normal_deg=45.0,
    size=(0.25, 0.15),
)
```

`triangle_id` selects a parent triangle. `barycentric` selects the anchor point
inside that triangle. `rotation_about_normal_deg` rotates the sticker footprint
in the triangle's tangent plane. `size` is the sticker width and height in
parent-object units: width follows the rotated tangent axis, and height follows
the rotated bitangent axis. A rotation of `0` means "aligned to the selected
triangle's deterministic tangent," not necessarily aligned to a cube face's
horizontal or vertical axis.

This representation is the ground truth because it is stable, serializable, and
directly tied to mesh geometry. The interactive picker is a convenience layer:
it resolves a user click to a concrete `SurfaceAttachment` before export.

## Interactive Texture Stamp

Run the picker in Blender's GUI so you can click a sensible surface point, such
as the body of a mug:

```bash
blender --python scripts/pick_texture_stamp.py -- \
  --object assets/3D_objects/mug.glb \
  --sticker assets/2D_stickers/star.png
```

By default this writes to `outputs/mug_star_texture_stamp.glb`. Pass `--out`
only when you want a custom output path.

After Blender opens, use the View3D sidebar's **TBP** panel:

1. Click **Pick Surface** and left-click the target mesh.
2. Adjust scale and rotation in the panel.
3. Click **Confirm Stamp** to bake the sticker into the parent texture, export
   the GLB, and write the sidecar metadata.

If the target mesh has no UV map, the picker creates one with Blender's smart UV
projection. The parent mesh must already have a material with a base-color image
texture. Existing image textures are duplicated before stamping so source
textures are not modified in place.

## Configured Six-Sticker Texture Stamp

Use the configured stamping script in Blender background mode when the parent object
should receive the standard six-sticker layout: three stickers on the front
side and three matching slots on the back side. Automatic placement follows
Blender object-space axes: X is right, Z is up, and Y is front/back. The front
side is the negative-Y side, matching Blender Front View.

```bash
blender --background --python scripts/stamp_object_from_config.py -- \
  --object assets/3D_objects/cylinder.glb \
  --config configs/six_sticker_stamp.json \
  --out outputs/cylinder_six_stickers.glb
```

The config lists six explicit sticker PNG paths plus each slot's side,
right/up offset, and optional extra rotation. The repository includes a
ready-to-run config at `configs/six_sticker_stamp.json` using the five
1024x1024 primitive stickers in `assets/2D_stickers`. The bundled 3D parents
are roughly 0.1 Blender units across, so the default offsets and sticker size
are intentionally small:

```json
{
  "min_coverage": 0.95,
  "gap": 0.004,
  "max_longest_side": 0.021,
  "min_longest_side": 0.01,
  "placement": {"right_axis": "x", "up_axis": "z", "front_axis": "y"},
  "slots": [
    {"name": "front_top", "side": "front", "offset": [0.0, 0.018], "sticker": "assets/2D_stickers/star.png", "rotation_deg": 0},
    {"name": "front_left", "side": "front", "offset": [-0.018, -0.012], "sticker": "assets/2D_stickers/square.png", "rotation_deg": 0},
    {"name": "front_right", "side": "front", "offset": [0.018, -0.012], "sticker": "assets/2D_stickers/triangle.png", "rotation_deg": 0},
    {"name": "back_top", "side": "back", "offset": [0.0, 0.018], "sticker": "assets/2D_stickers/circle.png", "rotation_deg": 0},
    {"name": "back_left", "side": "back", "offset": [-0.018, -0.012], "sticker": "assets/2D_stickers/heart.png", "rotation_deg": 0},
    {"name": "back_right", "side": "back", "offset": [0.018, -0.012], "sticker": "assets/2D_stickers/star.png", "rotation_deg": 0}
  ]
}
```

The configured stamping script imports a `.glb` parent asset into a clean Blender scene
and stamps the first mesh object in that scene. Use another parent by changing
`--object` and `--out`:

```bash
blender --background --python scripts/stamp_object_from_config.py -- \
  --object assets/3D_objects/cylinder.glb \
  --config configs/six_sticker_stamp.json \
  --out outputs/cylinder_six_stickers.glb
```

Offsets are object/world units relative to the mesh bounds center, so the
layout is calibrated for similarly scaled parent assets. The script ray-casts
each slot onto the mesh, aligns sticker height with the configured up axis,
computes a shared aspect-preserving sticker size from slot spacing and `gap`,
rejects same-side footprint overlaps, bakes all six stickers into one texture,
exports a GLB, and writes a multi-sticker metadata sidecar.

## Batch Six-Sticker Config Generation

Generate deterministic per-object configs before running Blender when you need
many random sticker combinations using the same six fixed slots:

```bash
python scripts/generate_compositional_object_configs.py \
  --out-dir ~/tbp/data/compositional_objects_1.2 \
  --count 100 \
  --seed 123
```

The generator writes `manifest.json` plus one `object_###/config.json` per
object. Parent GLBs are balanced across `assets/3D_objects/*.glb`; stickers are
sampled independently with replacement for the six fixed slots; rotations are
sampled from all 15-degree increments; and each parent GLB receives unique
slot-to-sticker layouts, ignoring rotation. The manifest records the exact
Blender command to turn each config into a stamped GLB later.

Existing output directories are replaced when generation starts.

To generate a complete Habitat-style scene dataset matching the
`compositional_objects_1.1` layout, including rendered `textured.glb` meshes:

```bash
python scripts/generate_scene_dataset.py \
  --out-dir ~/tbp/data/compositional_objects_1.2 \
  --count 100 \
  --seed 123
```

This writes `compositional_objects.scene_dataset_config.json`,
`configs/<object_id>.object_config.json`, and
`meshes/<object_id>/textured.glb`. Each mesh directory also keeps
`textured.json` placement provenance and the stamped base-color texture as
`textured.png`. The `textured.png` file is downsampled to a 512-pixel maximum
side by default for quick visual inspection; the rendered GLB remains the
authoritative asset. Pass `--preview-texture-max-size` to choose a different
preview size. Existing scene dataset output directories are always replaced.
