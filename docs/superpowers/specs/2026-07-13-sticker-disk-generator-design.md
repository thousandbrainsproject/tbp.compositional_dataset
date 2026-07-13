# Reproducible Sticker Disk Generator Design

## Goal

Add a Blender Python script that turns one square RGBA sticker PNG into one
genuinely circular, thin GLB object. The GLB must use circular mesh geometry
rather than relying on transparent corners of a square plane.

The script will be reusable and generate one asset per invocation:

```bash
blender --background --python scripts/create_sticker_disk.py -- \
  --image assets/2D_stickers/square.png \
  --out outputs/square.glb
```

## Scale and Orientation

The default disk diameter will be `0.02` Blender units. Existing 3D assets in
`compositional_objects_1.2` have a median longest dimension of `0.10`, so the
disk will normally be about 20% of a parent object's width. This closely
matches the configured maximum texture-stamp size of `0.021`.

The default thickness will be `0.001` Blender units. The disk will lie in the
XY plane, centered at the origin, with its front face normal pointing along
+Z. Its object coordinate conventions are therefore:

- front: `[0.0, 0.0, 1.0]`
- up: `[0.0, 1.0, 0.0]`

The existing `201_square` through `205_heart` object configs currently use
the same vector for front and up. Those five external configs should be
updated to the distinct vectors above when their GLBs are regenerated.

## Command-Line Interface

The script will require:

- `--image`: source PNG path
- `--out`: destination `.glb` path

It will support these optional arguments:

- `--diameter`, default `0.02`
- `--thickness`, default `0.001`
- `--segments`, default `64`

Diameter and thickness must be positive. Segments must be at least three.
The source must be an existing PNG and the output must use the `.glb` suffix.
The script will refuse to overwrite an existing output so accidental dataset
asset replacement is explicit rather than silent.

## Geometry and UV Construction

The generator will construct mesh data directly with
`bpy.data.meshes.new`, avoiding selection-dependent mesh and UV operators.
The mesh will contain:

- one center vertex and one perimeter ring for the front cap;
- one center vertex and one perimeter ring for the back cap;
- triangular cap faces with outward-facing winding;
- quad faces joining the two rings into the circular side wall.

The cap UV coordinates will map the circular perimeter onto the circle
centered at `(0.5, 0.5)` in the source image. Both caps will display the
sticker artwork in a consistent upright orientation when viewed directly.
The side wall does not need image UV detail because it will use a separate
plain-white material.

## Materials and Export

The cap material will use an Image Texture node connected to the Principled
BSDF base color. Both cap sets will use this material. Because the mesh itself
clips the square image to a circle, material alpha will not define the object
silhouette.

The side material will use an opaque white Principled BSDF. The resulting
object will remain circular in renderers that ignore material transparency.

The script will clear the Blender scene, create exactly one named mesh object,
assign the two materials by face, and export only that object as an embedded
GLB. Parent directories for a new output path will be created automatically.

## Code Organization

Pure geometry and validation functions will live in the importable
`tbp.compositional_datasets` package so they can be tested with normal Python.
The Blender entry-point script will be limited to argument parsing, Blender
data-block creation, material wiring, and GLB export.

All new Python functions will have Google-style docstrings.

## Error Handling

Invalid paths, suffixes, dimensions, or segment counts will produce concise
exceptions before the Blender scene is changed. Export failures will propagate
and leave no claim of a completed asset. The script will print the final GLB
path only after export succeeds.

## Testing and Verification

Tests will be written before implementation and will cover:

- CLI argument parsing and defaults;
- invalid diameter, thickness, segment count, and file suffixes;
- deterministic vertex and face counts;
- cap winding and coordinate bounds;
- circular cap UV coordinates and side material assignments.

An integration check will run the script through the installed Blender binary
against one repository sticker. The exported GLB will then be inspected to
verify that it contains one mesh, circular bounds of approximately
`0.02 × 0.02 × 0.001`, and the expected embedded material/texture data.

The full repository test suite will run before the implementation is committed.
