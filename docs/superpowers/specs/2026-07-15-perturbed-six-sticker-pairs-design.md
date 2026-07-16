# Perturbed Six-Sticker Object Pairs Design

## Purpose

Extend `~/tbp/data/compositional_objects_1.2` with 100 perturbed counterparts
for the existing six-sticker objects `101` through `200`. Each new object keeps
the same parent and children as its source while moving every sticker enough to
be perceptible. The paired data is intended to test whether compositional Monty
recognizes an object when its parts retain their identities and approximate
arrangement but occupy slightly different locations.

The expected experimental contrast is that monolithic Monty will struggle with
the perturbed objects while compositional Monty will generalize better.

## Scope

The workflow will:

- Pair source IDs `101` through `200` with target IDs `201` through `300`.
- Append the targets to `compositional_objects_1.2` without changing objects
  `001` through `200`.
- Preserve each source object's parent mesh, six slot names, sticker files,
  sides, configured rotations, and rendered sticker sizes.
- Independently perturb the right/up offset of every sticker.
- Preserve the recognizable top-left-right triangular arrangement on both the
  front and back sides.
- Support a five-pair pilot before the complete append.
- Provide a paired source/target visualization for human calibration.

The workflow will not randomize sticker identity, rotation, scale, parent
geometry, material, or object pose. It will not preserve obsolete batch CLI
behavior or introduce a general-purpose constraint solver.

## Source and Target Mapping

The source dataset contains exactly 100 generated six-sticker configs whose
numeric prefixes span `101` through `200`. A target ID is formed by adding 100
to the source numeric prefix while retaining the descriptive suffix:

```text
101_cube_6x2d_stickers -> 201_cube_6x2d_stickers
102_cylinder_6x2d_stickers -> 202_cylinder_6x2d_stickers
...
200_sphere_6x2d_stickers -> 300_sphere_6x2d_stickers
```

For each source object, the workflow reads:

- `generation_configs/<source_id>.json` for slots, stickers, offsets, and
  configured rotations.
- `meshes/<source_id>/textured.json` for the parent mesh provenance and rendered
  source attachment metadata.
- `configs/<source_id>.object_config.json` for the Habitat object settings.

It writes the corresponding target generation config, mesh directory, and
Habitat object config under the target ID. The existing
`compositional_objects.scene_dataset_config.json` remains unchanged because it
already discovers object configs from `configs/`.

## Perturbation Model

Each of the six slots is perturbed independently in the two-dimensional
right/up offset plane. A displacement is sampled in polar form:

```text
direction ~ Uniform(0, 2*pi)
distance  ~ Uniform(min_displacement, max_displacement)
delta     = distance * (cos(direction), sin(direction))
```

The initial bounds are:

- `min_displacement = 0.002`
- `max_displacement = 0.006`

The minimum guarantees that every child moves perceptibly rather than relying
on the aggregate layout to differ. The maximum is a pilot value, not a hard-coded
experimental constant. It may be increased and the pilot rerun if the movement
is still too subtle, provided all geometric and render validations pass.

Sampling uses a required seed and a stable source-ID order. Repeating a run with
the same inputs and seed must produce identical target configs.

## Structural Constraints

A complete six-slot candidate is accepted only when both same-side triads meet
all of these conditions:

1. The named top slot remains above both named lower slots.
2. The named left slot remains left of the named right slot.
3. The triangle retains the source triangle's winding and has nonzero area.
4. Every same-side center-to-center distance is at least `0.025`, derived from
   `max_longest_side + gap = 0.021 + 0.004`.
5. Every slot displacement is within the configured inclusive minimum and
   maximum bounds, subject only to floating-point tolerance.

The spacing rule prevents `compute_common_slot_sizes` from reducing the common
longest side below the source value. The existing render path remains the
authority for actual footprint overlap because sticker aspect ratio, configured
rotation, local tangent frames, and curved surfaces affect the final projected
quads.

Sampling rejects the entire candidate and resamples all six slots when a
structural constraint fails. The sampler has a fixed attempt limit and reports
the source object plus the last violated constraint if it cannot find a valid
candidate.

## Rotation Semantics

The target generation config copies every source slot's `rotation_deg` exactly.
This is the experiment's definition of keeping rotation fixed. The renderer
combines that configured value with the local tangent-frame alignment. Moving a
sticker on a curved surface can therefore change low-level
`rotation_about_normal_deg` metadata while preserving the same visual rotation
relative to the surface-aligned layout.

## Rendering and Output Safety

The append workflow validates the complete source range and target range before
rendering. It refuses to start if:

- A required source generation config, object config, mesh, or metadata file is
  missing.
- Source numeric IDs are duplicated or discontinuous for the requested range.
- Any requested target config or mesh directory already exists.
- The source and target mapping is not one-to-one.

All requested targets render into a staging directory. The existing Blender
stamping path must validate successful surface ray hits, at least `0.95` texture
coverage, and non-overlapping same-side footprints. A subprocess failure,
missing render output, or failed paired-metadata check aborts the run and leaves
the installed dataset unchanged. Only after the full requested batch passes are
the staged generation configs, object configs, and mesh directories moved into
the output dataset.

This all-or-nothing rule applies to both the five-object pilot and the complete
100-object append. A separate pilot output directory starts empty; an in-place
production append preserves every existing file.

## Post-Render Pair Verification

Before installation, each target is compared with its source. Verification
requires:

- Identical parent mesh provenance.
- The same six slot names and sides.
- The same sticker path in each named slot.
- Exact equality of configured `rotation_deg` per slot.
- Rendered sticker sizes equal within floating-point tolerance.
- A target anchor displacement of at least `min_displacement` and no more than
  `max_displacement` in the configured right/up axes.
- Coverage at or above the config's `min_coverage`.
- No renderer-reported footprint overlap.

The config offsets are the canonical displacement values. World-space anchors
are checked as render provenance, with tolerance for ray casting and
floating-point representation.

## Components

Implementation will add three focused capabilities:

1. A perturbation module with pure functions for ID mapping, annular sampling,
   triangle validation, config cloning, and paired metadata verification. It
   also coordinates staged rendering and installation.
2. A CLI accepting source dataset, output dataset, source range, ID offset,
   seed, displacement bounds, Blender executable, and preview texture size.
   Using a separate output directory creates the pilot; using the source dataset
   as the output performs the final append.
3. A paired visualization mode that displays a source GLB and its target GLB
   side by side with consistent camera and up-axis settings.

The implementation will reuse the existing config loader, Blender command
construction, render failure diagnostics, texture preview normalization, and
Habitat object-config conventions. New Python functions will use Google-style
docstrings.

## Pilot and Calibration Workflow

The first run selects sources `101` through `105`, which cover cube, cylinder,
disk, mug, and sphere, and targets them to `201` through `205` in a separate
pilot directory. The initial pilot uses `[0.002, 0.006]` displacement bounds.

Each source/target pair is reviewed side by side. Approval requires that every
object remains recognizable as the same six-sticker composition and that the
location changes are readily perceptible when looking back and forth. If the
changes are too subtle, only `max_displacement` is increased and the same seeded
pilot is rerun. The accepted maximum is then used for the full `101` through
`200` production run.

## Testing

Unit tests will cover:

- Stable `+100` ID mapping and suffix preservation.
- Deterministic configs for a fixed seed and source order.
- Inclusive minimum and maximum displacement bounds for all six slots.
- Exact preservation of sticker paths, sides, slot names, rotations, and
  non-offset config fields.
- Preservation of top/left/right ordering, winding, and minimum spacing.
- Rejection and resampling of candidates that violate each constraint.
- Clear failure after the configured sampling-attempt limit.
- Preflight refusal when any target already exists.
- Staging cleanup and no installed partial batch after a simulated render
  failure.
- Post-render source/target metadata checks, including deliberate mismatch
  failures.

CLI-level tests will use a temporary miniature dataset and a fake render command
to verify source discovery, target paths, staged installation, and error
reporting without invoking Blender. The five-shape Blender pilot is the real
integration and visual acceptance test before rendering all 100 targets.

## Acceptance Criteria

The design is satisfied when:

- A five-shape pilot can be generated reproducibly and reviewed side by side.
- A human approves an upper displacement bound that is perceptible without
  breaking the composition.
- Objects `201` through `300` are appended to
  `compositional_objects_1.2` as direct pairs of `101` through `200`.
- Every target preserves its source parent, sticker identities, sides,
  configured rotations, and rendered sticker sizes.
- Every sticker moves by at least `0.002` and no more than the approved maximum.
- Both triads remain recognizable, valid, and non-overlapping.
- The original dataset files remain unchanged.
- All automated tests pass and the complete rendered batch passes paired
  metadata verification.
