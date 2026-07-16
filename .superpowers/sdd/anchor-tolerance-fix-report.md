# Rendered anchor provenance tolerance fix report

## Root cause

Rendered anchor provenance compares the signed right/up displacement derived
from two Blender metadata files with the signed displacement configured for the
same slot. The previous absolute-only tolerance was `1e-8`. Actual cylinder
renders showed right-axis errors from approximately `5.4e-8` through `1.03e-7`
while up-axis errors remained around `1e-9`. That drift comes from ray casting
against triangulated cylinder geometry, so the old threshold rejected valid
shared-seed source/target pairs even though their configuration provenance
matched.

## Change

`RENDERED_ANCHOR_PROVENANCE_ABS_TOLERANCE` is now exactly `1e-6`. Comparisons
remain signed and component-by-component for right and up, with
`rel_tol=0.0`. A mismatch reports the configured slot, component, expected
displacement, and rendered displacement.

No perturbation displacement minimum, displacement maximum, geometry tolerance,
or other validation rule changed.

## TDD evidence

RED command against unchanged production code:

```text
pytest -q tests/test_perturbed_scene_dataset.py::test_rendered_anchor_provenance_allows_independent_float_noise tests/test_perturbed_scene_dataset.py::test_rendered_anchor_provenance_rejects_signed_component_mismatch
```

The realistic-drift regression failed with `rendered anchor displacement
changed` because its right/up displacement errors were `1.1e-7` and `1e-7`.
The mismatch regression also failed its diagnostic assertion because the old
exception did not contain `slot front_top`. Summary: `2 failed in 0.10s`.

GREEN after the minimal production change:

```text
2 passed in 0.08s
```

The regressions prove that realistic independent rendered-anchor drift passes,
while a signed right-component error of `1.1e-6` still fails and identifies
`front_top`, `right`, expected `0.002`, and actual `0.0020011`.

The pre-existing rendered-integrity regression continues to reverse an anchor
displacement from positive to negative, preserving coverage for a genuine sign
change.

## Verification

```text
pytest -q tests/test_perturbed_scene_dataset.py
39 passed in 0.21s

pytest -q
168 passed in 7.53s

git diff --check
exit 0
```

Final fresh verification is run after this report is written and before commit.
