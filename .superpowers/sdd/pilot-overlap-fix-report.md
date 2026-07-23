# Pilot overlap fix report

## Root cause

The existing candidate validator enforced a Euclidean center-distance floor but
did not model the renderer's projected sticker footprints. For object 101, the
first candidate sampled with seed 123 places `back_top` and `back_right`
`0.0328261621633871` apart, above the `0.025` spacing floor. Their configured
rotations of 255 and 330 degrees enlarge the projected square bounds enough to
overlap on both right/up axes, so Blender rejected the candidate later.

## Change

Candidate validation now conservatively treats every sticker as a square whose
side is `max_longest_side`. A rotation of theta gives a projected half-extent
of `side / 2 * (abs(cos(theta)) + abs(sin(theta)))` on both axes. Same-side
pairs are rejected when they overlap by more than `1e-12` on both axes. The
existing displacement, center-spacing, order, and winding constraints remain.

This deliberately overapproximates non-square stickers. It can reject a layout
whose real sticker images would fit, but avoids image or Blender geometry I/O
while preventing candidates that could fail renderer footprint validation.

## TDD evidence

RED command:

```text
pytest -q tests/test_perturbed_scene_dataset.py -k 'seed_123 or projected_footprint_boundary'
```

Before the production change, the exact first-candidate regression failed with
`Failed: DID NOT RAISE <class 'ValueError'>`, and the resampling regression
failed because `perturb_stamp_config` returned attempt 1 instead of the expected
attempt 2. The boundary-contact control passed. Summary: `2 failed, 1 passed`.

GREEN after the minimal production change:

```text
3 passed, 35 deselected in 0.07s
```

The regressions prove that:

- seed 123 attempt 1 is rejected as `back_top`/`back_right` footprint overlap;
- `perturb_stamp_config` deterministically returns seed 123 attempt 2;
- every returned displacement remains within `[0.002, 0.006]`; and
- projected bounds that merely touch are accepted.

## Verification

```text
pytest -q tests/test_perturbed_scene_dataset.py
38 passed in 0.23s

pytest -q
167 passed in 7.48s
```

Final fresh verification and `git diff --check` are run after this report is
written and before commit.
