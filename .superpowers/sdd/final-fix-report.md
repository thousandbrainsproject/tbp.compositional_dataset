# Final Pilot Fix Report

## Scope

Completed the whole-branch review fixes requested at starting HEAD `68d7f39`:

- Relaxed signed rendered-anchor component provenance comparisons to a named
  absolute tolerance of `1e-8`, with relative tolerance disabled.
- Added focused validator coverage for top ordering, left/right ordering, and
  triangle winding reversal.
- Added a controlled real-validator reject-then-resample regression.
- Added batch failure-atomicity coverage where target one verifies, target two
  writes all render outputs but fails rendered-pair metadata verification, and
  neither requested target is installed.
- Clarified that `target_object_id` is generic while protected pair discovery
  and append orchestration require `id_offset == 100`.
- Hardened paired-viewer tests for actors, label contents and styling, viewport
  indices, axes, `viewup`, interactivity, close behavior, and one-record
  delegation through `render_records([record])`.

## TDD Evidence

### RED

Command:

```text
source .venv/bin/activate && pytest -q tests/test_perturbed_scene_dataset.py::test_rendered_anchor_provenance_allows_independent_float_noise
```

Result before the production change:

```text
1 failed in 0.09s
ValueError: rendered anchor displacement changed
```

The source and target metadata used independent signed anchor errors of
`+4e-9/-4e-9` on the right component and `-3e-9/+3e-9` on the up component.
The resulting `8e-9` and `6e-9` provenance discrepancies exceeded the old
`1e-9` threshold as intended by the regression.

### GREEN

Minimal implementation: added `ABSOLUTE_PROVENANCE_TOLERANCE = 1e-8` and used:

```text
math.isclose(rendered, configured, rel_tol=0.0,
             abs_tol=ABSOLUTE_PROVENANCE_TOLERANCE)
```

Command covering the new tolerance regression and existing genuine signed
mismatch cases:

```text
source .venv/bin/activate && pytest -q tests/test_perturbed_scene_dataset.py::test_rendered_anchor_provenance_allows_independent_float_noise tests/test_perturbed_scene_dataset.py::test_verification_rejects_rendered_integrity_mismatch
```

Result:

```text
7 passed in 0.09s
```

The genuine mismatch case still reverses the signed right-axis displacement
relative to configuration and remains rejected.

## Verification

Focused perturbed-dataset and viewer command:

```text
source .venv/bin/activate && pytest -q tests/test_perturbed_scene_dataset.py tests/test_paired_visualize_model.py tests/test_visualize_model.py
```

Result:

```text
41 passed in 0.70s
```

Full suite command, run once after the final test/code changes:

```text
source .venv/bin/activate && pytest -q
```

Result:

```text
164 passed in 7.43s
```

Whitespace verification:

```text
git diff --check
```

Result: exit 0 with no output.

## Files Changed

- `src/tbp/compositional_datasets/perturbed_scene_dataset.py`
- `tests/test_perturbed_scene_dataset.py`
- `tests/test_paired_visualize_model.py`
- `.superpowers/sdd/final-fix-report.md`

## Self-Review

- The provenance tolerance applies only to signed rendered/configured right and
  up displacement components; sticker-size tolerance and geometric validation
  are unchanged.
- Relative tolerance is explicitly disabled, preventing comparison scale from
  weakening the absolute provenance bound.
- The metadata failure-atomicity test reaches the second render after the first
  target has already verified, so it proves prior staged work is not installed
  when later metadata verification fails.
- Viewer production behavior was not changed; the new assertions document its
  existing contract.
- Unrelated untracked `.DS_Store` files were left untouched.

## Acknowledged Operational Residuals

- Installation still uses three sequential `shutil.move` operations per target,
  so a filesystem failure during installation is not transactionally rolled
  back. Transactional multi-move installation was explicitly out of scope.
- The CLI does not enforce a separate pilot-gate policy beyond the library and
  workflow validations already present. CLI pilot-gate enforcement was
  explicitly out of scope.
