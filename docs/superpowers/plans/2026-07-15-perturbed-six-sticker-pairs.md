# Perturbed Six-Sticker Object Pairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproducibly append objects `201`–`300` to `compositional_objects_1.2` as position-perturbed pairs of objects `101`–`200`, after a five-shape visual pilot.

**Architecture:** A new dataset module clones source configs, independently samples bounded annular displacement, rejects invalid triads, stages Blender renders, verifies paired metadata, and installs only a fully successful batch. A thin CLI supports both a separate pilot and the in-place append. The Vedo viewer gains a shared-camera two-model mode.

**Tech Stack:** Python 3.13, Blender `bpy` 4.3+, Pillow, Vedo 2026.6+, trimesh, pytest, Habitat-style JSON

## Global Constraints

- Map source IDs `101`–`200` to target IDs `201`–`300` by adding 100.
- Never modify files belonging to IDs `001`–`200`.
- Preserve parent, sticker, slot, side, `rotation_deg`, non-offset config fields, and rendered sticker size.
- Sample every sticker independently with radial displacement in `[0.002, 0.006]` for the initial pilot.
- Preserve top/left/right ordering, winding, and same-side center spacing of at least `0.025`.
- Require a seed and stable source order.
- Refuse target conflicts before rendering and install no requested targets after render or verification failure.
- Keep Blender authoritative for ray hits, coverage of at least `0.95`, and actual footprint overlap.
- Require explicit human approval after the five-shape pilot and before production.
- Add Google-style docstrings to every new Python function.
- Add no dependencies.

---

## File Structure

- Create `src/tbp/compositional_datasets/perturbed_scene_dataset.py`: sampling, validation, discovery, staged rendering, installation, and verification.
- Create `scripts/append_perturbed_scene_objects.py`: pilot/production CLI.
- Create `tests/test_perturbed_scene_dataset.py`: pure and orchestration tests.
- Create `tests/test_append_perturbed_scene_objects.py`: CLI tests.
- Modify `scripts/visualize_model.py`: one/two-model rendering.
- Create `tests/test_paired_visualize_model.py`: non-GUI subplot tests.
- Modify `README.md`: exact pilot, review, and production commands.

The repository ignores new `tests/` and `docs/superpowers/` files. Force-add only the named new files; preserve unrelated ignored tests, `.superpowers/`, and `.DS_Store` files.

### Task 1: Pure Perturbation Model

**Files:**
- Create: `src/tbp/compositional_datasets/perturbed_scene_dataset.py`
- Create: `tests/test_perturbed_scene_dataset.py`

**Interfaces:**
- Produces `PerturbationBounds(min_displacement=0.002, max_displacement=0.006, max_attempts=10_000)`.
- Produces `target_object_id(source_id: str, id_offset: int) -> str`.
- Produces `validate_perturbed_config(source: dict[str, Any], target: dict[str, Any], bounds: PerturbationBounds) -> None`.
- Produces `perturb_stamp_config(source: dict[str, Any], rng: random.Random, bounds: PerturbationBounds) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests**

Create a six-slot fixture using offsets `(0, .018)`, `(-.018, -.012)`, and `(.018, -.012)` on both sides. Add these exact assertions:

```python
def test_target_id_adds_100_and_preserves_suffix():
    assert target_object_id("101_cube_6x2d_stickers", 100) == "201_cube_6x2d_stickers"
    assert target_object_id("200_sphere_6x2d_stickers", 100) == "300_sphere_6x2d_stickers"


def test_seeded_perturbation_is_deterministic_and_changes_only_offsets(source_config):
    bounds = PerturbationBounds(0.002, 0.006)
    first = perturb_stamp_config(source_config, random.Random(123), bounds)
    second = perturb_stamp_config(source_config, random.Random(123), bounds)
    assert first == second
    for source_slot, target_slot in zip(source_config["slots"], first["slots"], strict=True):
        assert {k: v for k, v in source_slot.items() if k != "offset"} == {
            k: v for k, v in target_slot.items() if k != "offset"
        }
        assert 0.002 <= math.dist(source_slot["offset"], target_slot["offset"]) <= 0.006
    validate_perturbed_config(source_config, first, bounds)


def test_impossible_spacing_reports_attempt_limit(source_config):
    source_config["max_longest_side"] = 1.0
    bounds = PerturbationBounds(0.002, 0.006, max_attempts=3)
    with pytest.raises(ValueError, match="after 3 attempts"):
        perturb_stamp_config(source_config, random.Random(7), bounds)
```

- [ ] **Step 2: Run the tests to verify the missing-module failure**

Run: `source .venv/bin/activate && pytest tests/test_perturbed_scene_dataset.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimal pure module**

Use these exact public definitions and validation formulas:

```python
@dataclass(frozen=True)
class PerturbationBounds:
    min_displacement: float = 0.002
    max_displacement: float = 0.006
    max_attempts: int = 10_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_displacement) or self.min_displacement <= 0:
            raise ValueError("min_displacement must be positive and finite")
        if not math.isfinite(self.max_displacement):
            raise ValueError("max_displacement must be finite")
        if self.max_displacement < self.min_displacement:
            raise ValueError("max_displacement must be at least min_displacement")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


def target_object_id(source_id: str, id_offset: int) -> str:
    match = re.fullmatch(r"(\d{3})_(.+)", source_id)
    if match is None or id_offset <= 0:
        raise ValueError(f"invalid source ID or offset: {source_id}, {id_offset}")
    target_number = int(match.group(1)) + id_offset
    if target_number > 999:
        raise ValueError("target object ID exceeds three digits")
    return f"{target_number:03d}_{match.group(2)}"


def perturb_stamp_config(source, rng, bounds):
    last_error = "no candidate"
    for _attempt in range(bounds.max_attempts):
        candidate = deepcopy(source)
        for slot in candidate["slots"]:
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.uniform(bounds.min_displacement, bounds.max_displacement)
            slot["offset"][0] += radius * math.cos(angle)
            slot["offset"][1] += radius * math.sin(angle)
        try:
            validate_perturbed_config(source, candidate, bounds)
        except ValueError as error:
            last_error = str(error)
            continue
        return candidate
    raise ValueError(
        f"could not sample a valid perturbed layout after {bounds.max_attempts} attempts: {last_error}"
    )
```

`validate_perturbed_config` must compare every non-offset field for equality; check radial bounds with tolerance `1e-12`; require top Y greater than both lower Y values; require left X less than right X; require source and target signed areas to have the same nonzero sign; and require each same-side distance to be at least `target["max_longest_side"] + target["gap"]`.

- [ ] **Step 4: Run tests and commit**

Run: `source .venv/bin/activate && pytest tests/test_perturbed_scene_dataset.py -v`

Expected: PASS.

```bash
git add src/tbp/compositional_datasets/perturbed_scene_dataset.py
git add -f tests/test_perturbed_scene_dataset.py
git commit -m "feat: add bounded sticker position perturbation"
```

### Task 2: Discovery and Pair Verification

**Files:**
- Modify: `src/tbp/compositional_datasets/perturbed_scene_dataset.py`
- Modify: `tests/test_perturbed_scene_dataset.py`

**Interfaces:**
- Produces immutable `ObjectPair` with source/target IDs and generation-config, object-config, and mesh paths.
- Produces `discover_object_pairs(source_dataset: Path, out_dir: Path, *, source_start: int, count: int, id_offset: int) -> tuple[ObjectPair, ...]`.
- Produces `target_object_config(source: dict[str, Any], target_id: str) -> dict[str, Any]`.
- Produces `verify_rendered_pair(source_config, target_config, source_metadata, target_metadata, bounds) -> None`.

Define the record exactly as:

```python
@dataclass(frozen=True)
class ObjectPair:
    source_id: str
    target_id: str
    source_generation_config: Path
    source_object_config: Path
    source_mesh_dir: Path
    target_generation_config: Path
    target_object_config: Path
    target_mesh_dir: Path
```

- [ ] **Step 1: Write failing discovery tests**

Build miniature dataset fixtures containing generation config, Habitat config, `textured.glb`, and `textured.json`. Assert:

```python
pairs = discover_object_pairs(dataset, dataset, source_start=101, count=2, id_offset=100)
assert [pair.target_id for pair in pairs] == [
    "201_cube_6x2d_stickers",
    "202_cylinder_6x2d_stickers",
]

with pytest.raises(FileNotFoundError, match="source object 103"):
    discover_object_pairs(dataset, dataset, source_start=101, count=3, id_offset=100)

(dataset / "generation_configs" / "201_cube_6x2d_stickers.json").write_text("{}")
with pytest.raises(FileExistsError, match="201_cube_6x2d_stickers"):
    discover_object_pairs(dataset, dataset, source_start=101, count=1, id_offset=100)
```

- [ ] **Step 2: Implement preflight discovery**

For every requested numeric prefix, glob `generation_configs/{number:03d}_*.json` and require exactly one result. Require its source Habitat config, `textured.glb`, and `textured.json`. Construct the target by adding `id_offset`. Reject any existing target generation config, Habitat config, or mesh directory before returning any pairs. `target_object_config` deep-copies the source and changes only:

```python
target["render_asset"] = f"../meshes/{target_id}/textured.glb"
```

- [ ] **Step 3: Write failing metadata-verification tests**

Create metadata records keyed by all six slot names with parent path, side, sticker path, `rotation_deg`, `size`, anchor, and coverage. Assert the matching pair passes, then mutate target size and assert:

```python
with pytest.raises(ValueError, match="rendered sticker size changed"):
    verify_rendered_pair(source_config, target_config, source_metadata, target_metadata, bounds)
```

- [ ] **Step 4: Implement paired verification**

`verify_rendered_pair` must first call `validate_perturbed_config`, then require:

```python
source_metadata["parent_mesh_path"] == target_metadata["parent_mesh_path"]
source_record["side"] == target_record["side"]
source_record["sticker_path"] == target_record["sticker_path"]
source_record["rotation_deg"] == target_record["rotation_deg"]
all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(source_record["size"], target_record["size"], strict=True))
target_record["texture_stamp_coverage_ratio"] >= target_config["min_coverage"]
```

Map configured `right_axis` and `up_axis` through `{"x": 0, "y": 1, "z": 2}` and require rendered anchor displacement to match config offset displacement within `1e-9`.

- [ ] **Step 5: Run tests and commit**

Run: `source .venv/bin/activate && pytest tests/test_perturbed_scene_dataset.py -v`

Expected: PASS.

```bash
git add src/tbp/compositional_datasets/perturbed_scene_dataset.py
git add -f tests/test_perturbed_scene_dataset.py
git commit -m "feat: validate perturbed object pairs"
```

### Task 3: Staged Batch Rendering

**Files:**
- Modify: `src/tbp/compositional_datasets/perturbed_scene_dataset.py`
- Modify: `tests/test_perturbed_scene_dataset.py`

**Interfaces:**
- Produces `append_perturbed_scene_objects(source_dataset, out_dir, *, source_start, count, id_offset, seed, bounds, stamping_script, blender_executable="blender", preview_texture_max_size=512) -> tuple[ObjectPair, ...]`.

- [ ] **Step 1: Write a fake-render happy-path test**

Monkeypatch `_run_render_command` with a callable that reads `--config`, writes `textured.glb`, `textured_stamped_texture.png`, and matching `textured.json`. Generate two pairs into a separate output and assert both target configs, object configs, meshes, normalized `textured.png` previews, and scene config exist.

- [ ] **Step 2: Write the failure-atomicity test**

Make the fake renderer raise on call two, run an in-place two-pair append, and assert none of the `201` config/object-config/mesh paths exist.

- [ ] **Step 3: Implement the staged workflow**

Use `tempfile.TemporaryDirectory(prefix=f".{out_dir.name}-perturbed-staging-", dir=out_dir.parent)`. For each ordered pair:

```python
source_config = read_json(pair.source_generation_config)
target_config = perturb_stamp_config(source_config, rng, bounds)
write_json(staged_generation_config, target_config)
write_json(staged_object_config, target_object_config(read_json(pair.source_object_config), pair.target_id))
source_metadata = read_json(pair.source_mesh_dir / "textured.json")
command = blender_stamp_command_args(
    blender_executable,
    stamping_script.resolve(),
    Path(source_metadata["parent_mesh_path"]),
    staged_generation_config,
    staged_mesh_dir / "textured.glb",
)
_run_render_command(command)
_normalize_texture_sidecar(staged_mesh_dir, preview_texture_max_size=preview_texture_max_size)
verify_rendered_pair(source_config, target_config, source_metadata, read_json(staged_mesh_dir / "textured.json"), bounds)
```

Reuse `_write_json`, `_run_render_command`, `_normalize_texture_sidecar`, and `scene_dataset_config` from `scene_dataset.py`. Only after every pair verifies, create final subdirectories and `shutil.move` all staged targets. Write the scene config only when the output lacks one; never rewrite the existing in-place scene config.

- [ ] **Step 4: Run regression tests and commit**

Run: `source .venv/bin/activate && pytest tests/test_perturbed_scene_dataset.py tests/test_scene_dataset_generation.py -v`

Expected: PASS.

```bash
git add src/tbp/compositional_datasets/perturbed_scene_dataset.py
git add -f tests/test_perturbed_scene_dataset.py
git commit -m "feat: stage and verify perturbed object pairs"
```

### Task 4: Append CLI

**Files:**
- Create: `scripts/append_perturbed_scene_objects.py`
- Create: `tests/test_append_perturbed_scene_objects.py`

**Interfaces:**
- Defaults: start `101`, count `100`, offset `100`, minimum `0.002`, maximum `0.006`, attempts `10_000`, preview size `512`.
- Requires `--source-dataset` and `--seed`; omitted `--out-dir` means in-place.

- [ ] **Step 1: Write failing parse/forward tests**

Assert the defaults above. Monkeypatch `append_perturbed_scene_objects`, call `main` with separate pilot output and `--count 5`, and assert it receives `PerturbationBounds(0.002, 0.006, 10_000)` and returns exit code 0.

- [ ] **Step 2: Implement the CLI**

The parser must expose:

```python
parser.add_argument("--source-dataset", required=True, type=Path)
parser.add_argument("--out-dir", type=Path, default=None)
parser.add_argument("--source-start", type=int, default=101)
parser.add_argument("--count", type=int, default=100)
parser.add_argument("--id-offset", type=int, default=100)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--min-displacement", type=float, default=0.002)
parser.add_argument("--max-displacement", type=float, default=0.006)
parser.add_argument("--max-attempts", type=int, default=10_000)
parser.add_argument("--configured-stamping-script", type=Path, default=ROOT / "scripts" / "stamp_object_from_config.py")
parser.add_argument("--blender", default="blender")
parser.add_argument("--preview-texture-max-size", type=int, default=512)
```

`main` constructs bounds, defaults output to source, forwards all arguments, prints `Installed {count} perturbed pairs: {first} through {last}`, and returns 0.

- [ ] **Step 3: Run tests and commit**

Run: `source .venv/bin/activate && pytest tests/test_append_perturbed_scene_objects.py tests/test_perturbed_scene_dataset.py -v`

Expected: PASS.

```bash
git add scripts/append_perturbed_scene_objects.py
git add -f tests/test_append_perturbed_scene_objects.py
git commit -m "feat: add perturbed pair append CLI"
```

### Task 5: Paired Vedo Viewer

**Files:**
- Modify: `scripts/visualize_model.py`
- Create: `tests/test_paired_visualize_model.py`

**Interfaces:**
- Produces `render_records(records: list[MeshRecord]) -> None` for one or two models.
- Preserves `render_record(record)` as a wrapper.

The installed Vedo API was verified to accept `Plotter(shape=(1, 2), sharecam=True, interactive=False)` and `Plotter.show(*objects, at=index, interactive=value)`.

- [ ] **Step 1: Write the non-GUI failing test**

Monkeypatch `Plotter`, `build_vedo_actor`, and `Text2D`. Call `render_records` with two named records and assert:

```python
assert constructor_kwargs["shape"] == (1, 2)
assert constructor_kwargs["sharecam"] is True
assert [call.kwargs["at"] for call in show_calls] == [0, 1]
assert show_calls[0].kwargs["interactive"] is False
assert show_calls[1].kwargs["interactive"] is True
```

- [ ] **Step 2: Implement shared rendering**

```python
def render_records(records: list[MeshRecord]) -> None:
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
```

Change positional `glb` to `nargs="+"`, accept exactly one or two existing paths, load them, print each summary, and call `render_records`. Keep `render_record` delegating to `[record]`.

- [ ] **Step 3: Run viewer tests and commit**

Run: `source .venv/bin/activate && pytest tests/test_paired_visualize_model.py tests/test_visualize_model.py -v`

Expected: PASS.

```bash
git add scripts/visualize_model.py
git add -f tests/test_paired_visualize_model.py
git commit -m "feat: compare paired objects in shared viewer"
```

### Task 6: Documentation and Automated Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the exact commands**

Add pilot:

```bash
source .venv/bin/activate
python scripts/append_perturbed_scene_objects.py \
  --source-dataset ~/tbp/data/compositional_objects_1.2 \
  --out-dir ~/tbp/data/compositional_objects_1.2_perturbation_pilot \
  --source-start 101 --count 5 --id-offset 100 --seed 123 \
  --min-displacement 0.002 --max-displacement 0.006
```

Add pair comparison:

```bash
python scripts/visualize_model.py \
  ~/tbp/data/compositional_objects_1.2/meshes/101_cube_6x2d_stickers/textured.glb \
  ~/tbp/data/compositional_objects_1.2_perturbation_pilot/meshes/201_cube_6x2d_stickers/textured.glb
```

Add production, explicitly noting it runs only after pilot approval:

```bash
python scripts/append_perturbed_scene_objects.py \
  --source-dataset ~/tbp/data/compositional_objects_1.2 \
  --source-start 101 --count 100 --id-offset 100 --seed 123 \
  --min-displacement 0.002 --max-displacement 0.006
```

- [ ] **Step 2: Run focused and full suites**

Run:

```bash
source .venv/bin/activate
pytest tests/test_perturbed_scene_dataset.py tests/test_append_perturbed_scene_objects.py tests/test_paired_visualize_model.py tests/test_visualize_model.py tests/test_scene_dataset_generation.py -v
pytest -q
git diff --check
```

Expected: all tests PASS and no whitespace errors.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs: document perturbed object pair workflow"
```

### Task 7: Five-Shape Pilot and Approval Gate

**Files:**
- External output: `~/tbp/data/compositional_objects_1.2_perturbation_pilot`

- [ ] **Step 1: Preflight without deleting anything**

Run: `test ! -e ~/tbp/data/compositional_objects_1.2_perturbation_pilot`

Expected: exit 0. If it exists, ask the user to choose a new path or explicitly authorize removal.

- [ ] **Step 2: Run the documented pilot command**

Run:

```bash
source .venv/bin/activate
python scripts/append_perturbed_scene_objects.py \
  --source-dataset ~/tbp/data/compositional_objects_1.2 \
  --out-dir ~/tbp/data/compositional_objects_1.2_perturbation_pilot \
  --source-start 101 --count 5 --id-offset 100 --seed 123 \
  --min-displacement 0.002 --max-displacement 0.006
```

Expected: `Installed 5 perturbed pairs: 201_cube_6x2d_stickers through 205_sphere_6x2d_stickers`.

- [ ] **Step 3: Verify output counts**

```bash
find ~/tbp/data/compositional_objects_1.2_perturbation_pilot/generation_configs -name '*.json' | wc -l
find ~/tbp/data/compositional_objects_1.2_perturbation_pilot/meshes -name textured.glb | wc -l
```

Expected: both print `5`.

- [ ] **Step 4: Review all five pairs**

Run the paired viewer for `101/201`, `102/202`, `103/203`, `104/204`, and `105/205`. Confirm perceptible motion, recognizable triangles, unchanged sticker identities, and unchanged rotations.

- [ ] **Step 5: Stop for explicit approval**

Ask whether `max_displacement=0.006` is approved. Do not begin Task 8 without approval. If stronger motion is requested, use a new pilot path and rerun with only `--max-displacement` changed.

### Task 8: Production Append and Final Verification

**Files:**
- External append: `~/tbp/data/compositional_objects_1.2/{generation_configs,configs,meshes}`

- [ ] **Step 1: Confirm target IDs are absent**

Run:

```bash
for number in {201..300}; do
  test -z "$(find ~/tbp/data/compositional_objects_1.2/generation_configs -maxdepth 1 -type f -name "${number}_*_6x2d_stickers.json" -print -quit)"
  test -z "$(find ~/tbp/data/compositional_objects_1.2/meshes -maxdepth 1 -type d -name "${number}_*_6x2d_stickers" -print -quit)"
done
```

Expected: both exit 0 with no output. Stop on any match; do not overwrite.

- [ ] **Step 2: Run the production command with the approved maximum**

Run this after `0.006` is approved:

```bash
source .venv/bin/activate
python scripts/append_perturbed_scene_objects.py \
  --source-dataset ~/tbp/data/compositional_objects_1.2 \
  --source-start 101 --count 100 --id-offset 100 --seed 123 \
  --min-displacement 0.002 --max-displacement 0.006
```

If the pilot approves another maximum, replace only `0.006` with the exact approved value.

Expected: `Installed 100 perturbed pairs: 201_cube_6x2d_stickers through 300_sphere_6x2d_stickers`.

- [ ] **Step 3: Verify counts and endpoints**

```bash
find ~/tbp/data/compositional_objects_1.2/generation_configs -maxdepth 1 -type f -name '*_6x2d_stickers.json' | wc -l
find ~/tbp/data/compositional_objects_1.2/meshes -maxdepth 1 -type d -name '*_6x2d_stickers' | wc -l
test -f ~/tbp/data/compositional_objects_1.2/generation_configs/201_cube_6x2d_stickers.json
test -f ~/tbp/data/compositional_objects_1.2/generation_configs/300_sphere_6x2d_stickers.json
test -f ~/tbp/data/compositional_objects_1.2/meshes/201_cube_6x2d_stickers/textured.glb
test -f ~/tbp/data/compositional_objects_1.2/meshes/300_sphere_6x2d_stickers/textured.glb
```

Expected: both counts print `200`; endpoint checks exit 0.

- [ ] **Step 4: Run final verification**

Run: `source .venv/bin/activate && pytest -q && git status --short && git log -5 --oneline`

Expected: all tests PASS; only pre-existing `.DS_Store` files remain untracked; all implementation changes are committed with Conventional Commits.

- [ ] **Step 5: Report exact evidence**

Report approved displacement range, seed, pair range, file counts, test result, and preservation of IDs `001`–`200`. Do not claim completion if any render or metadata check failed.
