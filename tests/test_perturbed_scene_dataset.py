from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
from typing import Any

import pytest
from PIL import Image

from tbp.compositional_datasets import perturbed_scene_dataset
from tbp.compositional_datasets.perturbed_scene_dataset import (
    ObjectPair,
    PerturbationBounds,
    discover_object_pairs,
    perturb_stamp_config,
    target_object_config,
    target_object_id,
    validate_perturbed_config,
    verify_rendered_pair,
)

# Historical initial-pilot range retained only for exact regression fixtures.
INITIAL_PILOT_MIN_DISPLACEMENT = 0.002
INITIAL_PILOT_MAX_DISPLACEMENT = 0.006
INITIAL_PILOT_BOUNDS = PerturbationBounds(
    INITIAL_PILOT_MIN_DISPLACEMENT,
    INITIAL_PILOT_MAX_DISPLACEMENT,
)


@pytest.fixture
def source_config() -> dict[str, Any]:
    """Return a six-slot sticker layout on the front and back sides."""
    slots = []
    for side in ("front", "back"):
        slots.extend(
            [
                {"name": f"{side}_top", "side": side, "offset": [0.0, 0.018]},
                {"name": f"{side}_left", "side": side, "offset": [-0.018, -0.012]},
                {"name": f"{side}_right", "side": side, "offset": [0.018, -0.012]},
            ]
        )
    return {
        "min_coverage": 0.95,
        "gap": 0.004,
        "max_longest_side": 0.021,
        "placement": {"right_axis": "x", "up_axis": "z", "front_axis": "y"},
        "slots": slots,
    }


@pytest.fixture
def miniature_dataset(tmp_path: Path, source_config: dict[str, Any]) -> Path:
    """Build a two-object scene dataset with all required source artifacts."""
    dataset = tmp_path / "dataset"
    for source_id in ("101_cube_6x2d_stickers", "102_cylinder_6x2d_stickers"):
        generation_config = dataset / "generation_configs" / f"{source_id}.json"
        object_config = dataset / "configs" / f"{source_id}.object_config.json"
        mesh_dir = dataset / "meshes" / source_id
        generation_config.parent.mkdir(parents=True, exist_ok=True)
        object_config.parent.mkdir(parents=True, exist_ok=True)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        generation_config.write_text(json.dumps(source_config))
        object_config.write_text(
            json.dumps({"render_asset": f"../meshes/{source_id}/textured.glb"})
        )
        (mesh_dir / "textured.glb").write_bytes(b"glTF")
        (mesh_dir / "textured.json").write_text(
            json.dumps(_rendered_metadata(source_config))
        )
    return dataset


def test_target_id_adds_100_and_preserves_suffix():
    """Verify generic numeric ID offsets leave object-name suffixes unchanged."""
    assert target_object_id("101_cube_6x2d_stickers", 100) == "201_cube_6x2d_stickers"
    assert target_object_id("200_sphere_6x2d_stickers", 100) == "300_sphere_6x2d_stickers"
    assert target_object_id("101_cube_6x2d_stickers", 7) == "108_cube_6x2d_stickers"


def test_perturbation_bounds_use_human_approved_calibration_defaults() -> None:
    """Use the approved displacement interval for production sampling."""
    assert PerturbationBounds() == PerturbationBounds(0.008, 0.016, 10_000)


def test_discovery_returns_complete_source_target_pairs(miniature_dataset: Path) -> None:
    """Verify discovery maps consecutive complete sources to offset targets."""
    pairs = discover_object_pairs(
        miniature_dataset,
        miniature_dataset,
        source_start=101,
        count=2,
        id_offset=100,
    )

    assert [pair.target_id for pair in pairs] == [
        "201_cube_6x2d_stickers",
        "202_cylinder_6x2d_stickers",
    ]
    assert pairs[0] == ObjectPair(
        source_id="101_cube_6x2d_stickers",
        target_id="201_cube_6x2d_stickers",
        source_generation_config=miniature_dataset
        / "generation_configs"
        / "101_cube_6x2d_stickers.json",
        source_object_config=miniature_dataset
        / "configs"
        / "101_cube_6x2d_stickers.object_config.json",
        source_mesh_dir=miniature_dataset / "meshes" / "101_cube_6x2d_stickers",
        target_generation_config=miniature_dataset
        / "generation_configs"
        / "201_cube_6x2d_stickers.json",
        target_object_config=miniature_dataset
        / "configs"
        / "201_cube_6x2d_stickers.object_config.json",
        target_mesh_dir=miniature_dataset / "meshes" / "201_cube_6x2d_stickers",
    )


def test_discovery_rejects_missing_requested_source(miniature_dataset: Path) -> None:
    """Verify discovery identifies the missing numeric source prefix."""
    with pytest.raises(FileNotFoundError, match="source object 103"):
        discover_object_pairs(
            miniature_dataset,
            miniature_dataset,
            source_start=101,
            count=3,
            id_offset=100,
        )


def test_discovery_rejects_existing_target_before_returning_pairs(
    miniature_dataset: Path,
) -> None:
    """Verify preflight rejects an existing target artifact."""
    target_path = (
        miniature_dataset / "generation_configs" / "201_cube_6x2d_stickers.json"
    )
    target_path.write_text("{}")

    with pytest.raises(FileExistsError, match="201_cube_6x2d_stickers"):
        discover_object_pairs(
            miniature_dataset,
            miniature_dataset,
            source_start=101,
            count=1,
            id_offset=100,
        )


@pytest.mark.parametrize(
    ("source_start", "count", "id_offset"),
    [
        (100, 1, 100),
        (200, 2, 100),
        (101, 1, 99),
    ],
)
def test_discovery_rejects_ids_outside_protected_pair_domain(
    miniature_dataset: Path,
    source_start: int,
    count: int,
    id_offset: int,
) -> None:
    """Verify discovery reserves sources 101-200 and targets 201-300."""
    with pytest.raises(ValueError, match="protected object ID domain"):
        discover_object_pairs(
            miniature_dataset,
            miniature_dataset,
            source_start=source_start,
            count=count,
            id_offset=id_offset,
        )


def test_target_object_config_changes_only_render_asset() -> None:
    """Verify target Habitat config creation does not mutate its source."""
    source = {
        "render_asset": "../meshes/101_cube_6x2d_stickers/textured.glb",
        "nested": {"values": [1, 2]},
    }

    target = target_object_config(source, "201_cube_6x2d_stickers")

    assert target == {
        "render_asset": "../meshes/201_cube_6x2d_stickers/textured.glb",
        "nested": {"values": [1, 2]},
    }
    assert source["render_asset"] == "../meshes/101_cube_6x2d_stickers/textured.glb"
    assert target["nested"] is not source["nested"]


def _rendered_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Build rendered metadata whose anchors match configured x/z offsets."""
    return {
        "parent_mesh_path": "assets/parents/cube.glb",
        "stickers": [
            {
                "slot_name": slot["name"],
                "side": slot["side"],
                "sticker_path": f"assets/stickers/{slot['name']}.png",
                "rotation_deg": 15.0,
                "size": [0.01, 0.02],
                "world_space_anchor_point": [
                    slot["offset"][0],
                    0.0,
                    slot["offset"][1],
                ],
                "texture_stamp_coverage_ratio": 1.0,
            }
            for slot in config["slots"]
        ],
    }


def _fake_render(command: list[str]) -> None:
    """Write fake render outputs matching the command's stamp config.

    Args:
        command: Blender-style command arguments containing config and output paths.
    """
    config_path = Path(command[command.index("--config") + 1])
    output_path = Path(command[command.index("--out") + 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"glTF")
    Image.new("RGBA", (1024, 256), (255, 255, 255, 255)).save(
        output_path.with_name("textured_stamped_texture.png")
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(_rendered_metadata(json.loads(config_path.read_text())))
    )


def test_append_renders_verifies_and_installs_complete_batch(
    miniature_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a staged batch installs every artifact after all renders pass."""
    monkeypatch.setattr(
        "tbp.compositional_datasets.perturbed_scene_dataset._run_render_command",
        _fake_render,
    )
    out_dir = tmp_path / "perturbed"

    pairs = perturbed_scene_dataset.append_perturbed_scene_objects(
        miniature_dataset,
        out_dir,
        source_start=101,
        count=2,
        id_offset=100,
        seed=17,
        bounds=PerturbationBounds(),
        stamping_script=Path("scripts/stamp_object_from_config.py"),
    )

    assert [pair.target_id for pair in pairs] == [
        "201_cube_6x2d_stickers",
        "202_cylinder_6x2d_stickers",
    ]
    for pair in pairs:
        assert pair.target_generation_config.exists()
        assert pair.target_object_config.exists()
        assert (pair.target_mesh_dir / "textured.glb").exists()
        assert (pair.target_mesh_dir / "textured.json").exists()
        preview_path = pair.target_mesh_dir / "textured.png"
        assert preview_path.exists()
        with Image.open(preview_path) as image:
            assert image.size == (512, 128)
    assert (out_dir / "compositional_objects.scene_dataset_config.json").exists()


def test_append_library_defaults_use_approved_seed_and_bounds(
    miniature_dataset: Path,
    source_config: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use seed 123 and calibrated bounds when library callers omit both."""
    monkeypatch.setattr(
        "tbp.compositional_datasets.perturbed_scene_dataset._run_render_command",
        _fake_render,
    )
    out_dir = tmp_path / "approved-defaults"

    pairs = perturbed_scene_dataset.append_perturbed_scene_objects(
        miniature_dataset,
        out_dir,
        source_start=101,
        count=1,
        id_offset=100,
        stamping_script=Path("scripts/stamp_object_from_config.py"),
    )

    target_config = json.loads(pairs[0].target_generation_config.read_text())
    expected_config = perturb_stamp_config(
        source_config,
        random.Random(123),
        PerturbationBounds(),
    )
    assert target_config == expected_config
    for source_slot, target_slot in zip(
        source_config["slots"], target_config["slots"], strict=True
    ):
        assert 0.008 <= math.dist(
            source_slot["offset"], target_slot["offset"]
        ) <= 0.016


def test_append_render_failure_installs_no_targets_or_rewrites_scene_config(
    miniature_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a staged render failure leaves the destination unchanged."""
    calls = 0

    def fail_on_second_render(command: list[str]) -> None:
        """Render the first target and fail before writing the second.

        Args:
            command: Blender-style command arguments.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("render failed")
        _fake_render(command)

    monkeypatch.setattr(
        "tbp.compositional_datasets.perturbed_scene_dataset._run_render_command",
        fail_on_second_render,
    )
    scene_config_path = (
        miniature_dataset / "compositional_objects.scene_dataset_config.json"
    )
    scene_config_path.write_text('{"existing": true}\n')

    with pytest.raises(RuntimeError, match="render failed"):
        perturbed_scene_dataset.append_perturbed_scene_objects(
            miniature_dataset,
            miniature_dataset,
            source_start=101,
            count=2,
            id_offset=100,
            seed=17,
            bounds=PerturbationBounds(),
            stamping_script=Path("scripts/stamp_object_from_config.py"),
        )

    for target_id in (
        "201_cube_6x2d_stickers",
        "202_cylinder_6x2d_stickers",
    ):
        assert not (
            miniature_dataset / "generation_configs" / f"{target_id}.json"
        ).exists()
        assert not (
            miniature_dataset / "configs" / f"{target_id}.object_config.json"
        ).exists()
        assert not (miniature_dataset / "meshes" / target_id).exists()
    assert scene_config_path.read_text() == '{"existing": true}\n'


def test_append_missing_rendered_glb_installs_no_target_artifacts(
    miniature_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a zero-exit render missing its GLB cannot install a target."""

    def render_without_glb(command: list[str]) -> None:
        """Write render sidecars but remove the required GLB.

        Args:
            command: Blender-style command arguments.
        """
        _fake_render(command)
        output_path = Path(command[command.index("--out") + 1])
        output_path.unlink()

    monkeypatch.setattr(
        "tbp.compositional_datasets.perturbed_scene_dataset._run_render_command",
        render_without_glb,
    )
    out_dir = tmp_path / "perturbed"

    with pytest.raises(RuntimeError, match="without writing expected GLB"):
        perturbed_scene_dataset.append_perturbed_scene_objects(
            miniature_dataset,
            out_dir,
            source_start=101,
            count=1,
            id_offset=100,
            seed=17,
            bounds=PerturbationBounds(),
            stamping_script=Path("scripts/stamp_object_from_config.py"),
        )

    assert not (
        out_dir / "generation_configs" / "201_cube_6x2d_stickers.json"
    ).exists()
    assert not (
        out_dir / "configs" / "201_cube_6x2d_stickers.object_config.json"
    ).exists()
    assert not (out_dir / "meshes" / "201_cube_6x2d_stickers").exists()


def test_append_invalid_rendered_metadata_installs_no_requested_targets(
    miniature_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify written render outputs remain staged when metadata verification fails."""
    render_calls = 0

    def render_with_invalid_metadata(command: list[str]) -> None:
        """Write complete outputs, corrupting the second target's metadata.

        Args:
            command: Blender-style command arguments.
        """
        nonlocal render_calls
        render_calls += 1
        _fake_render(command)
        if render_calls != 2:
            return
        output_path = Path(command[command.index("--out") + 1])
        metadata_path = output_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["parent_mesh_path"] = "assets/parents/unexpected.glb"
        metadata_path.write_text(json.dumps(metadata))

    monkeypatch.setattr(
        "tbp.compositional_datasets.perturbed_scene_dataset._run_render_command",
        render_with_invalid_metadata,
    )
    out_dir = tmp_path / "perturbed"

    with pytest.raises(ValueError, match="rendered parent mesh changed"):
        perturbed_scene_dataset.append_perturbed_scene_objects(
            miniature_dataset,
            out_dir,
            source_start=101,
            count=2,
            id_offset=100,
            seed=17,
            bounds=PerturbationBounds(),
            stamping_script=Path("scripts/stamp_object_from_config.py"),
        )

    assert render_calls == 2
    for target_id in ("201_cube_6x2d_stickers", "202_cylinder_6x2d_stickers"):
        assert not (out_dir / "generation_configs" / f"{target_id}.json").exists()
        assert not (
            out_dir / "configs" / f"{target_id}.object_config.json"
        ).exists()
        assert not (out_dir / "meshes" / target_id).exists()


def test_append_rejects_low_coverage_before_render_or_install(
    miniature_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify source coverage below 0.95 is rejected before rendering."""
    source_config_path = (
        miniature_dataset
        / "generation_configs"
        / "101_cube_6x2d_stickers.json"
    )
    source_config = json.loads(source_config_path.read_text())
    source_config["min_coverage"] = 0.9
    source_config_path.write_text(json.dumps(source_config))
    renderer_calls: list[list[str]] = []

    def record_render(command: list[str]) -> None:
        """Record an unexpected Blender render command.

        Args:
            command: Blender-style command arguments.
        """
        renderer_calls.append(command)

    monkeypatch.setattr(
        "tbp.compositional_datasets.perturbed_scene_dataset._run_render_command",
        record_render,
    )
    out_dir = tmp_path / "perturbed"

    with pytest.raises(ValueError, match="min_coverage must be at least 0.95"):
        perturbed_scene_dataset.append_perturbed_scene_objects(
            miniature_dataset,
            out_dir,
            source_start=101,
            count=1,
            id_offset=100,
            seed=17,
            bounds=PerturbationBounds(),
            stamping_script=Path("scripts/stamp_object_from_config.py"),
        )

    assert renderer_calls == []
    assert not (
        out_dir / "generation_configs" / "201_cube_6x2d_stickers.json"
    ).exists()
    assert not (
        out_dir / "configs" / "201_cube_6x2d_stickers.object_config.json"
    ).exists()
    assert not (out_dir / "meshes" / "201_cube_6x2d_stickers").exists()


def test_matching_rendered_pair_passes_verification(source_config: dict[str, Any]) -> None:
    """Verify paired metadata preserves layout semantics and displacement."""
    target_config = deepcopy(source_config)
    for slot in target_config["slots"]:
        slot["offset"][0] += 0.002

    verify_rendered_pair(
        source_config,
        target_config,
        _rendered_metadata(source_config),
        _rendered_metadata(target_config),
        INITIAL_PILOT_BOUNDS,
    )


def test_rendered_anchor_provenance_allows_independent_float_noise(
    source_config: dict[str, Any],
) -> None:
    """Allow realistic source/target anchor noise without losing signed provenance."""
    target_config = deepcopy(source_config)
    for slot in target_config["slots"]:
        slot["offset"][0] += 0.002
        slot["offset"][1] -= 0.002
    source_metadata = _rendered_metadata(source_config)
    target_metadata = _rendered_metadata(target_config)
    for source_record, target_record in zip(
        source_metadata["stickers"], target_metadata["stickers"], strict=True
    ):
        source_record["world_space_anchor_point"][0] += 5.5e-8
        target_record["world_space_anchor_point"][0] -= 5.5e-8
        source_record["world_space_anchor_point"][2] -= 5e-8
        target_record["world_space_anchor_point"][2] += 5e-8

    verify_rendered_pair(
        source_config,
        target_config,
        source_metadata,
        target_metadata,
        INITIAL_PILOT_BOUNDS,
    )


def test_rendered_anchor_provenance_rejects_signed_component_mismatch(
    source_config: dict[str, Any],
) -> None:
    """Reject and identify a signed component mismatch beyond float tolerance."""
    target_config = deepcopy(source_config)
    for slot in target_config["slots"]:
        slot["offset"][0] += 0.002
    source_metadata = _rendered_metadata(source_config)
    target_metadata = _rendered_metadata(target_config)
    target_metadata["stickers"][0]["world_space_anchor_point"][0] += 1.1e-6

    with pytest.raises(ValueError) as exc_info:
        verify_rendered_pair(
            source_config,
            target_config,
            source_metadata,
            target_metadata,
            INITIAL_PILOT_BOUNDS,
        )

    message = str(exc_info.value)
    assert "slot front_top" in message
    assert "component right" in message
    assert "expected 0.002" in message
    assert "actual 0.0020011" in message


def test_verification_rejects_changed_rendered_sticker_size(
    source_config: dict[str, Any],
) -> None:
    """Verify paired renders cannot change physical sticker size."""
    target_config = deepcopy(source_config)
    for slot in target_config["slots"]:
        slot["offset"][0] += 0.002
    source_metadata = _rendered_metadata(source_config)
    target_metadata = _rendered_metadata(target_config)
    target_metadata["stickers"][0]["size"][0] += 0.001

    with pytest.raises(ValueError, match="rendered sticker size changed"):
        verify_rendered_pair(
            source_config,
            target_config,
            source_metadata,
            target_metadata,
            INITIAL_PILOT_BOUNDS,
        )


@pytest.mark.parametrize("metadata_name", ["source", "target"])
@pytest.mark.parametrize("invalid_record", ["duplicate", "extra"])
def test_verification_rejects_duplicate_or_extra_rendered_records(
    source_config: dict[str, Any],
    metadata_name: str,
    invalid_record: str,
) -> None:
    """Verify both rendered metadata lists contain each configured slot once."""
    target_config = deepcopy(source_config)
    for slot in target_config["slots"]:
        slot["offset"][0] += 0.002
    source_metadata = _rendered_metadata(source_config)
    target_metadata = _rendered_metadata(target_config)
    metadata = source_metadata if metadata_name == "source" else target_metadata
    added_record = deepcopy(metadata["stickers"][0])
    if invalid_record == "extra":
        added_record["slot_name"] = "unexpected_slot"
    metadata["stickers"].append(added_record)

    with pytest.raises(
        ValueError, match="rendered sticker slots do not match configuration"
    ):
        verify_rendered_pair(
            source_config,
            target_config,
            source_metadata,
            target_metadata,
            INITIAL_PILOT_BOUNDS,
        )


def test_verification_reports_first_configured_slot_failure(
    source_config: dict[str, Any],
) -> None:
    """Verify multiple metadata failures report in configured slot order."""
    target_config = deepcopy(source_config)
    for slot in target_config["slots"]:
        slot["offset"][0] += 0.002
    source_metadata = _rendered_metadata(source_config)
    target_metadata = _rendered_metadata(target_config)
    target_metadata["stickers"][0]["side"] = "back"
    for record in target_metadata["stickers"][1:]:
        record["sticker_path"] = "assets/stickers/wrong.png"

    with pytest.raises(ValueError, match="rendered sticker side changed"):
        verify_rendered_pair(
            source_config,
            target_config,
            source_metadata,
            target_metadata,
            INITIAL_PILOT_BOUNDS,
        )


@pytest.mark.parametrize(
    ("mismatch", "expected_error"),
    [
        ("parent", "rendered parent mesh changed"),
        ("sticker", "rendered sticker path changed"),
        ("side", "rendered sticker side changed"),
        ("rotation", "rendered sticker rotation changed"),
        ("coverage", "rendered sticker coverage is below minimum"),
        ("anchor", "rendered anchor displacement changed"),
    ],
)
def test_verification_rejects_rendered_integrity_mismatch(
    source_config: dict[str, Any],
    mismatch: str,
    expected_error: str,
) -> None:
    """Verify every required rendered-pair invariant rejects a mismatch."""
    target_config = deepcopy(source_config)
    for slot in target_config["slots"]:
        slot["offset"][0] += 0.002
    source_metadata = _rendered_metadata(source_config)
    target_metadata = _rendered_metadata(target_config)
    target_record = target_metadata["stickers"][0]
    if mismatch == "parent":
        target_metadata["parent_mesh_path"] = "assets/parents/other.glb"
    elif mismatch == "sticker":
        target_record["sticker_path"] = "assets/stickers/other.png"
    elif mismatch == "side":
        target_record["side"] = "back"
    elif mismatch == "rotation":
        target_record["rotation_deg"] += 1.0
    elif mismatch == "coverage":
        target_record["texture_stamp_coverage_ratio"] = 0.89
    else:
        source_anchor = source_metadata["stickers"][0]["world_space_anchor_point"]
        target_record["world_space_anchor_point"][0] = source_anchor[0] - 0.002

    with pytest.raises(ValueError, match=expected_error):
        verify_rendered_pair(
            source_config,
            target_config,
            source_metadata,
            target_metadata,
            INITIAL_PILOT_BOUNDS,
        )


def test_approved_default_sampling_is_deterministic_and_changes_only_offsets(
    source_config: dict[str, Any],
) -> None:
    """Sample reproducibly with approved seed 123 and default radial bounds."""
    bounds = PerturbationBounds()
    first = perturb_stamp_config(source_config, random.Random(123), bounds)
    second = perturb_stamp_config(source_config, random.Random(123), bounds)
    assert first == second
    for source_slot, target_slot in zip(source_config["slots"], first["slots"], strict=True):
        assert {k: v for k, v in source_slot.items() if k != "offset"} == {
            k: v for k, v in target_slot.items() if k != "offset"
        }
        assert 0.008 <= math.dist(
            source_slot["offset"], target_slot["offset"]
        ) <= 0.016
    validate_perturbed_config(source_config, first, bounds)


def _candidate_at_attempt(
    source: dict[str, Any], seed: int, attempt: int, bounds: PerturbationBounds
) -> dict[str, Any]:
    """Return the requested deterministic perturbation candidate.

    Args:
        source: Original sticker layout configuration.
        seed: Random seed used for candidate sampling.
        attempt: One-based candidate attempt to return.
        bounds: Radial displacement sampling bounds.

    Returns:
        Deep-copied configuration containing the sampled offsets.
    """
    rng = random.Random(seed)
    candidate = deepcopy(source)
    for _ in range(attempt):
        candidate = deepcopy(source)
        for slot in candidate["slots"]:
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.uniform(bounds.min_displacement, bounds.max_displacement)
            slot["offset"][0] += radius * math.cos(angle)
            slot["offset"][1] += radius * math.sin(angle)
    return candidate


def test_initial_pilot_seed_123_first_candidate_rejects_footprint_overlap(
    source_config: dict[str, Any],
) -> None:
    """Reject the historical initial-pilot candidate despite center spacing."""
    rotations = (75.0, 60.0, 150.0, 255.0, 150.0, 330.0)
    for slot, rotation in zip(source_config["slots"], rotations, strict=True):
        slot["rotation_deg"] = rotation
    bounds = INITIAL_PILOT_BOUNDS
    candidate = _candidate_at_attempt(source_config, 123, 1, bounds)
    back_top, _back_left, back_right = candidate["slots"][3:]

    assert math.dist(back_top["offset"], back_right["offset"]) == pytest.approx(
        0.0328261621633871
    )
    with pytest.raises(
        ValueError,
        match="sticker footprints overlap: back_top and back_right",
    ):
        validate_perturbed_config(source_config, candidate, bounds)


def test_initial_pilot_seed_123_resamples_to_second_footprint_candidate(
    source_config: dict[str, Any],
) -> None:
    """Reproduce the historical initial pilot's valid second candidate."""
    rotations = (75.0, 60.0, 150.0, 255.0, 150.0, 330.0)
    for slot, rotation in zip(source_config["slots"], rotations, strict=True):
        slot["rotation_deg"] = rotation
    bounds = INITIAL_PILOT_BOUNDS
    expected = _candidate_at_attempt(source_config, 123, 2, bounds)

    first = perturb_stamp_config(source_config, random.Random(123), bounds)
    second = perturb_stamp_config(source_config, random.Random(123), bounds)

    assert first == second == expected
    for source_slot, target_slot in zip(
        source_config["slots"], first["slots"], strict=True
    ):
        assert bounds.min_displacement <= math.dist(
            source_slot["offset"], target_slot["offset"]
        ) <= bounds.max_displacement


def test_projected_footprint_boundary_contact_is_not_overlap(
    source_config: dict[str, Any],
) -> None:
    """Allow rotated conservative square bounds that only touch at an edge."""
    half_extent = 0.021 / math.sqrt(2.0)
    for slot in source_config["slots"]:
        slot["rotation_deg"] = 45.0
        position = slot["name"].rsplit("_", maxsplit=1)[-1]
        if position == "left":
            slot["offset"][0] = -half_extent
        elif position == "right":
            slot["offset"][0] = half_extent
    target = deepcopy(source_config)
    for slot in target["slots"]:
        slot["offset"][0] += 0.002

    validate_perturbed_config(
        source_config, target, INITIAL_PILOT_BOUNDS
    )


def test_impossible_spacing_reports_attempt_limit(source_config):
    """Verify impossible spacing reports the exhausted attempt limit."""
    source_config["max_longest_side"] = 1.0
    bounds = PerturbationBounds(
        INITIAL_PILOT_MIN_DISPLACEMENT,
        INITIAL_PILOT_MAX_DISPLACEMENT,
        max_attempts=3,
    )
    with pytest.raises(ValueError, match="after 3 attempts"):
        perturb_stamp_config(source_config, random.Random(7), bounds)


def test_spacing_enforces_absolute_floor(source_config):
    """Verify same-side spacing cannot fall below the absolute floor."""
    source_config["max_longest_side"] = 0.01
    source_config["gap"] = 0.001
    target = deepcopy(source_config)
    for slot in target["slots"]:
        slot["offset"][0] += 0.002
    target["slots"][1]["offset"][0] = -0.012
    target["slots"][2]["offset"][0] = 0.012

    with pytest.raises(ValueError, match="slots overlap on side front"):
        validate_perturbed_config(
            source_config, target, INITIAL_PILOT_BOUNDS
        )


def test_validation_rejects_top_below_a_lower_slot(source_config):
    """Reject a target whose top slot is no longer above both lower slots."""
    target = deepcopy(source_config)
    for slot in target["slots"]:
        slot["offset"][0] += 0.002
    target["slots"][0]["offset"][1] = -0.020

    with pytest.raises(ValueError, match="top slot is not above both lower slots"):
        validate_perturbed_config(
            source_config, target, PerturbationBounds(0.001, 0.1)
        )


def test_validation_rejects_left_slot_to_right_of_right_slot(source_config):
    """Reject a target whose signed horizontal slot order is reversed."""
    target = deepcopy(source_config)
    for slot in target["slots"]:
        slot["offset"][0] += 0.002
    target["slots"][1]["offset"][0] = 0.020
    target["slots"][2]["offset"][0] = -0.020

    with pytest.raises(ValueError, match="left slot is not left of right slot"):
        validate_perturbed_config(
            source_config, target, PerturbationBounds(0.001, 0.1)
        )


def test_validation_rejects_triangle_winding_reversal(source_config):
    """Reject a target triangle with the opposite source winding."""
    source_config["slots"][0]["offset"][1] = -0.020
    target = deepcopy(source_config)
    for slot in target["slots"]:
        slot["offset"][0] += 0.002
    target["slots"][0]["offset"][1] = 0.018

    with pytest.raises(ValueError, match="triangle orientation changed"):
        validate_perturbed_config(
            source_config, target, PerturbationBounds(0.001, 0.1)
        )


def test_perturbation_resamples_after_a_rejected_candidate(
    source_config: dict[str, Any],
) -> None:
    """Retry with a second sample after the first violates top ordering."""

    class ControlledRandom:
        """Return one invalid candidate followed by a valid translated candidate."""

        def __init__(self) -> None:
            """Prepare angle/radius values for two six-slot candidates."""
            invalid_candidate = [3.0 * math.pi / 2.0, 0.04]
            invalid_candidate.extend([0.0, 0.002] * 5)
            valid_candidate = [0.0, 0.002] * 6
            self.values = iter(invalid_candidate + valid_candidate)

        def uniform(self, start: float, end: float) -> float:
            """Return the next controlled value.

            Args:
                start: Requested lower sampling bound.
                end: Requested upper sampling bound.

            Returns:
                Next angle or radius in the controlled sequence.
            """
            del start, end
            return next(self.values)

    result = perturb_stamp_config(
        source_config,
        ControlledRandom(),  # type: ignore[arg-type]
        PerturbationBounds(0.002, 0.1, max_attempts=2),
    )

    for source_slot, target_slot in zip(
        source_config["slots"], result["slots"], strict=True
    ):
        assert target_slot["offset"] == pytest.approx(
            [source_slot["offset"][0] + 0.002, source_slot["offset"][1]]
        )


def test_validation_reports_front_before_back_when_both_are_invalid(source_config):
    """Verify invalid sides are validated in stable front-before-back order."""
    target = deepcopy(source_config)
    for slot in target["slots"]:
        position = slot["name"].rsplit("_", maxsplit=1)[-1]
        if position == "top":
            slot["offset"][0] += 0.002
        elif position == "left":
            slot["offset"][0] += 0.006
        else:
            slot["offset"][0] -= 0.006

    with pytest.raises(ValueError, match="slots overlap on side front"):
        validate_perturbed_config(
            source_config, target, INITIAL_PILOT_BOUNDS
        )
