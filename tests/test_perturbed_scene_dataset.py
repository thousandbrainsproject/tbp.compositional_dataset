from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
from typing import Any

import pytest

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
        "min_coverage": 0.9,
        "gap": 0.004,
        "max_longest_side": 0.021,
        "placement": {"right_axis": "x", "up_axis": "z", "front_axis": "y"},
        "slots": slots,
    }


@pytest.fixture
def miniature_dataset(tmp_path: Path) -> Path:
    """Build a two-object scene dataset with all required source artifacts."""
    dataset = tmp_path / "dataset"
    for source_id in ("101_cube_6x2d_stickers", "102_cylinder_6x2d_stickers"):
        generation_config = dataset / "generation_configs" / f"{source_id}.json"
        object_config = dataset / "configs" / f"{source_id}.object_config.json"
        mesh_dir = dataset / "meshes" / source_id
        generation_config.parent.mkdir(parents=True, exist_ok=True)
        object_config.parent.mkdir(parents=True, exist_ok=True)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        generation_config.write_text("{}")
        object_config.write_text(
            json.dumps({"render_asset": f"../meshes/{source_id}/textured.glb"})
        )
        (mesh_dir / "textured.glb").write_bytes(b"glTF")
        (mesh_dir / "textured.json").write_text("{}")
    return dataset


def test_target_id_adds_100_and_preserves_suffix():
    """Verify numeric ID offsets leave object-name suffixes unchanged."""
    assert target_object_id("101_cube_6x2d_stickers", 100) == "201_cube_6x2d_stickers"
    assert target_object_id("200_sphere_6x2d_stickers", 100) == "300_sphere_6x2d_stickers"


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
        PerturbationBounds(),
    )


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
            PerturbationBounds(),
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
            PerturbationBounds(),
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
            PerturbationBounds(),
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
            PerturbationBounds(),
        )


def test_seeded_perturbation_is_deterministic_and_changes_only_offsets(source_config):
    """Verify seeded perturbations are bounded and change only offsets."""
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
    """Verify impossible spacing reports the exhausted attempt limit."""
    source_config["max_longest_side"] = 1.0
    bounds = PerturbationBounds(0.002, 0.006, max_attempts=3)
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
        validate_perturbed_config(source_config, target, PerturbationBounds())


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
        validate_perturbed_config(source_config, target, PerturbationBounds())
