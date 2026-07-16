from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tbp.compositional_datasets.perturbed_scene_dataset import PerturbationBounds


def load_script():
    """Load the append CLI module from its script path.

    Returns:
        Loaded append CLI module.
    """
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "append_perturbed_scene_objects.py"
    )
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location(
        "append_perturbed_scene_objects", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_perturbed_pair_defaults() -> None:
    """Parse the documented defaults for an in-place append."""
    script = load_script()

    args = script.parse_args(["--source-dataset", "dataset"])

    assert args.source_dataset == Path("dataset")
    assert args.out_dir is None
    assert args.source_start == 101
    assert args.count == 100
    assert args.id_offset == 100
    assert args.seed == 123
    assert args.min_displacement == 0.008
    assert args.max_displacement == 0.016
    assert args.max_attempts == 10_000
    assert args.configured_stamping_script == (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "stamp_object_from_config.py"
    )
    assert args.blender == "blender"
    assert args.preview_texture_max_size == 512


def test_main_forwards_pilot_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forward a separate five-object pilot and report installed targets."""
    script = load_script()
    calls = []

    def fake_append(
        source_dataset: Path,
        out_dir: Path,
        **kwargs,
    ):
        """Record forwarded arguments and return representative object pairs.

        Args:
            source_dataset: Forwarded source dataset path.
            out_dir: Forwarded output dataset path.
            **kwargs: Forwarded keyword arguments.

        Returns:
            Representative first and last installed object pairs.
        """
        calls.append((source_dataset, out_dir, kwargs))
        return (
            SimpleNamespace(target_id="201_cube_6x2d_stickers"),
            SimpleNamespace(target_id="205_sphere_6x2d_stickers"),
        )

    monkeypatch.setattr(script, "append_perturbed_scene_objects", fake_append)

    exit_code = script.main(
        [
            "--source-dataset",
            "source",
            "--out-dir",
            "pilot",
            "--count",
            "5",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            Path("source"),
            Path("pilot"),
            {
                "source_start": 101,
                "count": 5,
                "id_offset": 100,
                "seed": 123,
                "bounds": PerturbationBounds(0.008, 0.016, 10_000),
                "stamping_script": (
                    Path(__file__).resolve().parents[1]
                    / "scripts"
                    / "stamp_object_from_config.py"
                ),
                "blender_executable": "blender",
                "preview_texture_max_size": 512,
            },
        )
    ]
    assert capsys.readouterr().out == (
        "Installed 5 perturbed pairs: 201_cube_6x2d_stickers "
        "through 205_sphere_6x2d_stickers\n"
    )
