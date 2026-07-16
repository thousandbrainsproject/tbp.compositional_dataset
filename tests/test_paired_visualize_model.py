from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import call

import numpy as np
import pytest

from test_visualize_model import load_visualize_model_module


def test_render_records_shows_pair_in_shared_side_by_side_view(monkeypatch):
    """Render two records in shared-camera side-by-side viewports."""
    visualize_model = load_visualize_model_module()
    constructor_kwargs = {}
    show_calls = []
    close_calls = []

    class FakePlotter:
        """Record Plotter construction and display calls without opening a GUI."""

        def __init__(self, **kwargs) -> None:
            """Capture Plotter construction.

            Args:
                **kwargs: Plotter constructor keyword arguments.
            """
            constructor_kwargs.update(kwargs)

        def show(self, *objects, **kwargs) -> None:
            """Capture a viewport display call.

            Args:
                *objects: Vedo objects to display.
                **kwargs: Plotter display options.
            """
            show_calls.append(call(*objects, **kwargs))

        def close(self) -> None:
            """Close the fake plotter without opening a GUI."""
            close_calls.append(call())

    monkeypatch.setattr(visualize_model, "Plotter", FakePlotter)
    monkeypatch.setattr(
        visualize_model,
        "build_vedo_actor",
        lambda record: f"actor:{record.name}",
    )
    monkeypatch.setattr(
        visualize_model,
        "Text2D",
        lambda name, **kwargs: (name, kwargs),
    )
    records = [SimpleNamespace(name="source"), SimpleNamespace(name="target")]

    visualize_model.render_records(records)

    assert constructor_kwargs["shape"] == (1, 2)
    assert constructor_kwargs["sharecam"] is True
    assert [show_call.args[0] for show_call in show_calls] == [
        "actor:source",
        "actor:target",
    ]
    assert [show_call.args[1] for show_call in show_calls] == [
        ("source", {"pos": "top-left", "s": 0.7, "c": "black"}),
        ("target", {"pos": "top-left", "s": 0.7, "c": "black"}),
    ]
    assert [call.kwargs["at"] for call in show_calls] == [0, 1]
    assert [call.kwargs["axes"] for call in show_calls] == [1, 1]
    assert [call.kwargs["viewup"] for call in show_calls] == ["z", "z"]
    assert show_calls[0].kwargs["interactive"] is False
    assert show_calls[1].kwargs["interactive"] is True
    assert close_calls == [call()]


def test_render_record_delegates_single_record_as_a_list(monkeypatch):
    """Keep the one-record compatibility helper delegated to the shared renderer."""
    visualize_model = load_visualize_model_module()
    render_calls = []
    record = SimpleNamespace(name="single")
    monkeypatch.setattr(
        visualize_model,
        "render_records",
        lambda records: render_calls.append(records),
    )

    visualize_model.render_record(record)

    assert render_calls == [[record]]


def test_main_loads_and_renders_two_existing_paths(monkeypatch, tmp_path, capsys):
    """Load, summarize, and render a source/target path pair."""
    visualize_model = load_visualize_model_module()
    paths = [tmp_path / "source.glb", tmp_path / "target.glb"]
    for path in paths:
        path.touch()
    records = [
        SimpleNamespace(
            name=path.stem,
            vertices=np.zeros((index + 3, 3)),
            faces=np.zeros((index + 1, 3)),
        )
        for index, path in enumerate(paths)
    ]
    loaded_paths = []
    rendered_records = []

    def fake_load(path):
        """Return the next synthetic record for a loaded path.

        Args:
            path: Mesh path requested by the CLI.

        Returns:
            Synthetic record corresponding to `path`.
        """
        loaded_paths.append(path)
        return records[len(loaded_paths) - 1]

    monkeypatch.setattr(visualize_model, "load_scene_mesh_records", fake_load)
    monkeypatch.setattr(
        visualize_model,
        "render_records",
        lambda values: rendered_records.extend(values),
    )
    monkeypatch.setattr(sys, "argv", ["visualize_model.py", *map(str, paths)])

    visualize_model.main()

    assert loaded_paths == paths
    assert rendered_records == records
    assert capsys.readouterr().out.splitlines() == [
        "Loaded source: 3 vertices, 1 faces, texture",
        "Loaded target: 4 vertices, 2 faces, texture",
    ]


def test_main_rejects_more_than_two_paths(monkeypatch, tmp_path, capsys):
    """Reject CLI input that is not one model or one source/target pair."""
    paths = [tmp_path / f"model-{index}.glb" for index in range(3)]
    for path in paths:
        path.touch()
    visualize_model = load_visualize_model_module()
    monkeypatch.setattr(sys, "argv", ["visualize_model.py", *map(str, paths)])

    with pytest.raises(SystemExit) as error:
        visualize_model.main()

    assert error.value.code == 2
    assert "provide one GLB or one source/target pair" in capsys.readouterr().err
