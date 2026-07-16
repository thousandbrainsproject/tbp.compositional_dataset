from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the Blender script with a minimal importable `bpy` double.

    Args:
        monkeypatch: Pytest fixture used to install the fake module.

    Returns:
        Loaded `create_sticker_disk` script module.
    """
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    path = Path(__file__).resolve().parents[1] / "scripts" / "create_sticker_disk.py"
    spec = importlib.util.spec_from_file_location("create_sticker_disk", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_approved_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only image and output are required for the standard disk.

    Args:
        monkeypatch: Pytest fixture used to install the fake `bpy` module.
    """
    script = load_script(monkeypatch)

    args = script.parse_args(["--image", "square.png", "--out", "square.glb"])

    assert args.image == Path("square.png")
    assert args.out == Path("square.glb")
    assert args.diameter == 0.02
    assert args.thickness == 0.001
    assert args.segments == 64


def test_validate_asset_paths_accepts_new_png_to_glb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing PNG and absent GLB form a valid conversion request.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Pytest fixture used to install the fake `bpy` module.
    """
    script = load_script(monkeypatch)
    image = tmp_path / "square.png"
    image.write_bytes(b"png")

    script.validate_asset_paths(image, tmp_path / "square.glb")


@pytest.mark.parametrize(
    ("image_name", "out_name", "create_out", "message"),
    [
        ("missing.png", "disk.glb", False, "sticker image does not exist"),
        ("sticker.jpg", "disk.glb", False, "sticker image must be a .png"),
        ("sticker.png", "disk.obj", False, "output must be a .glb"),
        ("sticker.png", "disk.glb", True, "output already exists"),
    ],
)
def test_validate_asset_paths_rejects_invalid_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_name: str,
    out_name: str,
    create_out: bool,
    message: str,
) -> None:
    """Invalid source or destination paths fail before Blender mutation.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Pytest fixture used to install the fake `bpy` module.
        image_name: Source filename to validate.
        out_name: Destination filename to validate.
        create_out: Whether to create an existing destination.
        message: Expected exception text.
    """
    script = load_script(monkeypatch)
    image = tmp_path / image_name
    if image_name != "missing.png":
        image.write_bytes(b"image")
    out = tmp_path / out_name
    if create_out:
        out.write_bytes(b"existing")

    with pytest.raises((FileNotFoundError, ValueError, FileExistsError), match=message):
        script.validate_asset_paths(image, out)
