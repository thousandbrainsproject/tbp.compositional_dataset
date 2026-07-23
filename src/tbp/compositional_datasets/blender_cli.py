from __future__ import annotations

import sys


def argv_after_blender_separator(argv: list[str] | None = None) -> list[str]:
    """Return script arguments passed after Blender's -- separator.

    Args:
        argv: Full process argument list. Uses `sys.argv` when None.

    Returns:
        Arguments after --, or an empty list when the separator is absent.
    """
    arguments = sys.argv if argv is None else argv
    if "--" not in arguments:
        return []
    return arguments[arguments.index("--") + 1 :]
