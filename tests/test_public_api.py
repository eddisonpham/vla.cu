"""The package exports one working name per user-facing object, and CI can check it."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("triton")

import vla_cu


@pytest.mark.cpu
def test_every_exported_name_resolves() -> None:
    """A name in __all__ that does not import is a broken public API."""
    missing = [name for name in vla_cu.__all__ if not hasattr(vla_cu, name)]
    assert missing == []


@pytest.mark.cpu
def test_the_blocks_and_the_linear_layer_are_classes() -> None:
    """The objects a caller constructs are classes, not modules behind a path."""
    names = (
        "ActionBlock",
        "AttentionBlock",
        "GqaTokenBlock",
        "GqaValueBlock",
        "Int4Linear",
        "MlpBlock",
        "MlpTokenBlock",
        "Bundle",
        "PackedMatrix",
        "Checkpoint",
    )
    for name in names:
        assert inspect.isclass(getattr(vla_cu, name)), name


@pytest.mark.cpu
def test_no_duplicate_exports() -> None:
    """One name per object, so an import cannot depend on order."""
    assert len(vla_cu.__all__) == len(set(vla_cu.__all__))
