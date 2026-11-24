"""The task registry, checked without a GPU because CI has none.

A task name that points at a missing or renamed module is a broken CLI, and the runner's
help text is built from the same registry, so both are cheap to hold still.
"""

from __future__ import annotations

import importlib

import pytest

from vla_cu.__main__ import TASKS, usage


@pytest.mark.cpu
@pytest.mark.parametrize("task", sorted(TASKS))
def test_task_resolves_to_a_module_with_main(task: str) -> None:
    """Every registry entry must import and expose the entry point the runner calls."""
    module = importlib.import_module(TASKS[task])
    assert callable(getattr(module, "main", None))


@pytest.mark.cpu
def test_usage_lists_every_task() -> None:
    """The help text is generated from the registry, so no task can be hidden from it."""
    text = usage()
    assert all(name in text for name in TASKS)
