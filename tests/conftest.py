"""Session guards so a machine without a CUDA device skips instead of erroring.

The kernels are GPU only, so CI runs on a CPU runner. Every test that needs a device is
skipped with a reason, and the tests marked `cpu` still run there, which is what keeps the
style gate and the task registry honest without a GPU.
"""

from __future__ import annotations

import pytest
import torch


def device_visible() -> bool:
    """Return whether a CUDA device is actually usable, which a driver alone does not prove.

    `torch.cuda.is_available()` reads true while `device_count()` reads zero when
    CUDA_VISIBLE_DEVICES hides every device, which is how a CPU CI image can look.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


@pytest.fixture(autouse=True)
def require_cuda(request: pytest.FixtureRequest) -> None:
    """Skip a device test when no CUDA device is visible, unless the test is marked cpu."""
    if request.node.get_closest_marker("cpu") is not None:
        return
    if not device_visible():
        pytest.skip("no CUDA device on this machine; the kernels are GPU only")
