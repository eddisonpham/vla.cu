"""Cross-check one frame's tower output against a transformers SigLIP run.

Concept: this repo reads the SigLIP weights by hand, so the only independent check of that
reading is a second implementation. transformers ships SiglipVisionModel, and loading the
checkpoint's own 197 vision tensors into it separates "the weights are the checkpoint's"
from "the math is right". Both sides run fp32 on the same seeded frame.

Task: python -m vla_cu visioncheck
"""

from __future__ import annotations

import torch

from bench.real_layers import CHECKPOINT, cosine, relative_error
from vla_cu.bench_utils import enforce_memory_budget, warm_gpu
from vla_cu.checkpoint import Checkpoint, open_checkpoint
from vla_cu.vision import BASE, EPS, HEADS, HIDDEN, IMAGE, LAYERS, PATCH, SCALE, Tower, pixel_shuffle

INTERMEDIATE = 3072


def vision_state(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    """Return the tower's tensors named the way transformers names them, as fp32 on the GPU.

    `SiglipVisionModel` owns the transformer directly, so the checkpoint's `vision_model.`
    segment is dropped and the rest of the key is already transformers' spelling.
    """
    state = {}
    for name in checkpoint.tensors:
        if name.startswith(f"{BASE}."):
            state[name[len(BASE) + 1:]] = checkpoint.read(name).to(device="cuda", dtype=torch.float32)
    return state


def main() -> int:
    """Run transformers' SigLIP and this repo's tower on one frame, then compare patches and rows."""
    try:
        from transformers import SiglipVisionConfig, SiglipVisionModel
    except ImportError:
        print("transformers is not installed; install 'transformers<6' to run this check")
        return 1

    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    frame = torch.rand((3, IMAGE, IMAGE), device="cuda", generator=torch.Generator(device="cuda").manual_seed(0)) * 2 - 1
    print(f"GPU {torch.cuda.get_device_name(0)} | one {IMAGE} frame, {IMAGE // PATCH} x {IMAGE // PATCH} patches, {LAYERS} layers")
    print("the frame is seeded uniform noise in [-1, 1], the same input on both sides")

    config = SiglipVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_channels=3,
        image_size=IMAGE,
        patch_size=PATCH,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=EPS,
        vision_use_head=False,
    )
    model = SiglipVisionModel(config).to(device="cuda", dtype=torch.float32).eval()
    state = vision_state(checkpoint)
    report = model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} checkpoint tensors, missing {len(report.missing_keys)}, unexpected {len(report.unexpected_keys)}")
    for label, keys in (("missing", report.missing_keys), ("unexpected", report.unexpected_keys)):
        for key in keys[:5]:
            print(f"  {label} {key}")

    with torch.no_grad():
        reference = model(pixel_values=frame[None]).last_hidden_state[0].clone()
    tower = Tower(checkpoint, 32)
    ours = tower.hidden(frame)

    rows = tower.rows(frame)
    reference_rows = tower.connector(pixel_shuffle(reference[None], SCALE)[0])
    print()
    print(f"{'comparison':>22}  {'shape':>14}  {'max abs':>9}  {'rel L2':>9}  {'cos':>9}")
    for label, left, right in (
        ("patches, 768 wide", ours, reference),
        ("connector rows, 960", rows, reference_rows),
    ):
        shape = " x ".join(str(size) for size in left.shape)
        print(
            f"{label:>22}  {shape:>14}  {(left - right).abs().max().item():>9.2e}  "
            f"{relative_error(left.reshape(-1), right.reshape(-1)):>9.2e}  {cosine(left.reshape(-1), right.reshape(-1)):>9.5f}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
