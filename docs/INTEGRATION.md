# Wiring vla.cu into a policy runtime

The library side exists: `SmolVlaPolicy` holds the int8 expert, builds the 241 row prefix per
observation, and replays one captured graph per denoising step. This page is the contract a
host such as LeRobot Edge has to meet to call it. The host itself is a separate repository,
which is why nothing here imports it.

## What the host supplies

| Input | Shape and dtype | Notes |
|---|---|---|
| frames | [3, 3, 512, 512], fp32 in [-1, 1] | Already resized and normalized. This repo has no camera or tokenizer path. |
| state | [32], fp32 | The padded robot state the checkpoint's `state_proj` expects. |
| noise | [50, 720], fp32 | The action guess for one denoising step. |
| checkpoint | `.safetensors` | `open_checkpoint` parses the header only, so one matrix reads at a time. |

## What the policy returns

| Call | Result |
|---|---|
| `prefill(frames, state)` | seconds spent, and the 16 layer key and value cache held on the GPU |
| `act(noise)` | [50, 720] written into a reused buffer, valid after the next synchronize |

## The three lines a backend needs

```python
from vla_cu import SmolVlaPolicy, open_checkpoint

policy = SmolVlaPolicy(open_checkpoint(path))
policy.prefill(frames, state)
chunk = policy.act(noise)
```

The first `prefill` in a process also captures the expert graph, which is 1.07 s of
compilation and capture on this machine. Later prefills cost 123.34 ms and each step 2.42 ms.

## The gate

| Check | Value | Where |
|---|---|---|
| cosine against an fp32 expert chain over the same prefix | 0.99996 | `python -m vla_cu backend` |
| cosine of the whole model against fp32 | 0.99990 | EXP-114 |
| the floor the backend refuses to ship under | 0.98 | `bench/backend.py` |

`bench/backend.py` exits nonzero below the gate, so a host can run it as an acceptance test
before switching a policy over.

## Files a deployment carries

| File | Size | Built by |
|---|---|---|
| checkpoint | 906.71 MB | shipped with the model |
| pack file, `model.pack.int8.pt` | 417.83 MB | `python -m vla_cu prepack`, used automatically when present |
| resident after warmup | about 103 MB tower, 629.27 MB text weights, 98.68 MB expert, 9.87 MB cache | |

## What is still missing on this side

| Gap | Consequence |
|---|---|
| No tokenizer | The 48 language rows are a fixed embedding slice, so the instruction text is ignored. |
| No timestep embedding | The checkpoint carries no time projection, so the ten steps differ only by the noise written into the graph input. |
| One observation per prefill | The cache is rebuilt for every frame triple, not reused across an episode. |
| Synthetic inputs | The measurements use seeded noise frames and a zero state, not camera data. |
