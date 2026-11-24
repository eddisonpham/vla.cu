# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The launch floor is named and now attacked twice. A cooperative CUDA kernel fuses the
whole layer tail into one launch: 3.13x at 1 token, 2.55x at 2, 2.22x at 3, 1.25x at 8,
0.24x at 50 (EXP-078). Above 3 tokens the body, not the launch, is the cost, and the
body split says the int4 GEMM moves bytes at 20 to 23 GB/s, 5 to 6 percent of peak,
while its int8 twin reaches 91 and cuBLAS bf16 172 (EXP-079).

## What works

- 35 tasks, `pytest -q` 165 passed with vcvars active, `style` at zero.
- `coop` builds `csrc/coop_layer.cu` and matches the five launch path at 1, 2, 3, 8 and
  50 tokens within 1.3e-3.
- The int8 menu now gives a chunk above 32 tokens a 16 row, 128 wide tile (EXP-080).
- 33 tasks before this pass, now `int8pair`, `body` and `coop` join.

## Open edges

- **The fused body is a GEMV.** The cooperative kernel re-reads the activation tile per
  output row and uses no tensor core, so it loses 4x at 50 tokens. Tiling is the fix.
- **The int4 dequant path.** Two dots per K step, a group scale expanded across the tile
  and a transpose in shared memory. That, not DRAM, is where the 5 to 6 percent of peak
  goes, and `ncu` is still blocked by ERR_NVGPUCTRPERM.
- **The cooperative launch assumes no other process takes a block.** Fine here, not a
  promise on a shared desktop.
- **Above 64 tokens the einsum mix is back**, 25x slower, covered by one test.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: tile the cooperative body, or strip the int4 dequant in a test variant.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Tile the cooperative body with tensor core dots and one shared activation tile.
2. An int4 variant without the group scale and transpose, to separate them.
3. Pair the new int8 tile at 8 tokens and at 1920 x 720.
