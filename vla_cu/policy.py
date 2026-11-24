"""The int8 SmolVLA path behind one object: prefill once per observation, then replay a chunk.

Concept: a deployed policy reads an observation once and denoises ten times, so the 241 row
prefix is built once, the 16 expert layers are captured in one graph, and a denoising step is
a buffer write plus a replay. Shapes are fixed at 3 frames, 48 language rows, 1 state row and
50 action tokens.

Task: python -m vla_cu backend
"""

from __future__ import annotations

import time

import torch
from torch import Tensor

from vla_cu.build import layer_from_checkpoint
from vla_cu.checkpoint import Checkpoint
from vla_cu.prefix import FULL_TOKENS, LAYERS, cache_from_rows, full_prefix_inputs

TOKENS = 50


class SmolVlaPolicy:
    """The deployed path: the 16 expert layers at int8, cross attending to one prefix cache.

    `prefill` builds the cache and captures the graph, `act` writes a noise guess into the
    captured input and replays. The policy is inference only, one CUDA device, fp32
    activations, and it holds about 100 MB of expert weights plus the prefix cache.
    """

    def __init__(
        self,
        checkpoint: Checkpoint,
        tokens: int = TOKENS,
        tower_bits: int = 8,
        attention_bits: int = 8,
        down_bits: int = 8,
    ) -> None:
        self.checkpoint = checkpoint
        self.tokens = tokens
        self.layers = [
            layer_from_checkpoint(
                checkpoint,
                index,
                tokens=tokens,
                part="expert",
                other_tokens=FULL_TOKENS if index % 2 else TOKENS,
                attention_bits=attention_bits,
                down_bits=down_bits,
            )
            for index in range(LAYERS)
        ]
        self.hidden = self.layers[0].hidden_width
        self.tower_bits = tower_bits
        self.bytes = sum(layer.bytes for layer in self.layers)
        self.input = torch.empty((tokens, self.hidden), device="cuda", dtype=torch.float32)
        self.first = torch.empty_like(self.input)
        self.second = torch.empty_like(self.input)
        self.output = torch.empty_like(self.input)
        self.prefixes: list[tuple[Tensor, Tensor] | None] = [None] * LAYERS
        self.keys: list[Tensor | None] = [None] * LAYERS
        self.values: list[Tensor | None] = [None] * LAYERS
        self.rows: Tensor | None = None
        self.visible: Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    def _chain(self) -> None:
        """Run every expert layer from the input buffer, alternating two scratch buffers."""
        current, target = self.first, self.second
        current.copy_(self.input)
        for layer, prefix in zip(self.layers, self.prefixes):
            if prefix is None:
                layer.into(current, target)
            else:
                layer.into(current, target, prefix[0], prefix[1])
            current, target = target, current
        self.output.copy_(current)

    def prefill(self, frames: Tensor, state: Tensor) -> float:
        """Build the 241 row prefix, write it into the held key and value buffers, and return seconds.

        `frames` is [3, 3, 512, 512] in [-1, 1] and `state` is the 32 wide padded robot state.
        The first call also captures the expert graph, so a server pays that capture once and
        every later call only copies the new cache in place.
        """
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.rows, self.visible = full_prefix_inputs(self.checkpoint, frames, state, tower_bits=self.tower_bits)
        keys, values = cache_from_rows(self.checkpoint, self.rows, self.visible, layers=LAYERS)
        if self.graph is None:
            for index, layer in enumerate(self.layers):
                if layer.attention.other_width == self.hidden:
                    continue
                self.keys[index] = keys[index].clone()
                self.values[index] = values[index].clone()
                self.prefixes[index] = (self.keys[index], self.values[index])
            self._chain()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._chain()
        else:
            for index in range(LAYERS):
                if self.keys[index] is not None:
                    self.keys[index].copy_(keys[index])
                    self.values[index].copy_(values[index])
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def act(self, noise: Tensor) -> Tensor:
        """Write `noise` [tokens, hidden] into the captured input, replay, and return the chunk.

        The returned tensor is the graph's output buffer, so the next `act` overwrites it.
        The caller synchronizes before reading.
        """
        if self.graph is None:
            raise RuntimeError("prefill must run before act, because it binds the prefix cache")
        self.input.copy_(noise)
        self.graph.replay()
        return self.output
