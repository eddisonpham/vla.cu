"""vla.cu: fused 4-bit dequantize and matmul for batch-1 vision-language-action policies.

Concept: weights stay packed in memory and are unpacked inside registers, so a forward
pass streams half a byte per parameter instead of four. `Int4Linear` and the block
classes are the user-facing API; the kernels and the quantizer are exported beside them.

Names resolve on first access, because a kernel module imports this package and an eager
import here would close that loop.

Task: python -m vla_cu demo
"""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, str] = {
    "ActionBlock": "vla_cu.blocks",
    "AttentionBlock": "vla_cu.blocks",
    "Bundle": "vla_cu.bundle",
    "Checkpoint": "vla_cu.checkpoint",
    "GqaTokenBlock": "vla_cu.blocks",
    "GqaValueBlock": "vla_cu.blocks",
    "Int4Linear": "vla_cu.linear",
    "MlpBlock": "vla_cu.blocks",
    "MlpTokenBlock": "vla_cu.blocks",
    "PackedMatrix": "vla_cu.bundle",
    "dequantize_int4": "vla_cu.quant",
    "gqa_mix": "vla_cu.blocks",
    "hadamard": "vla_cu.rotate",
    "int4_gemm": "kernels.triton.int4_gemm",
    "int4_gemv": "kernels.triton.int4_gemv",
    "int4_grouped_gemv_into": "kernels.triton.int4_grouped_gemv",
    "int8_gemv": "kernels.triton.int8_gemv",
    "int8_grouped_gemv_into": "kernels.triton.int8_grouped_gemv",
    "load_bundle": "vla_cu.bundle",
    "open_checkpoint": "vla_cu.checkpoint",
    "pack_int4": "vla_cu.quant",
    "pack_matrix": "vla_cu.bundle",
    "quantize_int8": "vla_cu.quant",
    "rotate_rows": "vla_cu.rotate",
    "rotate_vector": "vla_cu.rotate",
    "rotary": "vla_cu.blocks",
    "save_bundle": "vla_cu.bundle",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import one public object on first access and cache it in the module globals."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names without importing them."""
    return __all__
