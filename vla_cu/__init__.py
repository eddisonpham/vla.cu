"""vla.cu: fused 4-bit dequantize and matmul for batch-1 vision-language-action policies.

Concept: weights stay packed in memory and are unpacked inside registers, so a forward
pass streams half a byte per parameter instead of four. The Triton kernels are the
public API here; the native CUDA entry points live in `vla_cu.cuda_int4` and
`vla_cu.cuda_int8`, because they need a host compiler at first use.

Task: python -m vla_cu --help
"""

from kernels.triton.int4_gemm import int4_gemm
from kernels.triton.int4_gemv import int4_gemv
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int8_gemv import int8_gemv
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into
from vla_cu.bundle import Bundle, PackedMatrix, load_bundle, pack_matrix, save_bundle
from vla_cu.quant import dequantize_int4, pack_int4, quantize_int8

__all__ = [
    "Bundle",
    "PackedMatrix",
    "dequantize_int4",
    "load_bundle",
    "int4_gemm",
    "int4_gemv",
    "int4_grouped_gemv_into",
    "int8_gemv",
    "int8_grouped_gemv_into",
    "pack_int4",
    "pack_matrix",
    "quantize_int8",
    "save_bundle",
]
