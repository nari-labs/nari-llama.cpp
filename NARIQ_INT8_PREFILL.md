# INT8 prefill path for NariQ (experimental)

An optional prefill kernel that uses INT8 tensor cores (`mma.m16n8k32`) instead of
FP16 (`mma.m16n8k16`). It is a pure speed change: weights are decoded to the same
integer grid and activations are quantized to int8 with a per-row scale.

Measured on one RTX 5060 Ti 16GB (sm_120), Qwen3.8-27B NariQ 3.0bpw, `llama-bench -p 8192 -n 128 -b 2048 -ub 768 -fa on`:

| | pp8192 | tg128 |
|---|---:|---:|
| default (FP16 m16n8k16) | 674.8 tok/s | 31.7 tok/s |
| `TQ_I8=1`, first version | 1110.3 tok/s | 31.7 tok/s |
| `TQ_I8=1`, current | **1417-1424 tok/s** | 32 tok/s |

The current version is 2.1x the FP16 path. Inside `pf_k_i8` the MMA count is fixed by the
model, so the kernel runs at ~125 TOPS = 60% of the 207 TOPS INT8 peak measured on this part;
the rest is shared-memory traffic, cp.async and barriers, not decode.

Quality against the FP16 path on 2 chunks of source text (`llama-perplexity --kl-divergence`):
`Same top p: 98.2%`, `RMS Δp: 1.36%`. The approximation comes only from the activation
quantization (per-row scale).

## Why it works

- The trellis grid is integer: `v = sum(four 6-bit digits) - 126`, so `v ∈ [-126, 126]`
  and the weights fit in signed int8 **exactly** (`common.cuh:72`). The FP16 path
  multiplies `4v` and scales by `0.25` in the epilogue; the INT8 path uses `v` directly.
- INT8 tensor throughput on this GPU is 4x the FP16-with-FP32-accumulate rate
  (measured: 207 TOPS vs 51.8 TFLOPS), which is what makes 1000+ tok/s possible at
  all — the FP16 path already runs at ~97% of its own peak.
- The bare FLOP count for the trellis linear layers is 48.2 GFLOP/token, so 1000 tok/s
  needs 48 TFLOP/s in the FP16 path — equal to the hardware peak. INT8 leaves ample room.

## What changed

1. `prologue_k` writes int8 activations with the existing per-row power-of-two scale `2^e`,
   shifted to `2^(e-7)` so the row max lands in `[64, 128)` before rounding/clamping to
   `[-127, 127]`. The layout is `[k32-block][group][token][32 int8]`: a lane's two B words
   for one `m16n8k32` step are adjacent, so the GEMM reads them with a single `LDS.64`
   (80 scalar `LDS` per main-loop pass became 16 `LDS` + 32 `LDS.64`). Worth ~7%.
2. `pf_k_i8` builds the A fragment in the m16n8k32 layout. For `bits <= 3` the four states a
   lane needs from one ring span fit a single 32-bit window (`16 + 5*K <= 31`), so the lane
   reads two windows and decodes its own fragment: no shuffles at all. For `bits > 3` two
   `__shfl_sync` carry the other half of the columns and one `__byte_perm` assembles each
   register. Worth ~4-5%.
3. `mma.m16n8k32` accumulates int32; the epilogue converts to float, scales by 512
   (`2^7 * 4`), and reuses the existing `finish_row` scales.
4. The INT8 kernel uses the 128x8 tile with two CTAs per SM (`CK=8`, two stages), which
   measured 3% faster than 128x4 once the decode got cheap.
5. The MLP chain (up/gate products, both output rotations, swiglu, down's input rotation) runs
   as one kernel at any row count. It was gated to one column because the products went through
   the GEMV group entry point; above one column they are plain `tq_gemm` calls into the same
   fused kernel, which is bit-identical and replaces three passes over gate/up. Worth ~9%.
6. `TQ_I8` env var selects the path; default behavior is unchanged. Matrices with
   `splits > 1` (split-K partials go through `reduce_k`) stay on the FP16 path.

Also adds `TQ_TILING=0|1|2` to force `T64X8` / `T128X8` / `T128X4` for experiments; the
INT8 path defaults to `T128X8`. `T64X8` and `T128X4` measure 1196 and 1258 in the same run.

## Use

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build -j --target llama-bench llama-server

TQ_I8=1 ./build/bin/llama-server -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj ... \
    -ngl 99 -c 131072 -fa on -b 2048 -ub 768
```

## Correctness and quality

The integer grid is exact, so every optimisation above only changes how the same values are
read and packed. Verified with a shape-sweep harness (K in {384, 512}, N in {144, 256},
rows in {65, 128, 257}, bits 1-6, all three tilings, padded strides, both Hadamard flags):
the output hash of all 216 cases, extended to 360 for the five tile/CPS variants, is
bit-identical to the first version. Against the FP16 path on 2 chunks of source text:
`Same top p: 98.235%`, `RMS d p: 1.363%`, and 16-chunk PPL 2.4655 (FP16) vs 2.4678 (INT8) —
the same numbers as before these speed changes.

## Benchmarks (Qwen3.8-27B NariQ 3.0bpw, one RTX 5060 Ti)

`llama-server` with `TQ_I8=1`, KV q4_0, `-fa on`, greedy, 4 parallel slots, 16384-token budget
(AIME26, thinking on), and the same binary with `TQ_I8` unset as the control:

| benchmark | INT8 prefill | FP16 prefill |
|---|---:|---:|
| AIME26 full (30) | **80.0 %** (24/30) | 73.3 % (22/30) |
| MMLU-Pro (70) | 65.7 % | 71.4 % |
| C-Eval (20) | 80.0 % | 75.0 % |

Per-question agreement between the two paths: AIME 93.3 % (28/30, INT8 wins 2, loses 0),
MMLU-Pro 88.6 %, C-Eval 95.0 %. At these sample sizes the differences are inside the noise
(binomial SE is 8 points on 30 questions); the deterministic alignment is in "Correctness"
above. For reference, the same model with llama.cpp's IQ3_S quant (12.4 GB, no trellis) scores
80.0 % on the same AIME setup; this 10.1 GB file matches it.

The MLP chain fusion is what a 488-token prompt mostly does *not* see; end-to-end the prefill
speedup shows up on long prompts and on the `pp8192` figure above.

## Limits / next steps

- Activation quantization is per-row; per-block scales (e.g. one per 32-64 K elements)
  should recover most of the remaining 1.8% top-1 gap, at a cost of one float FMA per
  accumulator per K block in the epilogue.
- `pf_k_i8` is 55.5% of the prefill time (3.1 s per 8192 tokens, ~60% of the INT8 peak).
  The other kernels: `rht_cta_k` 10.3%, `gdn_single` 6.5%, `flash_attn` 5.7%,
  smaller kernels.
- `splits > 1` matrices stay FP16; they are the small projections (a few percent of time).

Things that were measured and did **not** help:
- `m16n8k16.s8` instead of `m16n8k32`: exactly half the throughput on this part
  (103 vs 207 TOPS measured), whatever it saves in the repack.
- Occupancy: at 127 registers/thread the 3-bit kernel gets 16 warps/SM; `CPS=3/4` spills
  (280/248 and 904/728 bytes per thread) and drops to 994/523 tok/s. One CTA of 8 warps
  already saturates the tensor pipe (206 TOPS), so occupancy was never the limit.
- Folding the `-126` into the dp4a accumulator: the compiler already merges part of the
  subtract and pack; 1102.7 vs 1102.7, no change.
- Interleaving decode with MMA (dropping the one-pair prefetch): 1299 vs 1304, no change.
- Swapping the grid axes so resident CTAs share one activation tile: 1243 vs 1256, slightly
  worse — the activation reads were already served from L2.
- Cheaper 1024-point RHT (reusing the FP32 transpose, 4 -> 2 barriers): 1111 vs 1111.
- Dropping the GEMM's activation reads entirely (a constant B operand) only reaches 1464 vs
  1418: the shared-memory B traffic is 3% of the run, so a wider n-tile (more outputs per CTA)
  has nothing to win.
- Sharing one prologue between the projections of one x (they look like an input group): the
  per-input scale `pre` differs per matrix, so the quantized activations differ too.
- Fusion candidates outside the GEMM (input gather + RHT, prologue recompute for large K)
  are each worth under 1% of the wall clock.

nsys breakdown at 1297 tok/s (kernel sum 12.6 s for two 8192-token passes):
`pf_k_i8` 52.7%, `rht_cta_k` 10.3%, `gdn_single` 6.5%, `flash_attn` 5.7%, `prologue_k` 5.2%,
`silu` 5.2%, `trellis_gather` 4.7%, `concat_non_cont` 2.0%, `k_bin_bcast` 1.5%.
