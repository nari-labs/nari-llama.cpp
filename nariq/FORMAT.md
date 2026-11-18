# NariQ GGUF format, version 1

The GGUF metadata contains `trellis.version` as UINT32 with value 1. The loader rejects missing, unsupported or incorrectly typed versions when coded tensors are present.

## Coded tensors

`GGML_TYPE_TRL1` through `GGML_TYPE_TRL6` use fork-specific IDs 44 through 49. Their block size is 16 and type size is `2 * bits` bytes. These IDs are not an upstream format registration.

A matrix with N output rows and K input columns has GGML shape `[K, N]`. Both dimensions are multiples of 16. Codes are stored in output-tile-major order: for each output tile row J, then input tile column i, store that tile's `16 * bits` int16 code words. A nominal row's bytes do not independently encode that row's weights; decoding operates on groups of 16 rows.

For a weight named `<w>.weight`, side tensors are:

| Tensor | Type and shape | Meaning |
| --- | --- | --- |
| `<w>.trellis_s16` | F16 `[K/16]` | Scale for each 16-input block |
| `<w>.trellis_pre` | F32 `[K]` | Input signs and scales |
| `<w>.trellis_post` | F32 `[N]` | Output signs and scales |
| `<w>.trellis_pin` | I32 `[K]`, optional | Input gather indices |
| `<w>.trellis_pout` | I32 `[N]`, optional | Output gather indices |

`trellis.desc.<w>` is an INT32 array:

```text
[rows_kind, rows_block, rows_shards, cols_kind, cols_block, cols_shards,
 nparts, part_0, ..., part_n]
```

Kinds are identity (0), normalized block Sylvester Hadamard (1), or whole-slice Hadamard (2). The latter supports slice lengths `m * 2^j` with m in 1, 12, 20 and 68; Paley matrices supply the non-power-of-two factors. The output side carries up to eight fused parts. `ggml_trellis_desc_supported` validates the dimensions and transforms.

The coded linear computes:

```text
y = gather_pout(post * T_rows^T(W_rotated * T_cols(pre * gather_pin(x))))
W_rotated = decoded_integer_grid * s16
```

The grid contains integers in [-126, 126]. Whole-slice transforms use the forward transform for input and its inverse for output. The reference implementation is `ggml/src/ggml-trellis.cpp`; CPU execution is in `ggml/src/ggml-cpu/ops.cpp`, and CUDA execution uses the vendored libtrellis.

## Embeddings and frequency bands

An ordinary coded embedding uses `token_embd.weight` and the same side tensors, with no row transform. `GGML_OP_TRELLIS_GET_ROWS` decodes selected rows and applies the inverse column transform and scales.

A banded output head uses `output.bandNN.weight`, side tensors for each band, `output.vocab_gather` (I32 vocabulary permutation), and `trellis.lm_head_bands` (UINT32 count). It has no unbanded `output.weight`.

A banded embedding uses `token_embd.bandNN.weight`, side tensors for each band, `token_embd.vocab_gather`, and `trellis.embed_bands`. It has no unbanded `token_embd.weight`. The graph chains in-place band lookups; each band writes only the tokens mapped to its row interval.

## Converter and runtime restrictions

`conversion/trellis.py` consumes a checkpoint whose `quantization_config.quant_method` is `native_trellis`. Coded modules provide `trellis`, `block_scale`, `su`, `sv`, and `row_scale`; permuted layouts also provide `perm_rows` and/or `perm_cols`. The converter composes those permutations with Qwen's DeltaNet V-head order into gathers, and retains the coded bytes.

These types have no ordinary per-row `to_float` conversion. Only the custom operations and K-aware host helpers understand the tile layout and side tensors. `llama-quantize` cannot directly requantize a NariQ file. LoRA adapters are currently rejected. Metal implements coded products on GPU and keeps coded embedding lookups on CPU. Vulkan and HIP do not implement these operations on GPU.
