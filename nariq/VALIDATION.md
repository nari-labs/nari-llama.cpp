# Numerical reference checks

A kernel-parity reference must come from the exact coded GGUF being tested. An assembled BF16 checkpoint can include intermediate rounding before tuned scales are applied; it is not necessarily the same matrix as decoding the final codes and rounding once. Comparing against it mixes reference-generation error with runtime error.

## Generate a dense reference

From the fork root, install the bundled Python GGUF package and build its CPU decoder:

```sh
python -m pip install -e ./gguf-py
cmake -S . -B build-reference -DGGML_CUDA=OFF -DGGML_NATIVE=OFF
cmake --build build-reference --target ggml-base -j
python nariq/decode_reference.py coded.gguf reference-bf16.gguf \
  --library build-reference/bin/libggml-base.so --outtype bf16
python nariq/decode_reference.py coded.gguf reference-f32.gguf \
  --library build-reference/bin/libggml-base.so --outtype f32
```

Use the corresponding `ggml-base` shared-library path on other platforms. The converter currently requires a little-endian host and GGUF. It runs on the CPU, copies uncoded tensors byte-for-byte, and decodes coded weights with `ggml_trellis_dequantize`. BF16 output rounds that F32 result once using round-to-nearest-even. F32 output avoids that additional BF16 weight rounding; it is still a floating-point reference, not exact arithmetic. It is useful to cross-check sampled matrices against an independent decoder before relying on it.

Embedding and output bands are restored to vocabulary order. Input/output gathers are applied by the decoder. Separate projection matrices stay separate, so this does not introduce a fused-projection conversion. The output contains no coded tensors or runtime side tensors.

The converter refuses to overwrite an output or partial file. A successful output has a `.gguf.json` manifest with source, decoder-library and output SHA256 hashes, output size, decoded type and completion status. Interrupted `.partial` files are not complete references. Keep the source model and decoder fixed throughout a comparison.

Plan for roughly 54 GB of BF16 output or 108 GB of F32 output for a 27B model, plus input and temporary files. Memory use is bounded by the worker count times the largest dense matrix, plus rounding and band buffers; the default is four workers. Reduce `--threads` on hosts with less RAM. F32 GPU inference also requires substantially more VRAM than coded inference.

## Compare without conflating numerical effects

Use the same corpus, tokenization, context length, scored positions, attention settings and microbatch size for both models. Record the exact runtime build and GPU. Check at least:

- Coded versus F32-decoded weights, to isolate coded-kernel arithmetic from BF16 weight rounding.
- Coded versus once-rounded BF16 weights, to measure the combined effect of BF16 weight and activation arithmetic.
- Each reference at both prefill and decode microbatch sizes, to measure its own execution-path sensitivity.
- The previous and candidate runtimes with identical coded weights, to check migration regressions.

Native `llama-perplexity --kl-divergence-base` uses compressed log probabilities with a 16-logit range and excludes sufficiently small reference probabilities during comparison. Its reported KL is an approximation, not an exact full-vocabulary sum; tiny negative estimates are possible. Use uncompressed F32 logits and a float64 log-softmax/reduction when a gate explicitly requires full-vocabulary KL. Verify token IDs, vocabulary order, shape, finite values and file completeness before comparing raw logits.

Keep thresholds and corpus fixed before execution. Reference corrections require a new, separately recorded diagnostic; retain the original failed results. A small runtime-parity sample does not establish model quality or replace release checks on other devices. Current candidate evidence is in [cuda-qualification.md](cuda-qualification.md).

## Converter checks

```sh
NARIQ_TEST_LIBRARY="$PWD/build-reference/bin/libggml-base.so" \
  python -m unittest discover -s nariq/tests -p test_decode_reference.py
```

The native fixture uses analytically known coded weights, row/column permutations and vocabulary bands. It checks both output types, BF16 halfway rounding, metadata preservation and rejection of invalid gathers or existing output files.
