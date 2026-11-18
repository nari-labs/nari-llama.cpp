# Native Metal coded products

This fork implements `GGML_OP_TRELLIS` on Metal for Apple desktop/server use. TRL1-TRL6 weights remain compressed. The GPU applies input gathers and scales, block or whole-slice Hadamard transforms (including Paley factors 12, 20 and 68), coded products, inverse output transforms, output scales and gathers. Both token decode and batched prefill are supported, including frequency-banded output heads.

Ordinary and banded coded embeddings (`GGML_OP_TRELLIS_GET_ROWS`) use the CPU backend. The loader's existing operation probes place those weights and side tensors in host buffers, while supported coded linear weights and their side tensors use Metal buffers. The scheduler transfers activations between backends. GPU embedding support is not implemented. Vulkan, mobile and browser execution are outside this implementation.

## Build and run

Use Apple Silicon macOS with Xcode command line tools. The tested configuration is an M3 Pro, 18 GB unified memory, macOS 14.6 and Xcode 16.2. Other Macs remain unqualified. Native products require Apple GPU family 7 capabilities and SIMD-group reductions; other devices and unsupported operation layouts fall back to CPU. The kernels use the [32-lane SIMD groups described by Apple](https://developer.apple.com/videos/play/wwdc2022/10159/).

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-metal -j 4
./build-metal/bin/llama-completion --list-devices
./build-metal/bin/llama-server -m /path/to/model.gguf -ngl 99 -c 2048 -b 32 -ub 32 --jinja
```

Supply a compatible NariQ GGUF separately. Check available unified memory and model hashes before loading a model. Select fewer GPU layers or use `-ngl 0` for CPU execution when necessary. The number of offloaded layers is reported during model loading. Supply the pinned BF16 projector with `--mmproj /path/to/projector.gguf` for vision; vision parity remains unqualified. Bounded full-model text checks are described below.

The default build embeds shader source and compiles it at runtime. To build a compiled library instead, configure with `-DGGML_METAL_EMBED_LIBRARY=OFF` and build the default target, or explicitly build both `ggml-metal-lib` and the executable targets. Keep `default.metallib` beside the executables when moving that build. No macOS downloadable package or launcher is provided yet.

## Memory and performance

The product kernels decode integer-grid entries directly from compressed 16-by-16 tiles. Bit-width-specialized vector products use SIMD-group reductions; batched products can use FP32 SIMD-group matrix multiplication. The codec value uses exact integer arithmetic: four six-bit fields are summed with a byte multiplication whose partial sums cannot carry. Eight neighboring states reuse packed-word loads. Block Hadamards up to 4096 entries use threadgroup memory; larger and whole-slice transforms use staged dispatches and barriers. Additional per-operation scratch is bounded by `2 * min(tokens, 32) * (K + N) * sizeof(float)`, plus the output tensor. Scratch is included in backend allocation sizing and reused across token chunks. No dense model weights are cached.

This implementation still needs further tuning. Larger-than-32 batches are processed in chunks; each matrix tile reuses decoded weights across up to 16 input columns. Throughput, long context, concurrent serving and consumer memory limits need model-level qualification before any performance or release support claim.

## Numerical and fallback checks

```sh
ctest --test-dir build-metal --output-on-failure \
  -R 'test-(trellis|trellis-metal|trellis-metal-private|trellis-metadata|gguf|quantize-fns|chat|grammar-parser|sampling)$'
MTL_SHADER_VALIDATION=1 ./build-metal/bin/test-trellis MTL0
GGML_METAL_SHARED_BUFFERS_DISABLE=1 ./build-metal/bin/test-trellis MTL0
```

The device harness creates backend allocations, uploads coded weights, side tensors and activations, executes Metal graphs and reads back outputs through backend APIs. It compares against independently decoded CPU dense products with tolerance `2e-4 * (1 + abs(reference))`. It covers all six bit widths, identity/block/whole-slice transforms, Paley factors, fused output parts and shards, permutations, non-contiguous input rows, four-dimensional input batches, and token counts 1, 7 and 35. Capability checks reject unsupported element strides and transform descriptors.

Mixed scheduler graphs check ordinary and banded CPU embeddings feeding two Metal output bands, concatenation and vocabulary gathers, against CPU-only execution. These tests assert that embeddings stay on CPU and both coded heads run on Metal. Missing requested Metal devices fail the test rather than silently selecting CPU.

The `NariQ CPU and Metal checks` workflow builds the compiled Metal shaders on hosted macOS and checks the CPU reference. Hosted build success is not GPU execution evidence. Its optional `metal_gpu` dispatch input runs shader-validated numerical checks on a self-hosted runner labelled `self-hosted`, `macOS`, `ARM64`, `metal`. This is a functional test job, not a scored benchmark.

## Local evidence and limits

On 2026-11-16, local development changes based on `120ae4f4a3a5bb5c03618c26d3b4de6e42e5824a` passed on the M3 Pro configuration above:

- 144 native Metal product cases: six bit widths, eight layout/transform modes and three batch sizes.
- Six mixed CPU/Metal graph comparisons covering ordinary/banded embeddings and decode/prefill output heads.
- Shader validation and private Metal buffer variants.
- Embedded-source and compiled-library Metal builds and numerical checks.
- The eight selected CTests, including format metadata, GGUF, quantization functions, chat, grammar and sampling regressions.
- 28 selected upstream F32/Q4_0 Metal matrix-product tests, plus server and CLI startup/device enumeration.
- A separate CPU-only build and trellis test. Packaging unit tests passed their one applicable test; six platform-specific cases were skipped on macOS.

Subsequent bounded 27B checks downloaded and hash-verified the pinned 2-bit, 2.5-bit and 3-bit GGUFs. Each model was loaded separately on this 18 GB Mac. CPU and Metal used the same five-token prompt and generated eight continuation tokens, capturing all 248320 vocabulary logits at 12 positions: five prompt positions and seven continuation positions. Greedy choices matched at every captured position. Maximum KL divergence was 1.16e-9, 3.77e-10 and 1.85e-9 respectively.

Additional checks used a 12-token prompt and microbatch 32 to exercise SIMD-group matrix prefill through the full models. Each captured 19 full-vocabulary rows and eight continuation choices. All top-1 choices matched; maximum KL divergence was 1.10e-8, 1.49e-9 and 1.49e-9 for the three budgets. Model hashes were rechecked before these captures. These are short functional comparisons, not general model-quality qualification.

The first qualified build measured about 1.6-1.7 tokens/s across the three budgets. Its server prefill measured about 14-17 tokens/s; decode fell to about 1.4 tokens/s around 51K context. These observations belong to the archived initial build.

On 2026-11-17, CPU sampling and Metal System Trace located about 90% of GPU time in two-bit vector products; the main CPU thread waited for GPU completion. The baseline GPU device utilization was about 95%. New single-token tiles reuse eight consecutive codec states and split K across four lane groups. Batched products decode four K tiles between barriers. Direct exact codec arithmetic replaced the lookup buffer, and no floating-point storage precision was reduced. Texture lookup and larger vector tiles were slower and were discarded.

The revised build passed 162 native cases (including a K=272, N=304 tail case), six mixed scheduler comparisons, shader validation, shared/private storage, embedded/compiled shaders, nine selected CTests and ordinary F32/Q4_0 matrix checks. All three full models passed the same 12-token prompt and eight-token forced-continuation checks against the unchanged CPU reference: top-1 and greedy agreement were 100%, with maximum KL 8.49e-9, 1.93e-9 and 1.84e-9. Short captures measured 4.90, 4.04 and 3.82 tokens/s for 2-bit, 2.5-bit and 3-bit respectively. These bounded comparisons do not establish general model quality or long-context numerical parity.

A two-repetition warm `llama-bench` check with 512 prompt tokens, 32 generated tokens, microbatch 32 and full Metal offload measured 26.90 prompt tokens/s and 4.94 decode tokens/s.

The requested 50 tokens/s single-sequence decode target has not been reached. The M3 Pro has [150 GB/s memory bandwidth](https://support.apple.com/en-ie/117736); reading roughly 6-7 GB of compressed weights each step alone gives an optimistic bandwidth ceiling around 20-25 tokens/s. Achieving 50 would require avoiding that full per-token read, for example through effective multi-token verification. The current path remains below that bandwidth ceiling.

On 2026-11-18, eight larger redesign configurations tested register matrix multiplication, cooperative split-K reductions, contiguous K chunks, and global or banked on-chip codebooks. Every bounded CPU/Metal logits comparison passed, but none improved throughput. Five repetitions of 32 generated tokens measured 4.98 tokens/s for the restored implementation, 3.51 for the matrix-instruction redesign and 3.37 for cooperative split-K. The earlier frozen-runtime control measured 4.54 tokens/s; all redesigns were slower than both controls. Codebook variants were slower in short captures. The original implementation was restored byte for byte. Final checks passed for all three pinned models, including shader validation and private buffers. Detailed Instruments GPU counters reported an unsupported counter profile on this Mac; precise ALU and DRAM stall attribution remains unresolved. No local inference service or DeepSWE job was restarted.

The local server also passed a deterministic arithmetic response, a streamed tool call, and use of a tool result.

### DeepSWE subset

A fixed two-task subset of [DeepSWE at revision `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`](https://github.com/datacurve-ai/deep-swe/tree/0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea) ran with local 2-bit Metal inference and remote CPU task containers. It used Pier 0.3.1 at `4d3c14041d16443f3f9f460dcdf23629994a304e`, native Claude Code 2.1.309, one attempt per task, no trial retries, the original three-hour agent and 30-minute verifier timeouts, and native tests/grading. Each task container had two CPUs and 8 GB memory. No remote GPU inference server was started.

The local variant used 65536 context and FP16 KV. Sampling was fixed at temperature 1, top-p 0.95, top-k 20, min-p 0.05 and seed 323198294, with the server's recorded sampler order. Agent thinking was enabled, effort was `max` (mapped to the template's `xhigh`), and the output limit was 65536 tokens. This differs from the original 262K-context/BF16-KV serving recipe and covers 2 of 113 tasks.

| Task | Native reward | Feature tests passed | Existing tests passed | Agent outcome |
| --- | --- | --- | --- | --- |
| `httpx-streaming-json-iteration` | 0 | 0/108 | 1404/1404 | Three-hour timeout |
| `httpx-multipart-response-parsing` | 0 | 0/122 | 1272/1272 | Three-hour timeout |

Both submitted patches were empty. Native grading completed, while both agent traces remain incomplete and are labelled accordingly. The server ran continuously through the two attempts. This setup did not complete either coding task within its budget; numerical parity does not establish DeepSWE success or general model quality.

Source, binaries, weights, prompts, settings and grading stayed fixed during the subset. Their hashes were checked afterward, and the scored source/binary snapshots were preserved. Subsequent review restricted the native kernels to Apple GPU capabilities and corrected codec initialization for private buffers, then repeated embedded/compiled Metal numerical tests and the batched full-model comparisons. The private-buffer regression is registered as `test-trellis-metal-private`. Those results refer to the archived initial build with default shared storage. The user subsequently authorized one rerun of the same two tasks after optimization, with the original prompts, budgets, attempts, sampling and grading held fixed. The optimized-runtime rerun was cancelled at the user's request during the streaming task; multipart never started. Its agent trace is incomplete and there is no completed scored rerun result. Subsequent profiling and redesign checks left DeepSWE stopped. Credential redaction preserves the original manifest and preflight report, with separate original-to-redacted artifact hashes.

Model hashes and pinned revisions are unchanged. Vision, long-context numerical parity, concurrency, CUDA revalidation, more Apple devices and exact-package qualification remain outstanding. No release or container image was published.

## Bounded full-model captures

`test-trellis-model` is built with the tests but is never registered with CTest and never downloads weights. It limits prompts to 64 tokens, prediction to 32 tokens, context to 128 and microbatches to 32. Supply an already hash-verified GGUF:

```sh
./build-metal/bin/test-trellis-model --model /path/to/model.gguf \
  --output /path/to/cpu --gpu-layers 0 --predict 8 --ubatch 32 \
  --prompt "Write a Python function that returns the sum of two integers."
./build-metal/bin/test-trellis-model --model /path/to/model.gguf \
  --output /path/to/metal --gpu-layers 99 --predict 8 --ubatch 32 \
  --prompt "Write a Python function that returns the sum of two integers." \
  --force /path/to/cpu.json
python3 nariq/compare_logits.py /path/to/cpu /path/to/metal \
  --output /path/to/comparison.json
```

The Metal check records its own greedy choices while evaluating the reference continuation tokens. This holds model inputs fixed if generation diverges. The comparator checks complete metadata, file sizes, finite logits, full-distribution KL, top-1 agreement and artifact hashes. Its default short-check gate is maximum KL 1e-3 with every top-1 choice matching. A failed or interrupted capture must not be reported as complete. The captured throughput includes logits reads and file writes.
