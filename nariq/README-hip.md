# HIP / ROCm (experimental)

NariQ coded linear layers (`GGML_OP_TRELLIS`) run on RDNA AMD GPUs with the weights kept compressed, for decode
and prefill, including banded output heads. Coded embedding lookups run on the CPU. There is no packaged ROCm
build; build from source.

The HIP path shares the orchestration of `ggml/src/ggml-cuda/trellis.cu`. Decode (batch 1) uses a scalar GEMV
(one thread per output row, integer dot products on the INT8 grid); prefill and batched runs on RDNA4 use a
gfx12 WMMA (matrix-core) kernel (`128x128` 2D wave-tiled for `batch >= 32, N >= 128`) that decodes each 16x16
weight tile into LDS and accumulates in fp32, paired with wave-shuffle fused blockwise Hadamard and MLP chain
kernels. The scalar path remains the correctness oracle and the fallback for odd shapes / small batches; force
it with `GGML_TRELLIS_HIP_MM=0`. Expect decode below the CUDA numbers; prefill is ~10x the scalar tier.

## Build

Needs CMake 3.18+, a ROCm installation that provides `hipcc` (tested with ROCm 10.1), and a C++17 compiler:

```sh
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DGGML_CUDA=OFF -DAMDGPU_TARGETS=gfx1201 -DLLAMA_BUILD_TESTS=ON
cmake --build build-hip --target llama-cli llama-server test-trellis -j "$(nproc)"
```

Set `-DAMDGPU_TARGETS` to your GPU (`rocminfo | grep gfx`); building only for the installed card keeps the compile
under a few minutes. Other devices and unsupported layouts fall back to the CPU.

```sh
./build-hip/bin/llama-cli -m /path/to/Qwen3.8-27B-NariQ-3.0bpw.gguf -ngl 99 -c 65536
```

Add `--mmproj /path/to/projector.gguf` for vision. Without `-c`, `llama-cli` fits the context to free device
memory and keeps 1 GiB in reserve; pass `-c N` to cap it. The default `32 GiB`-class card fits about 83K tokens of
full offload at 3.0bpw (KV is 64 KiB/token); **64K is the tested working point** and leaves room for the vision
projector.

## Checks

```sh
ctest --test-dir build-hip --output-on-failure -R trellis
./build-hip/bin/test-trellis ROCm0
```

`test-trellis ROCm0` compares HIP results with independently decoded CPU products (tolerance `2e-4 * (1 +
|reference|)`) across all bit widths, transforms, permutations, input shapes and mixed CPU/ROCm graphs.

## ROCm 10.1 crashes on AVX-512 hosts

On a Zen 4/5 host with ROCm 10.1, any HIP program dies with `SIGSEGV` (`si_addr == 0`) on its first upload or
stream creation. The cause is in ROCm, not in llama.cpp: `amd::roc::VirtualGPU` is allocated with plain `new`
(`rocclr/device/rocm/rocdevice.cpp:1900`), but this build emits 64-byte-aligned AVX-512 stores into it, and glibc
`malloc` only guarantees 16-byte alignment. Whether the object happens to land aligned is heap-layout dependent,
which is why the crash looks timing sensitive and appears or disappears between builds.

Preload a shim that aligns `operator new` to 64 bytes:

```sh
cat > /tmp/align-new.cpp <<'EOF'
#include <cstdlib>
#include <new>
static void * aligned(size_t n) {
    void * p = std::aligned_alloc(64, (n + 63) & ~size_t(63));
    if (!p && n) throw std::bad_alloc();
    return p;
}
void * operator new(size_t n)              { return aligned(n); }
void * operator new[](size_t n)            { return aligned(n); }
void   operator delete(void * p) noexcept  { std::free(p); }
void   operator delete[](void * p) noexcept{ std::free(p); }
void   operator delete(void * p, size_t) noexcept   { std::free(p); }
void   operator delete[](void * p, size_t) noexcept { std::free(p); }
EOF
g++ -O2 -shared -fPIC -o /tmp/libalignnew.so /tmp/align-new.cpp
LD_PRELOAD=/tmp/libalignnew.so ./build-hip/bin/test-trellis ROCm0
```

This is needed even for runs that look CPU-only: context fitting initialises the HIP backend to probe device
memory, so `-ngl 0` crashes the same way (use `-fit off` to skip it). Vulkan, and CUDA on NVIDIA, are unaffected.

**Recovering after a crash.** A killed or crashed HIP process can leave GPU memory accounted but unreferenced
(`mem_info_gtt_used` in `/sys/class/drm/card1/device/` stays high while no process owns it). If a display server
shares the GPU, that leak can starve the compositor and take the desktop down. Check the counter after a crash and
reboot if it does not fall back.

## Tested configuration and limits

One machine has run this: a Radeon RX 9070 XT (16 GB, gfx1201, RDNA4), ROCm 10.1, in-tree `amdgpu` on Linux 7.3,
Ryzen 7 9700X.

3.0bpw, text-only, `llama-bench` (3 repetitions), WMMA prefill on (default):

| Test | Result |
| --- | ---: |
| `pp512` | 700.6 tokens/s |
| `tg32` | 19.9 tokens/s |
| `pp512 @ d4096` / `tg32 @ d4096` | 680.8 / 19.7 tokens/s |
| `pp512` with `GGML_TRELLIS_HIP_MM=0` (scalar) | 67.5 tokens/s |
| Full GPU offload ceiling | ~83K tokens (14.3 GiB peak at 64K) |
| Full-vocab FP32 logit parity vs scalar (`ubatch=32`, 60 rows) | `mean_kl = 7.41e-07`, `top1 = 100%`, greedy identical |

Decode sits in the 15-30 tokens/s target range and is still scalar; prefill uses the matrix cores (~10x the
scalar tier, ~3.1x the initial 64x32 WMMA port) while keeping full fp32-level accuracy via hi/lo f16 activation
splitting. Fine for agent sessions and interactive use. Above the full-offload ceiling the context fit drops GPU
layers silently rather than failing; read the `offloaded N/65` log line when benchmarking.

Note for porters: the gfx12 WMMA accumulator feedback rounds with a small bias when chained across many k-blocks,
so the prefill kernel groups a few k-tiles and flushes each group into an fp32 register accumulator. Also,
`mma.cuh`'s fragment index helpers assume one wave per workgroup; the multi-wave kernel uses lane-explicit loads.
`GGML_TRELLIS_HIP_MM_KG` / `_NSUB` tune the k-tile grouping and token amortization.

Not qualified: vision, concurrent requests, other AMD GPUs, RDNA3 and older, Windows. Unknown: behaviour with
ROCm releases other than 10.1.
