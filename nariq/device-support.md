# NariQ device coverage

NariQ currently implements its coded operations on CPU and NVIDIA CUDA. Other llama.cpp backends do not automatically support the new tensor types and operations. A working CPU fallback, native GPU acceleration, and a tested downloadable package are separate support claims.

The desktop/server coverage target is x86-64 and ARM64 CPUs, Apple GPUs, NVIDIA GPUs, AMD GPUs and Intel GPUs. Minimum hardware capabilities, available memory, supported OS versions and tested devices must be specified before making a release support claim. Mobile and browser targets are outside the initial scope; their additional work is listed below for completeness.

## Implementation and qualification matrix

| Device class | Current NariQ path | Missing for supported release coverage |
| --- | --- | --- |
| Linux/Windows x86-64 CPU | Coded products and embedding lookups; numerical tests and package startup checks | Standalone CPU packages and launcher selection without the CUDA runtime download; representative CPU qualification |
| Apple Silicon CPU | Portable decoder and generic NEON product path | ARM/macOS CI completion, full-model hardware checks, macOS packaging and launcher |
| Intel Mac CPU | Portable x86 CPU implementation | Darwin x86-64 build and execution checks; packaging if included in the supported OS matrix |
| Linux/Windows ARM64 CPU | Portable CPU implementation, not qualified | Native builds and tests, ARM64 artifacts, dependency and launcher support |
| NVIDIA, compute capability 8.0+ | CUDA coded products; coded embeddings run on CPU | Exact-package GPU correctness checks, consumer GPU coverage, full/partial offload and multi-GPU checks |
| Older NVIDIA GPUs | Coded products fall back to CPU | Verify mixed-backend fallback; native acceleration requires kernels that do not depend on the current sm_80+ implementation |
| NVIDIA on Linux ARM64 | No ARM64 release package | Native CUDA toolchain/build, runtime compatibility and hardware qualification |
| Apple GPU | No NariQ Metal implementation | Metal kernels, backend dispatch and tensor placement, macOS build/package and hardware checks |
| AMD GPU | No NariQ Vulkan kernels or native HIP implementation | Vulkan implementation for broad desktop coverage; HIP kernel port for a native ROCm path; capability checks and hardware qualification |
| Intel GPU | No NariQ Vulkan or SYCL implementation | Vulkan implementation for broad desktop coverage; a separate SYCL implementation if that backend is promised; hardware qualification |
| Mobile GPUs / CPUs | No qualified mobile NariQ path or package | Android/iOS build integration, appropriate GPU backend, memory limits, device tests and application packaging |
| Browser / other accelerators | No qualified NariQ path | Explicit WebGPU or accelerator backend implementation, build/runtime integration and device-specific tests |

## Work shared by new GPU backends

1. Implement TRL1-TRL6 tile decoding, scales, permutation gathers and the required Hadamard transforms for coded products. Cover single-token decode and batched prefill, including banded output heads.
2. Keep weights compressed during execution, with bounded temporary storage. Expanding the entire model to BF16/F32 would lose the memory benefit the format is intended to provide.
3. Implement capability detection, operation dispatch and placement of weights, side tensors and indices. Test CPU fallback for unsupported operations and devices, including partial offload.
4. Extend the numerical test harness to allocate, upload, execute and read back tensors on each backend. The current trellis numerical test explicitly uses the CPU backend; it does not test GPU kernels.
5. Validate all three published weight budgets against the fixed decoded reference: text/logits, tools, vision, long context and concurrent requests. Build success and short generation checks do not establish this parity.
6. Supply platform packages, dependency checks, backend selection, download integrity and clean-machine startup checks.

Coded embedding lookups currently execute on CPU even with CUDA. GPU ordinary/banded embedding lookup is a further performance improvement; a correctly tested CPU embedding fallback can provide functional support first. LoRA remains explicitly unsupported and requires separate adapter application work if included in the feature contract.

## Suggested implementation order

1. Establish portable CPU packages and fallback tests across the supported OS/architecture matrix.
2. Add Metal for Apple GPUs and Vulkan for AMD/Intel desktop coverage, with backend numerical tests developed alongside the kernels.
3. Qualify NVIDIA consumer and datacenter hardware using the exact release packages. Add native HIP/SYCL optimizations where the support contract requires those backends.
4. Add mobile, browser and other accelerators as explicit targets with their own memory and capability requirements.

The existing CUDA implementation uses NVIDIA-specific matrix instructions and asynchronous copies. Enabling a HIP CMake option alone does not provide the missing AMD kernels. Similarly, changing a macOS build flag does not supply Metal kernels.

No release should claim device coverage based only on the presence of an upstream backend. Record the artifact revision, device, OS, driver and test evidence for each supported configuration.

## Maintaining a complete runtime fork

The implementation is already integrated into the source tree and can build without another checkout. Broader device coverage still requires substantial backend work. Maintaining the fork also needs the following:

- **Regression coverage:** retain relevant upstream model, backend and server checks, including ordinary GGUF models. Add conversion fixtures, malformed-format cases, backend numerical tests and end-to-end NariQ checks. Seven selected CPU tests are a starting point rather than the full regression suite.
- **Stable format compatibility:** lock the existing tensor IDs, metadata version and on-disk layout with golden fixtures. TRL1-TRL6 currently occupy fork-specific IDs 44-49; inspect upstream enum changes for collisions during every sync. Existing model files must remain readable.
- **Upstream integration:** document the upstream base, keep backend changes separable, and run a compatibility gate before adopting upstream revisions. Track changes to loader placement, graph construction, backend capabilities and public APIs.
- **Complete user entry points:** maintain fork-specific install/build instructions, artifact names and backend selection. Check CLI/server behavior, tools, vision and context/offload modes with the supported models. Clearly document unsupported features such as NariQ LoRA and direct NariQ encoding through `llama-quantize`.
- **Release ownership:** provide reproducible build inputs, provenance, checksums, licenses, platform dependency requirements and a published support matrix. Qualification must refer to the exact candidate artifacts, with publication kept separate from building them.

The largest new implementation work for the initial scope is Metal and Vulkan. Native HIP and SYCL implementations are additional projects if those specific APIs, rather than device coverage through Vulkan, become release requirements. GPU embeddings, more ARM SIMD decoding and vendor-specific kernel tuning can follow a correct, qualified baseline.
