# NariQ release readiness

This checklist describes the fork candidate, not a published release. Exact-candidate CUDA results, artifact hashes and remaining blockers are recorded in [cuda-qualification.md](cuda-qualification.md). Historical tests of the previous b12375-based runtime alone do not qualify this revision.

## Linux/NVIDIA release gates

- [x] Linux and Windows CUDA artifact builds and available CPU checks pass for the frozen candidate. Apple CPU CI remains unverified because hosted runner capacity was unavailable.
- [x] Exercise the exact packaged artifact: text, tools, vision, 26K context, all three weight budgets and eight concurrent requests. The BF16 references also pass the functional suite.
- [x] Diagnose the reference mismatch and verify same-weight kernel parity. Direct-GGUF F32 references pass all three budgets at mean KL <= 0.001 and top-1 >= 98%, including prefill-reference/decode comparisons. The reference generator and CPU tests are included.
- [ ] Review BF16 cross-precision acceptance: corrected once-rounded BF16 references still exceed the fixed KL limit. Preserve the original failures and the corrected diagnostics; do not label the BF16 gate passed. See [cuda-qualification.md](cuda-qualification.md).
- [x] Recheck full and partial GPU offload on the exact candidate. H100 passed explicit 16/65-layer offload. L4 with 16 GiB available passed automatic 49/65-layer offload in a 128K context and retrieved a needle from a 123,323-token prompt.
- [ ] Exercise one real 16/24 GB GeForce card before claiming consumer qualification. Historical datacenter checks covered A10, L4, L40S, A100, H100, B200 and RTX PRO 6000.
- [x] Verify candidate checksums and clean-download Linux launcher behavior with the exact model files. The served build reports `b12440-459d5976a`, matching the clean artifact revision.
- [ ] Confirm the model-card commands, candidate tag, driver requirements and supported platforms match the artifacts.
- [ ] Obtain publication approval. Build workflows never publish releases or container images.

These are runtime correctness checks. No scored benchmark needs to be resumed or relaunched for this migration. Changes to a scored run still require fresh explicit approval.

## Platform boundary

| Platform | Implemented | Qualification needed |
| --- | --- | --- |
| Linux NVIDIA, compute capability 8.0+ | Coded linear CUDA kernels; coded embeddings on CPU | Functional checks passed; numerical acceptance and real GeForce qualification remain open |
| Windows NVIDIA | CPU/CUDA implementation; native artifact build and CPU startup checks passed | Real GPU execution and exact-package parity; a native build does not establish CUDA runtime correctness |
| macOS Apple Silicon CPU | Portable scalar decoder with generic NEON product path | ARM CI, then full-model execution on a sufficiently large Mac |
| Apple Metal | No custom trellis kernels | Separate backend implementation and hardware validation |
| AMD HIP / Vulkan | No custom trellis kernels | Separate backend implementations; do not advertise NariQ GPU acceleration |
| Older NVIDIA / CPU-only | CPU fallback | Slow execution; no GPU acceleration promise |

Apple GPU support requires dispatch for the trellis operations, 1-6-bit tile decoding, scales, permutation gathers and Hadamard transforms, decode/prefill products that retain compressed weights, and correct fallback and tensor placement. A CPU build passing on macOS is not a substitute. Hosted ARM runners can run small numerical tests but do not have enough memory for meaningful 27B-model qualification.

See [device-support.md](device-support.md) for the broader device matrix and implementation work needed beyond NVIDIA CUDA. Successful Linux and Windows artifact builds establish buildability and package startup; GPU execution remains a separate qualification gate.

## Correctness safeguards in this migration

- Reject LoRA attachment to NariQ models rather than silently skipping adapter deltas or dereferencing a missing unbanded embedding.
- Require supported format metadata and validate side descriptors before loading coded weights.
- Reject Linux launcher use on unsupported operating systems and architectures before downloading model files.
- Build packages from this checkout and record their source provenance.

## Further performance work

GPU embedding lookup would remove a CPU dependency, especially for the 2-bit banded embedding. Faster ARM decoding and batched CUDA decode are separate optimization projects. The Linux launcher already limits CPU threads in constrained containers; this mitigates the known oversubscription slowdown but does not add a GPU embedding implementation.
