# NariQ release readiness

This checklist describes the fork candidate, not a published release. Historical tests of the previous b12375-based runtime are evidence for the implementation, not validation of this newer upstream revision.

## Linux/NVIDIA release gates

- [ ] CPU CI and CUDA artifact builds pass on the candidate commit.
- [ ] Test the exact packaged artifact against the existing BF16-decode reference: text, tools, vision, long context, all three weight budgets, and concurrent requests.
- [ ] Recheck full and partial GPU offload on the new build. Partial offload was tested successfully on A10/L4 in the previous runtime, including 16 GiB budgets and a 124K-token needle; it is not an entirely unimplemented feature.
- [ ] Exercise one real 16/24 GB GeForce card before claiming consumer qualification. Historical datacenter checks covered A10, L4, L40S, A100, H100, B200 and RTX PRO 6000.
- [ ] Verify candidate checksums and clean-download launcher behavior with the exact model files. Confirm the served build reports the candidate commit.
- [ ] Confirm the model-card commands, candidate tag, driver requirements and supported platforms match the artifacts.
- [ ] Obtain publication approval. Build workflows never publish releases or container images.

These are runtime correctness checks. No scored benchmark needs to be resumed or relaunched for this migration. Changes to a scored run still require fresh explicit approval.

## Platform boundary

| Platform | Implemented | Qualification needed |
| --- | --- | --- |
| Linux NVIDIA, compute capability 8.0+ | Coded linear CUDA kernels; coded embeddings on CPU | Candidate artifact parity and runtime checks |
| Windows NVIDIA | CPU/CUDA source and native build recipe | Native build plus real GPU execution; earlier Wine CPU checks do not establish CUDA compatibility |
| macOS Apple Silicon CPU | Portable scalar decoder with generic NEON product path | ARM CI, then full-model execution on a sufficiently large Mac |
| Apple Metal | No custom trellis kernels | Separate backend implementation and hardware validation |
| AMD HIP / Vulkan | No custom trellis kernels | Separate backend implementations; do not advertise NariQ GPU acceleration |
| Older NVIDIA / CPU-only | CPU fallback | Slow execution; no GPU acceleration promise |

Apple GPU support requires dispatch for the trellis operations, 1-6-bit tile decoding, scales, permutation gathers and Hadamard transforms, decode/prefill products that retain compressed weights, and correct fallback and tensor placement. A CPU build passing on macOS is not a substitute. Hosted ARM runners can run small numerical tests but do not have enough memory for meaningful 27B-model qualification.

## Correctness safeguards in this migration

- Reject LoRA attachment to NariQ models rather than silently skipping adapter deltas or dereferencing a missing unbanded embedding.
- Require supported format metadata and validate side descriptors before loading coded weights.
- Reject Linux launcher use on unsupported operating systems and architectures before downloading model files.
- Build packages from this checkout and record their source provenance.

## Further performance work

GPU embedding lookup would remove a CPU dependency, especially for the 2-bit banded embedding. Faster ARM decoding and batched CUDA decode are separate optimization projects. The Linux launcher already limits CPU threads in constrained containers; this mitigates the known oversubscription slowdown but does not add a GPU embedding implementation.
