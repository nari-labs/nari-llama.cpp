# CUDA candidate qualification

Status as of 2026-11-17: the BF16 failure investigation is complete; publication remains on hold. A reference-construction bug is fixed, and all three budgets pass corrected F32 runtime-parity checks at the unchanged limits. BF16 cross-precision checks remain above the KL limit and are not marked passed. Native Windows GPU execution and a real 16/24 GB GeForce test are also outstanding.

## Frozen artifacts

These checks use the Linux Actions artifact from run `43847930005`, paired with the Windows artifact from run `43847930030`. Both record clean source revision `459d5976a2fc7cef569b09d29fe60bb4a722efe4`, the PR merge revision containing runtime commit `70b6c0a02eda62828a7764faaf2c006321e55b4c`. The reserved tag is `b12431-nari2`; it is not published. Later documentation edits are not a claim that newer binaries were tested.

| File | SHA256 |
| --- | --- |
| Linux program archive | `53b4db31ccacac89a657c658a7419cb92fcde9536c77449dddc5d797196b9cf0` |
| Linux CUDA runtime archive | `aa8c79b02243f9bef16f6501b6265ad55a608308b4e79e4c8a8e351be91dbff3` |
| Windows program archive | `c5c48f7ed6a8812bb1648c8cceff8f153e61fd1d939ee3f10d3462e0188f3c65` |
| Windows CUDA runtime archive | `fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d` |
| Tested Linux launcher | `96d5b21664daa3b5a193bfcafd9149a67b655d44cf1c3c9d01063ad2b0aa6fd3` |
| Paired Windows launcher | `4211e59c25e13c2cfefd4eb42c05d3786d13c7af252121f7ad4231613d6df8f6` |

The assembler accepted the matching clean program/runtime pairs. The Linux launcher downloaded the real archives and all four pinned model files from an isolated HTTP fixture, checked their hashes, extracted them, and started the packaged server without an injected library search path. The web UI returned valid HTML. A subsequent comment-only launcher correction adds the already-documented `GLIBCXX_3.4.30` requirement and removes an unsupported blanket RHEL 9 compatibility claim; executable lines are unchanged. No runtime or model was published.

The model files retain the launcher's pinned hashes. BF16 reference hashes are:

| Budget | Original assembled BF16 reference SHA256 |
| --- | --- |
| 3.0 | `bd05d3d2b4625244363cb5a4211ee3f04048a8d27792476cb61c9210e44443f5` |
| 2.5 | `d9894c56fa8fc037f1ff3deb47ac5c8b37f5eb972b230936a8f6e932829a2b82` |
| 2.0 | `f1094c6aaa4b44eb793ff7569ae36bdc6db565b7d3f21e375bb69355383be7a1` |

## Functional checks

| Hardware | Scope | Result |
| --- | --- | --- |
| H100 80 GB | All three coded budgets and their BF16 references: thinking and non-thinking chat, tool call/result, a fixed tool loop, 26K-token needle, two vision cases | Passed |
| H100 80 GB | Eight concurrent requests for each budget; explicit 16/65-layer offload for 3-bit | Passed |
| L4 24 GB | All three budgets: the same serving suite; startup/generation with 16 GiB available | Passed |
| L4 with 16 GiB available | 3-bit, 128K context, automatic partial offload, 123,323-token needle prompt | Passed; 49/65 layers on GPU |
| A10, A100 40 GB, L40S, B200, RTX PRO 6000 | 2-bit initial/warm startup, GPU placement, UI and generation | Passed |
| Native Windows | Artifact build and CPU checks | Passed; GPU execution untested |
| Real 16/24 GB GeForce | Exact-package execution | Not available |

Logs confirm 65/65 layers offloaded in full-GPU cases and 16/65 in the explicit partial-offload case. Smaller-memory tests use a separate CUDA allocation to leave 16 GiB available; they do not establish GeForce qualification. Initial/warm startup was 46/6 seconds on A100 and 42/6 seconds on B200. These are individual functional observations, not performance benchmarks.

Linux test containers used Ubuntu 22.04 with driver 610.57.04, except A100 with 620.70.06. The documented minimum drivers remain Linux 570 and Windows 572; those minimum versions were not tested here. The seven launcher/packaging unit tests passed. Apple CPU runner capacity and Metal qualification are separate work.

## Numerical check

The original reference files came from assembled BF16 checkpoints. Follow-up investigation found intermediate rounding in their construction; they are not exact once-rounded decodes of the final coded GGUFs. See [VALIDATION.md](VALIDATION.md) for corrected reference generation. The fixed input SHA256 is `b46edf2f31842f098af0f1fdd221564bfc42ebb1aa200caeca66a388332d4474`. `llama-perplexity` evaluates two 512-token chunks, scoring 510 positions with its native compressed KL format. That format truncates small reference probabilities, so these original values are approximate KL, not a full-vocabulary sum. This short corpus is a runtime diagnostic, not a model-quality benchmark.

Before execution, the gate was fixed at mean full-vocabulary KL <= 0.001 and top-1 agreement >= 98%. The reference uses microbatch 512; coded paths use 512 and 1. A BF16 microbatch-1 comparison records batch-path sensitivity. Results and failures are retained without selecting successful retries.

| Budget | BF16 batch-path KL | Coded prefill KL / top-1 | Coded decode KL / top-1 | Gate |
| --- | ---: | ---: | ---: | --- |
| 3.0 | 0.010398 | 0.012491 / 98.824% | 0.016661 / 98.824% | Failed |
| 2.5 | 0.001867 | 0.004462 / 99.020% | 0.003841 / 99.020% | Failed |
| 2.0 | 0.012758 | 0.034907 / 97.843% | 0.029563 / 97.843% | Failed |

The 2-bit coded paths also fall below the top-1 threshold. The BF16 batch-path comparison itself exceeds the KL limit for every budget. This shows sensitivity to numerical execution paths, but does not make the coded-model gate pass.

The follow-up checks retain the same corpus and numerical limits. Identical-path BF16 self-comparisons give top-1 agreement of 100% and reported mean KL of -0.000001 for all three budgets, consistent with reference-format rounding. Comparing coded decode with a BF16 reference also generated at microbatch 1 reduces the discrepancy, but every budget still fails the KL limit.

| Budget | Matched decode KL | Top-1 | Gate |
| --- | ---: | ---: | --- |
| 3.0 | 0.002783 | 99.608% | Failed |
| 2.5 | 0.002250 | 99.020% | Failed |
| 2.0 | 0.021945 | 99.020% | Failed |

A separate regression check compares the frozen candidate directly with the previous `b12375-nari1` runtime, using identical coded weights, inputs and microbatch sizes. All six comparisons pass the unchanged numerical limits:

| Budget | Prefill KL / top-1 | Decode KL / top-1 |
| --- | ---: | ---: |
| 3.0 | -0.000001 / 100.000% | 0.000000 / 100.000% |
| 2.5 | 0.000000 / 100.000% | -0.000001 / 100.000% |
| 2.0 | 0.000000 / 100.000% | 0.000000 / 100.000% |

The previous program archive SHA256 is `076b757a333d73188ae058c888af65d6153cf6b38ead6b584e14aa14fdb5fc49`; its CUDA runtime archive SHA256 is `90d13cf14dd778e41f10ee829f4b996c7b66711d6f64f746fea0bb3b6158b704`. Both were verified before use.

No numerical migration regression is observed on this sample. This is evidence that the BF16 discrepancy predates the migration, not a pass for the BF16 gate. The reference-generation investigation below distinguishes this failure from coded-kernel arithmetic. These original failures remain in the record; thresholds and corpus have not been changed.

Reference generation:

```sh
./llama-perplexity -m bf16-decode.gguf -f validation.txt \
  -c 512 --chunks 2 -b 512 -ub 512 -t 8 -ngl 99 -fa on \
  --no-warmup --kl-divergence-base reference.kld
```

Comparison uses the same settings with `-m coded.gguf --kl-divergence`; repeat with `-ub 1` for the decode path. The tool stores reference probabilities in its native compressed format, which can produce tiny negative KL estimates from rounding. The threshold is applied to the reported mean without changing it after observing results.

## Reference investigation (2026-11-17)

The reference-selection bug is reproducible at the weight level. On the 2-bit `blk.0.ffn_down` matrix (5120 by 17408), an independent float64 decoder and the runtime's F32 CPU decoder differ by only `8.54e-8` relative RMS. The old BF16 reference differs from that mathematical decode by `0.002342` relative RMS. Only 74.945% of its entries equal the independently decoded matrix rounded once to BF16. Its bytes instead exactly match the assembled BF16 checkpoint, whose construction rounds before applying tuned scales and then rounds again. Emulating that intermediate rounding reproduces 99.988% of the assembled entries. It is a valid assembled checkpoint, but an incorrect exact-decode reference for kernel parity.

`decode_reference.py` now builds the reference from the actual coded GGUF, applies its transforms and gathers, restores vocabulary order, and preserves uncoded tensors. It records source, decoder and output hashes. It provides both once-rounded BF16 and F32 output, so BF16 weight rounding can be measured separately from coded-kernel arithmetic. Synthetic known-value tests exercise both output types, permutations, vocabulary bands and halfway rounding; the CPU workflow includes those tests.

The original 2-bit failure was also repeated using uncompressed F32 logits and a float64 full-vocabulary KL reduction on the H100. Token IDs, vocabulary order, shapes and finite values were checked before comparison. This independently confirms that compression was not responsible for the large original failure:

| Same frozen candidate, original references | Full-vocabulary mean KL | Top-1 |
| --- | ---: | ---: |
| Old BF16 prefill vs coded prefill | 0.034912947 | 97.843% |
| Old BF16 prefill vs old BF16 decode | 0.012769477 | 98.039% |
| Old BF16 decode vs coded decode | 0.021939589 | 99.020% |
| Coded prefill vs coded decode | 0.000257244 | 99.608% |
| Coded decode vs ungrouped coded decode | 0.000000000 | 100.000% |

Ten positions contribute 95.92% of the first row's KL; the worst is token index 460 with KL 11.4406. These outliers remain in the evidence. Eight sampled CUDA linears, covering attention projections, the FFN and output bands, have relative RMS error of about `2.04e-4` to `2.12e-4` against float64 products using CPU-decoded weights. An additional requested `blk.0.attn_output` probe reported a missing tensor and is excluded from that coverage. Grouping produces bit-identical decode logits on this sample. No serving-kernel change was made on the basis of the reference mismatch.

Corrected full-model diagnostics use a B200 (the F32 model requires more than 80 GB), the same frozen Linux runtime, the same corpus and all 510 scored positions. The original numerical limits are unchanged: mean KL <= 0.001 and top-1 >= 98%. Both a prefill reference compared with decode and matched-microbatch comparisons are retained. These are separate corrected-reference diagnostics, not replacements for the recorded original failures.

### Corrected-reference results

All nine F32 comparisons below pass the unchanged mean-KL and top-1 limits. This is a same-weight runtime-parity check: F32 avoids the additional BF16 weight rounding and BF16 matrix-multiply input rounding. It does not establish bitwise equality or model quality beyond this short sample.

| Budget | F32 prefill vs coded prefill KL / top-1 | F32 prefill vs coded decode KL / top-1 | Matched F32 decode KL / top-1 |
| --- | ---: | ---: | ---: |
| 3.0 | 0.000167679 / 100.000% | 0.000055761 / 100.000% | 0.000018325 / 100.000% |
| 2.5 | 0.000122308 / 99.804% | 0.000100768 / 99.804% | 0.000116562 / 99.804% |
| 2.0 | 0.000237184 / 99.608% | 0.000607337 / 99.412% | 0.000083811 / 100.000% |

The corrected BF16 references still fail the original prefill-reference gate. The remaining difference includes BF16 weight rounding and execution-path arithmetic; the BF16 references themselves exceed the KL limit between prefill and decode. No tolerance was relaxed, and these results are not marked passed:

| Budget | BF16 prefill vs coded prefill KL | BF16 prefill vs coded decode KL | Matched BF16 decode KL | BF16 prefill vs BF16 decode KL |
| --- | ---: | ---: | ---: | ---: |
| 3.0 | 0.001608518 | 0.001546879 | 0.000866830 | 0.001363188 |
| 2.5 | 0.004374524 | 0.005208110 | 0.001791077 | 0.001614137 |
| 2.0 | 0.021630961 | 0.019642403 | 0.004356498 | 0.007609271 |

The evidence supports a reference-construction bug plus a cross-precision comparison problem, rather than a CUDA migration regression. The fix is the direct-GGUF reference generator and explicit separation of kernel parity from BF16 rounding diagnostics. Serving kernels, model weights, corpus and numerical thresholds are unchanged. The original BF16 acceptance gate remains open; F32 parity passing does not silently redefine it.

The reference decoder is the frozen package's `libggml-base.so`, SHA256 `4cc83c49635ec41f56eedb768bbab14254ef972e71b98a3e0370961ada27d8dd`. B200 checks use driver 610.57.04. Generated reference hashes are:

| Budget | Once-rounded BF16 SHA256 | F32 SHA256 |
| --- | --- | --- |
| 3.0 | `4e6f3acb90e27f37fb05fe0069a25d45445206683eeebbdaeeba47d165e0bbaf` | `36399a2f72cfb3979c51c9d9aac6cc10417e2cf27c37edeffe2ab49a786d9293` |
| 2.5 | `d8903338ab873f6985684a3a7bbc85fc0cdd15a58cc3428dc6230292fb873d62` | `f771e1cd87cfd8bf6181f3103621436561e1efd4a035120c26f90a797dff2a29` |
| 2.0 | `f218b6264c9796d43f1db5f91c84c4c4d6179463474e2b41dbfab172705aeffc` | `121fa4d052ce5d08dbd7bd1f281f904b885fef831977a3dc203defc96b0751d5` |

Both reference types preserve uncoded tensors; their sizes are 53,808,282,368 and 107,594,988,288 bytes respectively. Ten local Linux reference/packaging tests pass. The new CPU workflow wiring has not been executed on macOS in this investigation.

## Evidence and scope

The run records retain input and artifact hashes, command lines, server logs, completion state and cost estimates. The fixed tool-loop outputs are trimmed functional records, not complete agent traces. No scored benchmark was started, resumed or retried.

Three initial attempts were stopped after the UI probe omitted `Accept-Encoding: gzip` and received the upstream server's expected HTTP 415 response. Their incomplete records remain available. The corrected probe decompresses and validates the UI response; it does not alter the runtime, weights, prompts or sampling. GPU-placement logging was also enabled.

The compute authorization remains capped at $50 cumulatively. Initial validation reserved $47.18 at full timeouts; those containers were confirmed stopped before reconciling unused reservations. The investigation carried a $10 prior-work allowance, then reserved each new job's full timeout, bounded CPU/memory allocation and startup margin. Stopped jobs retain their original reservation records and receive conservative settled allowances; the budget was not reset.

The six investigation attempts include four completed diagnostics and two incomplete infrastructure attempts (a result-bookkeeping typo and a missing Python dependency). Both failures are retained. The initial 13 attempts are also retained. Cumulative estimated job compute is $13.59, including failed attempts; the final conservative allowance including prior setup/margin is $24.81, below $50. These are elapsed-time estimates using [Modal's listed rates](https://modal.com/pricing), not invoices. All investigation and earlier validation GPU containers are stopped. Raw logits and manifests have been downloaded and hash-verified; generated dense models were temporary diagnostic files, not published artifacts.

## Remaining release gates

- Review the remaining BF16 cross-precision acceptance requirement. Its cause is diagnosed, and direct-GGUF F32 kernel parity passes, but the fixed BF16 limit is still not met. Keep both sets of results.
- Execute the exact Windows package on an NVIDIA GPU.
- Test a real 16/24 GB GeForce card before claiming consumer qualification.
- Review the final model-card commands and published platform claims against the frozen artifacts. Runtime requirements are Linux driver 570+, Windows driver 572+, and the documented system libraries.
- Obtain explicit publication approval. No release, tag, container image or model upload was published by this validation.

Apple Silicon/Metal qualification is separate. Vulkan remains deferred.
