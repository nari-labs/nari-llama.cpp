# NariQ runtime and build tools

This fork adds NariQ trellis-coded Qwen3.8 weights to llama.cpp. It includes TRL1-TRL6 tensor types, the converter, the model loader, CPU operations, and CUDA kernels from the vendored Apache-2.0 libtrellis. The NariQ additions retain Apache-2.0 licensing; upstream llama.cpp retains its MIT license. See [NOTICE](NOTICE).

The initial integration is based on upstream b12431 (`a946eb541a6d3051496bdcc451a0fa85772b8dfd`). `TAG` reserves `b12431-nari2` for the next candidate; it is not a published release. No workflow in this fork creates releases or pushes container images.

## Build this checkout

Linux with an NVIDIA CUDA toolkit:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j 4
./build/bin/llama-server -m /path/to/model.gguf --mmproj /path/to/projector.gguf --jinja
```

libtrellis is included under `ggml/src/ggml-cuda/libtrellis`; no other checkout is needed. CUDA trellis kernels require compute capability 8.0 or newer. CUDA 12.8 is used for packaged binaries; use a Linux NVIDIA driver of 570 or newer (Windows 572 or newer). Source builds can select architectures with `-DCMAKE_CUDA_ARCHITECTURES=89`, for example.

CPU-only, including experimental Apple Silicon builds:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure -R test-trellis
./build-cpu/bin/llama-server -m /path/to/model.gguf -ngl 0
```

The model and projector are supplied separately. NariQ Metal, Vulkan and HIP GPU kernels are not implemented. Apple Silicon CPU builds are not a claim of Apple GPU support or full-model qualification. LoRA adapters are rejected for NariQ models, because the coded and banded tensors do not yet implement adapter application.

## Convert an existing NariQ checkpoint

`convert_hf_to_gguf.py` detects `quantization_config.quant_method = native_trellis` and uses `conversion/trellis.py`. Its input must already contain NariQ codes, scales and layout metadata; this converter does not quantize ordinary BF16 weights.

```sh
python convert_hf_to_gguf.py /path/to/native-trellis-checkpoint --outfile model.gguf --outtype bf16 --no-mtp
```

The `bf16` setting applies to the uncoded tensors. Coded tensors retain their 1-6-bit widths, and the file carries `trellis.version = 1`. Vision projection uses the base model's separate BF16 projector.

## Linux archives and Docker image

Run from the repository root. These commands compile the current checkout on CPUs; no GPU or model is needed to build.

```sh
# CUDA 12.8 development environment, with CMake 3.28+, Ninja and Python 3:
bash nariq/build_linux.sh nariq-dist

# Or use the pinned CUDA 12.8 container and export the archives locally:
docker buildx build -f nariq/Dockerfile --target artifacts \
  --build-arg NARIQ_SOURCE_REVISION="$(git rev-parse HEAD)" \
  --output type=local,dest=nariq-dist .

# Build a local serving image (does not push it):
docker build -f nariq/Dockerfile -t nari-llama.cpp:local .
```

The two archives contain programs/backends and CUDA runtime libraries, respectively. Extract each into a shared destination using `tar -xzf ARCHIVE --strip-components=1 -C DESTINATION`; each archive has its own top-level directory. The program archive includes source revision, dirty status, source hashes and compiler/build settings in `build-info.json`, plus both licenses. `SHA256SUMS` covers both archives. Set `JOBS` to limit build parallelism; `CUDA_ARCHS` can reduce a development build's architecture set. A reduced-architecture build must not replace the all-architecture release candidate.

The package build downloads the UI pinned in `ui.json`, checks its SHA256, and fails if it cannot supply the chat page. It does not fall back to a mutable `latest` UI.

## Windows archives

The `NariQ Windows artifacts` workflow builds this checkout with clang CPU variants and MSVC/nvcc CUDA 12.8. It runs synthetic CPU tests and uploads ZIPs and checksums as Actions artifacts. It does not publish a release. A Windows NVIDIA machine is still required for CUDA execution testing; hosted build runners do not provide one.

## Assemble a release candidate locally

```sh
# Extract the Linux and Windows Actions artifacts into separate directories first.
bash nariq/assemble.sh page linux-artifacts windows-artifacts candidate
# Linux-only candidate:
bash nariq/assemble.sh page linux-artifacts - candidate
```

The assembler verifies input hashes, requires the program/runtime pairs from the same clean source revision, checks the candidate tag, and produces a combined checksum file plus `run.sh` and `run.ps1`. For a Linux-only candidate it omits `run.ps1`. Nothing is uploaded. The launchers' default download URLs will not work until the matching runtime release and model files are published. For local acceptance tests, `TRELLIS_RELEASE_URL` and `TRELLIS_MODEL_URL` can point to a local HTTP server.

`run.sh` is Linux x86-64 only, and fits context to available GPU memory. It limits CPU threads in constrained containers because coded embeddings currently execute on CPU. `run.ps1` is the Windows x86-64 counterpart. Both pin the four model files by SHA256. Model file revisions and hashes remain unchanged by this integration.

## Checks and release gates

```sh
python3 -m unittest discover -s nariq/tests -v
ctest --test-dir build-cpu --output-on-failure \
  -R 'test-(trellis|trellis-metadata|gguf|quantize-fns|chat|grammar-parser|sampling)$'
```

The CPU workflow runs Linux and Apple Silicon checks. CUDA build workflows run on relevant pull requests or manual dispatch and store artifacts only. The copied upstream publication workflows have been removed so a branch push cannot publish an upstream-style release.

See [release-readiness.md](release-readiness.md) for the platform support boundary and validation still required before publication.
