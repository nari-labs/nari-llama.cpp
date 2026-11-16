# nari-llama.cpp

The Nari Labs fork of llama.cpp, with NariQ trellis-coded Qwen3.8-27B inference. It includes TRL1-TRL6 tensors, checkpoint conversion, the model loader, CPU operations and vendored CUDA/libtrellis kernels.

This is a development candidate based on upstream b12431. No NariQ release is published yet. Build this checkout to use the custom tensor format.

## Build and run

For Linux with NVIDIA CUDA:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j 4
./build/bin/llama-server -m /path/to/model.gguf --mmproj /path/to/projector.gguf --jinja
```

For CPU execution, including experimental Apple Silicon builds:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF
cmake --build build-cpu -j 4
./build-cpu/bin/llama-server -m /path/to/model.gguf -ngl 0
```

Supply the NariQ GGUF and optional BF16 vision projector separately. The converter accepts existing native-trellis checkpoints; it does not quantize ordinary BF16 checkpoints into NariQ.

## Device support

- NariQ coded products run on CPU or NVIDIA GPUs of compute capability 8.0 or newer. Coded embeddings currently execute on CPU.
- Linux and Windows CUDA artifact builds and CPU startup checks have passed. Exact-package GPU qualification remains required.
- Apple Silicon CPU qualification is pending. NariQ Metal, Vulkan, HIP and SYCL acceleration is not implemented.
- Desktop/server coverage across Apple, NVIDIA, AMD, Intel and x86-64/ARM64 CPUs is the target described in the [device coverage plan](nariq/device-support.md).
- LoRA adapters on NariQ weights are unsupported and rejected explicitly.

## Guides

- [Builds, packages, conversion and launchers](nariq/README.md)
- [NariQ tensor format](nariq/FORMAT.md)
- [Device coverage and fork maintenance plan](nariq/device-support.md)
- [Release qualification checklist](nariq/release-readiness.md)

Linux/Docker packaging and Windows artifact workflows build this checkout. CI uploads build artifacts only; it does not publish releases or push container images. The launchers' default release URLs become usable only after a matching release and model files are published.

## Upstream and licensing

General llama.cpp documentation is available in [docs](docs/) and the [upstream project](https://github.com/ggml-org/llama.cpp). Upstream installers and stock distributions do not include this fork's custom NariQ tensor implementation.

Upstream llama.cpp retains its [MIT license](LICENSE). NariQ additions and libtrellis retain [Apache-2.0 licensing](nariq/LICENSE); see [NOTICE](nariq/NOTICE).
