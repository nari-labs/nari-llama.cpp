# nari-llama.cpp: Linux x86-64, CUDA 12.8

llama.cpp with the NariQ trellis tensor types (TRL1-TRL6) that the Qwen3.8-27B trellis GGUFs use. Stock llama.cpp,
Ollama and LM Studio cannot load those files.

- Unpack this archive and the matching `cudart-nari-llama.cpp-...` archive into the same folder.
- Needs:
  - an NVIDIA driver of 570 or newer;
  - glibc 2.34 or newer (Ubuntu 22.04, Debian 12, RHEL 9 or newer);
  - libgomp.
- GPUs: real code for RTX 30xx, RTX 40xx / L4 and RTX 50xx / RTX PRO 6000. Any other card of compute capability 8.0
  or newer (A100, H100) compiles PTX on the first run.
- Without a GPU, the CPU backend runs, with the best variant for the processor chosen at start.

```bash
./llama-server -m Qwen3.8-27B-NariQ-3bit.gguf --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf
```

This opens a chat page and an OpenAI-compatible API at http://127.0.0.1:8080. Without `-c`, the context is the longest
that free GPU memory holds, up to 262,144 tokens.

`build-info.json` gives the source revision and hashes. The llama.cpp parts are under
the MIT license (`LICENSE`); NariQ additions and libtrellis are Apache-2.0 (`LICENSE-NariQ`, `NOTICE-NariQ`, `LICENSE-libtrellis`).
