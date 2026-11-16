#!/usr/bin/env bash
# Build and package this checkout. Does not fetch another source tree or publish.
# Usage: nariq/build_linux.sh OUT [TAG]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
[ $# -ge 1 ] && [ $# -le 2 ] || { echo "usage: $0 OUT [TAG]" >&2; exit 2; }
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"
TAG="${2:-$(cat "$HERE/TAG")}"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid tag" >&2; exit 2; }
CUDA="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
# run.sh and run.ps1 name the CUDA 12.8 archives.
[ "$CUDA" = 12.8 ] || { echo "release packaging requires CUDA 12.8 (found $CUDA); use nariq/Dockerfile" >&2; exit 2; }
BUILD="${NARIQ_BUILD_DIR:-$ROOT/nariq-build/linux}"
NAME="nari-llama.cpp-$TAG-bin-linux-cuda-$CUDA-x64"
CUDART="cudart-nari-llama.cpp-$TAG-linux-cuda-$CUDA-x64"
ARCHS="${CUDA_ARCHS:-50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real}"
args=("-DCMAKE_CUDA_ARCHITECTURES:STRING=$ARCHS")
python3 "$HERE/prepare_ui.py"
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
    -DLLAMA_BUILD_BORINGSSL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF "${args[@]}"
cmake --build "$BUILD" -j "${JOBS:-4}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME" "$STAGE/$CUDART"
cp -a "$BUILD/bin/." "$STAGE/$NAME/"
cp "$ROOT/LICENSE" "$STAGE/$NAME/LICENSE"
cp "$HERE/LICENSE" "$STAGE/$NAME/LICENSE-NariQ"
cp "$HERE/NOTICE" "$STAGE/$NAME/NOTICE-NariQ"
cp "$ROOT/ggml/src/ggml-cuda/libtrellis/LICENSE" "$STAGE/$NAME/LICENSE-libtrellis"
cp "$HERE/README-release.md" "$STAGE/$NAME/README.md"
for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
    cp -L "${CUDA_HOME:-/usr/local/cuda}/lib64/$lib" "$STAGE/$CUDART/"
done
python3 "$HERE/provenance.py" "$ROOT" "$BUILD" > "$STAGE/$NAME/build-info.json"
cp "$STAGE/$NAME/build-info.json" "$OUT/build-info-linux.json"
tar -czf "$OUT/$NAME.tar.gz" -C "$STAGE" "$NAME"
tar -czf "$OUT/$CUDART.tar.gz" -C "$STAGE" "$CUDART"
(cd "$OUT" && sha256sum "$NAME.tar.gz" "$CUDART.tar.gz" > SHA256SUMS)
cat "$OUT/SHA256SUMS"
