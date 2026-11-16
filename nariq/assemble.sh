#!/usr/bin/env bash
# Put the two upload sets together locally; nothing is uploaded (nariq/README.md).
#
#   assemble.sh page LINUX_DIR WINDOWS_DIR OUT   # the fork's release page: build_linux.sh's archives, the Windows
#                                                # workflow's zips (its artifact, unzipped; "-" for none yet), run.sh,
#                                                # run.ps1 and one SHA256SUMS over all of them
#   assemble.sh models GGUF_DIR OUT              # the model repository's files under their release names, hard-linked,
#                                                # each checked against the sha256 that run.sh pins, and SHA256SUMS
#
# SHA256SUMS is written with LF line endings: run.sh matches whole lines, so a CRLF file (PowerShell's) would not match.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="$(cat "$HERE/TAG")"

prepare_out() {
    if [ -e "$1" ] && [ -n "$(ls -A "$1")" ]; then
        echo "assemble.sh: output must be empty: $1" >&2
        exit 2
    fi
    mkdir -p "$1"
}


case "${1:-}" in
page)
    [ $# -eq 4 ] || { echo "usage: assemble.sh page LINUX_DIR WINDOWS_DIR|- OUT" >&2; exit 2; }
    exec python3 "$HERE/assemble_page.py" "$2" "$3" "$4"
    ;;
models)
    [ $# -eq 3 ] || { echo "usage: assemble.sh models GGUF_DIR OUT" >&2; exit 2; }
    gguf="$2" out="$3"
    prepare_out "$out"
    declare -A FROM=([Qwen3.8-27B-NariQ-3bit.gguf]=qwen38-27b-3.0-permattn-trellis.gguf
                     [Qwen3.8-27B-NariQ-2.5bit.gguf]=qwen38-27b-2.5-permattn-trellis.gguf
                     [Qwen3.8-27B-NariQ-2bit.gguf]=qwen38-27b-2.0-permattn-trellis.gguf
                     [Qwen3.8-27B-NariQ-mmproj-BF16.gguf]=qwen38-27b-mmproj-bf16.gguf)
    files=()
    for name in Qwen3.8-27B-NariQ-3bit.gguf Qwen3.8-27B-NariQ-2.5bit.gguf Qwen3.8-27B-NariQ-2bit.gguf \
                Qwen3.8-27B-NariQ-mmproj-BF16.gguf; do
        want="$(grep -F "\"$name\"" "$HERE/run.ps1" | grep -o "[0-9a-f]\{64\}" | head -n 1)" || true
        [ -n "$want" ] || { echo "assemble.sh: run.ps1 pins no sha256 for $name" >&2; exit 1; }
        ln -f "$gguf/${FROM[$name]}" "$out/$name" 2> /dev/null || cp -f "$gguf/${FROM[$name]}" "$out/$name"
        got="$(sha256sum "$out/$name" | cut -d' ' -f1)"
        [ "$got" = "$want" ] || { echo "assemble.sh: $name is $got, run.ps1 pins $want" >&2; exit 1; }
        grep -q "$want" "$HERE/run.sh" || { echo "assemble.sh: run.sh does not pin $name's $want" >&2; exit 1; }
        files+=("$name")
    done
    ;;
*)
    echo "usage: assemble.sh page LINUX_DIR WINDOWS_DIR|- OUT | assemble.sh models GGUF_DIR OUT" >&2; exit 2 ;;
esac
(cd "$out" && sha256sum "${files[@]}" > SHA256SUMS)
cat "$out/SHA256SUMS"
