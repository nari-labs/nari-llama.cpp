#!/usr/bin/env bash
# Put the two upload sets together locally; nothing is uploaded (docs/llamacpp.md, "Release files").
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

verify() {  # dir: check the files a SHA256SUMS there lists (CRLF tolerated), print the checked names
    local dir="$1"
    [ -f "$dir/SHA256SUMS" ] || { echo "assemble.sh: no SHA256SUMS in $dir" >&2; exit 1; }
    (cd "$dir" && tr -d '\r' < SHA256SUMS | sha256sum --quiet -c -) || { echo "assemble.sh: $dir does not match its SHA256SUMS" >&2; exit 1; }
    tr -d '\r' < "$dir/SHA256SUMS" | sed 's/^[0-9a-f]\{64\}  //'
}

case "${1:-}" in
page)
    [ $# -eq 4 ] || { echo "usage: assemble.sh page LINUX_DIR WINDOWS_DIR|- OUT" >&2; exit 2; }
    linux="$2" windows="$3" out="$4"
    prepare_out "$out"
    files=($(verify "$linux"))
    if [ "$windows" = - ]; then
        echo "assemble.sh: Linux-only candidate; omitting run.ps1" >&2
    else
        files+=($(verify "$windows"))
    fi
    for f in "${files[@]}"; do
        case "$f" in *"-$TAG-"*) ;; *) echo "assemble.sh: $f is not a $TAG file" >&2; exit 1 ;; esac
        src="$linux/$f"; [ -f "$src" ] || src="$windows/$f"
        cp -f "$src" "$out/$f"
    done
    cp -f "$HERE/run.sh" "$out/"
    files+=(run.sh)
    if [ "$windows" != - ]; then
        cp -f "$HERE/run.ps1" "$out/"
        files+=(run.ps1)
    fi
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
