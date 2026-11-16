"""Check and extract final archives, then run model-free CPU startup checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

folder = Path(sys.argv[1]).resolve()
windows = sys.platform == "win32"
suffix = ".zip" if windows else ".tar.gz"
paths = sorted([*folder.glob("nari-llama.cpp-*" + suffix), *folder.glob("cudart-nari-llama.cpp-*" + suffix)])
if len(paths) != 2:
    raise SystemExit("Expected a program archive and a CUDA runtime archive")
sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (folder / "SHA256SUMS").read_text().splitlines()}
with tempfile.TemporaryDirectory() as tmp:
    dest = Path(tmp)
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if sums.get(path.name) != digest.hexdigest():
            raise SystemExit(f"Checksum mismatch: {path.name}")
        if windows:
            with zipfile.ZipFile(path) as archive:
                archive.extractall(dest)
        else:
            with tarfile.open(path) as archive:
                archive.extractall(dest)
    executable = "llama-server.exe" if windows else "llama-server"
    binaries = list(dest.rglob(executable))
    if len(binaries) != 1:
        raise SystemExit("Missing or ambiguous server in package")
    binary = binaries[0]
    info = json.loads((binary.parent / "build-info.json").read_text(encoding="utf-8-sig"))
    if "tools/ui/dist/index.html" not in info["source_sha256"]:
        raise SystemExit("Package does not record pinned UI assets")
    runtime_dirs = [str(p.parent) for p in dest.rglob("libcudart.so.12")]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": os.pathsep.join([str(binary.parent), *runtime_dirs])}
    subprocess.run([str(binary), "--version"], env=env, check=True, timeout=60)
    cli = binary.parent / ("llama-cli.exe" if windows else "llama-cli")
    subprocess.run([str(cli), "--list-devices"], env=env, check=True, timeout=60)
    if not list(binary.parent.glob("*ggml-cpu*")):
        raise SystemExit("Package contains no CPU backend")
print("Archive checksums, extraction, provenance, UI and CPU startup checks passed")
