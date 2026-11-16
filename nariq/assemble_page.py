"""Verify matching platform archives and assemble a local release candidate."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import zipfile

here = Path(__file__).resolve().parent
tag = (here / "TAG").read_text().strip()
linux, windows, output = sys.argv[1:]
out = Path(output)
if out.exists() and any(out.iterdir()):
    raise SystemExit("assemble.sh: output must be empty")
archives = []
revisions = set()
for directory, platform, ext in ((linux, "linux", "tar.gz"), (windows, "win", "zip")):
    if directory == "-":
        continue
    folder = Path(directory)
    program = f"nari-llama.cpp-{tag}-bin-{platform}-cuda-12.8-x64"
    runtime = f"cudart-nari-llama.cpp-{tag}-{platform}-cuda-12.8-x64"
    expected = {f"{program}.{ext}", f"{runtime}.{ext}"}
    sums = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if not match or match[2] in sums:
            raise SystemExit("assemble.sh: invalid or duplicate checksum entry")
        sums[match[2]] = match[1]
    if set(sums) != expected:
        raise SystemExit(f"assemble.sh: expected both {tag} {platform} archives, got {sorted(sums)}")
    for name, sha in sums.items():
        path = folder / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha:
            raise SystemExit(f"assemble.sh: {name} does not match its SHA256SUMS")
        archives.append((path, sha))
    path = folder / f"{program}.{ext}"
    if platform == "linux":
        with tarfile.open(path) as archive:
            metadata = json.load(archive.extractfile(f"{program}/build-info.json"))
    else:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("build-info.json").decode("utf-8-sig"))
    revision = metadata.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or metadata.get("dirty") is not False:
        raise SystemExit("assemble.sh: candidate requires a clean, identified source revision")
    revisions.add(revision)
if len(revisions) != 1:
    raise SystemExit("assemble.sh: Linux and Windows archives have different source revisions")
out.mkdir(parents=True, exist_ok=True)
lines = []
for path, sha in archives:
    shutil.copy2(path, out / path.name)
    lines.append(f"{sha}  {path.name}")
for name in ("run.sh", "run.ps1") if windows != "-" else ("run.sh",):
    shutil.copy2(here / name, out / name)
    lines.append(f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}")
(out / "SHA256SUMS").write_text("\n".join(lines) + "\n", newline="\n")
print("\n".join(lines))
