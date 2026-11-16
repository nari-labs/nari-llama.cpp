"""Provision the pinned UI; packaging must fail instead of falling back to latest."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

root = Path(__file__).resolve().parents[1]
pin = json.loads((root / "nariq/ui.json").read_text())
cache = root / "nariq-build/ui"
cache.mkdir(parents=True, exist_ok=True)
archive = cache / "dist.tar.gz"
if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != pin["sha256"]:
    part = archive.with_suffix(".part")
    with urllib.request.urlopen(pin["url"], timeout=120) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out)
    if hashlib.sha256(part.read_bytes()).hexdigest() != pin["sha256"]:
        part.unlink()
        raise SystemExit("UI archive checksum mismatch")
    part.replace(archive)
with tempfile.TemporaryDirectory(dir=cache) as tmp:
    dest = Path(tmp)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if (target != dest.resolve() and dest.resolve() not in target.parents) or not (member.isfile() or member.isdir()):
                raise SystemExit(f"Unexpected UI archive entry: {member.name}")
        tar.extractall(dest)
    if not (dest / "index.html").is_file() or not (dest / "index.html").stat().st_size:
        raise SystemExit("UI archive contains no index.html")
    output = root / "tools/ui/dist"
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(dest, output)
print(f"Prepared UI {pin['version']} ({pin['sha256']})")
