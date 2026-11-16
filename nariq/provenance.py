"""Record the checkout and source hashes without needing the quantization repo."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(sys.argv[1]).resolve()


def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


try:
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
except (subprocess.CalledProcessError, FileNotFoundError):
    revision = os.environ.get("NARIQ_SOURCE_REVISION", "unknown")
    dirty = None

sources = {}
for directory in ("app", "cmake", "common", "conversion", "ggml", "gguf-py/gguf", "include", "src", "tools", "vendor", "nariq"):
    for path in sorted((root / directory).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and "node_modules" not in path.parts:
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
for filename in ("CMakeLists.txt", "convert_hf_to_gguf.py", "LICENSE"):
    sources[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
builds = {}
for directory in sys.argv[2:]:
    build = Path(directory)
    options = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^:#]+):([^=]+)=(.*)", line)
        if not match:
            continue
        key, kind, value = match.groups()
        if (kind in ("BOOL", "STRING") and key.startswith(("GGML_", "LLAMA_"))) or key in (
                "CMAKE_BUILD_TYPE", "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_C_FLAGS"):
            options[key] = value
    compilers = {}
    for file in (build / "CMakeFiles").glob("*/CMake*Compiler.cmake"):
        for key, value in re.findall(r'set\((CMAKE_(?:C|CXX|CUDA)_COMPILER_(?:ID|VERSION)) "([^"]*)"\)', file.read_text()):
            compilers[key] = value
    builds[build.name] = {"options": options, "compilers": compilers}
json.dump({"revision": revision, "dirty": dirty, "source_sha256": sources, "builds": builds}, sys.stdout, indent=2)
print()
