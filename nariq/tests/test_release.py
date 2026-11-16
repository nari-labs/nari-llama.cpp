import functools
import hashlib
import http.server
import io
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import unittest

RELEASE = Path(__file__).resolve().parents[1]


def command(*args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


@unittest.skipUnless(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux launcher")
class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = {**os.environ, "TRELLIS_HOME": str(self.root / "cache")}
        self.tag = (RELEASE / "TAG").read_text().strip()

    def test_invalid_arguments_do_not_download(self):
        for args in [("--bogus",), ("1.5",), ("--port",), ("--host",)]:
            result = command("bash", str(RELEASE / "run.sh"), *args, env=self.env)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.root / "cache").exists())

    def test_unsupported_architecture_stops_before_download(self):
        mock = self.root / "mock"
        mock.mkdir()
        (mock / "uname").write_text('#!/bin/sh\nif [ "$1" = -s ]; then echo Linux; else echo aarch64; fi\n')
        (mock / "uname").chmod(0o755)
        self.env["PATH"] = str(mock) + os.pathsep + self.env["PATH"]
        result = command("bash", str(RELEASE / "run.sh"), env=self.env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Linux x86-64", result.stderr)
        self.assertFalse((self.root / "cache").exists())

    def archives(self, directory, platform_name, crlf=False):
        directory.mkdir()
        ext = "zip" if platform_name == "win" else "tar.gz"
        names = [f"nari-llama.cpp-{self.tag}-bin-{platform_name}-cuda-12.8-x64.{ext}",
                 f"cudart-nari-llama.cpp-{self.tag}-{platform_name}-cuda-12.8-x64.{ext}"]
        lines = []
        for name in names:
            data = name.encode()
            (directory / name).write_bytes(data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        (directory / "SHA256SUMS").write_bytes((("\r\n" if crlf else "\n").join(lines) + "\n").encode())
        return names

    def test_assemble_checks_inputs_and_combines_windows_checksums(self):
        linux, windows, out = [self.root / x for x in ("linux", "windows", "out")]
        names = self.archives(linux, "linux") + self.archives(windows, "win", crlf=True)
        result = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), str(windows), str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(b"\r", (out / "SHA256SUMS").read_bytes())
        self.assertEqual(command("sha256sum", "-c", "SHA256SUMS", cwd=out).returncode, 0)
        self.assertTrue((out / "run.ps1").is_file())
        (linux / names[0]).write_bytes(b"corrupt")
        bad = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), str(windows), str(self.root / "bad"))
        self.assertNotEqual(bad.returncode, 0)

    def test_linux_only_candidate_omits_windows_launcher(self):
        linux, out = self.root / "linux", self.root / "out"
        self.archives(linux, "linux")
        result = command("bash", str(RELEASE / "assemble.sh"), "page", str(linux), "-", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((out / "run.ps1").exists())

    def test_launcher_download_extract_hash_and_forward_arguments(self):
        web = self.root / "web"
        web.mkdir()
        model = b"small test fixture, not model weights"
        sha = hashlib.sha256(model).hexdigest()
        script = (RELEASE / "run.sh").read_text()
        script = re.sub(r"[a-f0-9]{64}", sha, script)
        launcher = self.root / "run.sh"
        launcher.write_text(script)
        (web / "Qwen3.8-27B-NariQ-3bit.gguf").write_bytes(model)
        sums = []
        for prefix in (f"nari-llama.cpp-{self.tag}-bin", f"cudart-nari-llama.cpp-{self.tag}"):
            name = f"{prefix}-linux-cuda-12.8-x64"
            path = web / f"{name}.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                data = b'#!/bin/sh\nprintf "%s\\n" "$@"\n'
                entry = tarfile.TarInfo(f"{name}/" + ("llama-server" if prefix.endswith("-bin") else "runtime.txt"))
                entry.size = len(data)
                entry.mode = 0o755
                archive.addfile(entry, io.BytesIO(data))
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        (web / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(web)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}"
        self.env.update(TRELLIS_RELEASE_URL=url, TRELLIS_MODEL_URL=url)
        result = command("bash", str(launcher), "3bit", "--no-vision", "--", "-t", "2", "--alias", "test model", env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--alias\ntest model\n", result.stdout)
        self.assertEqual(result.stdout.splitlines().count("-t"), 1)
        self.assertEqual((self.root / "cache/models/Qwen3.8-27B-NariQ-3bit.gguf").read_bytes(), model)
        shutil.rmtree(self.root / "cache")
        (web / "Qwen3.8-27B-NariQ-3bit.gguf").write_bytes(b"corrupt")
        bad = command("bash", str(launcher), "--no-vision", env=self.env)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("does not match its sha256", bad.stderr)


class CrossPlatformTests(unittest.TestCase):
    def test_launchers_pin_identical_files_and_tag(self):
        sh, ps = [(RELEASE / name).read_text() for name in ("run.sh", "run.ps1")]
        self.assertEqual(set(re.findall(r"[a-f0-9]{64}", sh)), set(re.findall(r"[a-f0-9]{64}", ps)))
        tag = (RELEASE / "TAG").read_text().strip()
        self.assertIn(tag, sh)
        self.assertIn(tag, ps)


if __name__ == "__main__":
    unittest.main()
