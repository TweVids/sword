"""Ollama lifecycle helpers for notebooks and containers without systemd."""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .defaults import DEFAULT_B_MODEL


def api_url(host):
    """Accept either host:port or an HTTP(S) URL."""
    return (host if "://" in host else f"http://{host}").rstrip("/")


def is_local_host(host):
    return urlsplit(api_url(host)).hostname in ("127.0.0.1", "localhost", "::1")


def default_models_dir():
    return "/marimo/.ollama" if Path("/marimo").is_dir() else str(Path.home() / ".ollama" / "models")


def _install_dir():
    return Path.home() / ".local" / "share" / "sword" / "ollama"


def _find_binary():
    candidates = [shutil.which("ollama"), _install_dir() / "bin" / "ollama",
                  Path("/usr/bin/ollama"), Path("/usr/local/bin/ollama")]
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def ensure_ollama_binary():
    """Reuse Ollama or install the official Linux archive into a user-writable prefix."""
    binary = _find_binary()
    if binary:
        return binary
    if platform.system() != "Linux":
        raise RuntimeError("Automatic Ollama installation requires Linux; install Ollama or configure a remote host.")
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if not arch:
        raise RuntimeError(f"Unsupported Ollama architecture: {platform.machine()}")
    if not shutil.which("tar") or not shutil.which("zstd"):
        packages = [p for p in ("tar", "zstd") if not shutil.which(p)]
        if os.geteuid() != 0 or not shutil.which("apt-get"):
            raise RuntimeError(f"Install required archive tools: {', '.join(packages)}")
        subprocess.run(["apt-get", "update"], check=True, timeout=300)
        subprocess.run(["apt-get", "install", "-y", *packages], check=True, timeout=300)
    prefix = _install_dir()
    prefix.mkdir(parents=True, exist_ok=True)
    url = f"https://ollama.com/download/ollama-linux-{arch}.tar.zst"
    print(f"[Ollama] Downloading {url}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sword-ollama-") as tmp:
        archive = Path(tmp) / "ollama.tar.zst"
        with requests.get(url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(prefix)], check=True, timeout=300)
    binary = _find_binary()
    if not binary:
        raise RuntimeError(f"Ollama binary missing after extracting to {prefix}")
    return binary


def test_ollama_binary():
    """Install if necessary and verify that the binary can execute."""
    try:
        binary = ensure_ollama_binary()
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
        print(f"[Ollama] {binary}: {(result.stdout + result.stderr).strip()}", flush=True)
        return result.returncode == 0
    except Exception as exc:
        print(f"[Ollama] Binary check failed: {exc}", flush=True)
        return False


def cgroup_limit_gb():
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(name).read_text().strip()
            if value == "max":
                return None
            size = int(value)
            return size / 1024**3 if size < 1 << 60 else None
        except (OSError, ValueError):
            continue
    return None


def _is_up(host):
    try:
        return requests.get(f"{api_url(host)}/api/version", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def _log_tail(log_path):
    try:
        with open(log_path, "rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 6000))
            return log.read().decode("utf-8", errors="replace")
    except OSError:
        return "(log unavailable)"


def start_ollama_server(models_dir=None, host="127.0.0.1:11434", pull_model=None,
                        wait_s=60, log_path=None):
    """Install/start a local server, reuse a running server, and wait for its API."""
    models_dir = models_dir or default_models_dir()
    log_path = log_path or str(Path(tempfile.gettempdir()) / "ollama.log")
    proc = None
    try:
        if _is_up(host):
            print(f"[Ollama] Reusing server at {api_url(host)}", flush=True)
        else:
            if not is_local_host(host):
                raise RuntimeError(f"Remote Ollama server is unavailable: {api_url(host)}")
            binary = ensure_ollama_binary()
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            limit = cgroup_limit_gb()
            if limit is not None:
                print(f"[Ollama] Container RAM limit: {limit:.1f} GiB", flush=True)
            env = os.environ.copy()
            env.update({"OLLAMA_HOST": urlsplit(api_url(host)).netloc,
                        "OLLAMA_MODELS": str(models_dir), "OLLAMA_NUM_PARALLEL": "1",
                        "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_FLASH_ATTENTION": "1",
                        "OLLAMA_KEEP_ALIVE": "5m"})
            # Sequential judging uses one inference slot to leave VRAM for training.
            kwargs = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": subprocess.CREATE_NO_WINDOW}
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.Popen([binary, "serve"], env=env, stdout=log,
                                        stderr=subprocess.STDOUT, **kwargs)
            print(f"[Ollama] Server PID {proc.pid}; log: {log_path}", flush=True)
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited ({proc.returncode})\n{_log_tail(log_path)}")
                if _is_up(host):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"Server not ready after {wait_s}s\n{_log_tail(log_path)}")
            for line in _log_tail(log_path).splitlines():
                if any(key in line.lower() for key in ("cuda", "gpu", "vram", "library=")):
                    print(f"[Ollama GPU] {line}", flush=True)
        if pull_model:
            return pull_ollama_model(pull_model, host=host, models_dir=models_dir)
        return True
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        print(f"[Ollama] Startup failed: {exc}", flush=True)
        return False


def pull_ollama_model(model=DEFAULT_B_MODEL, host="http://127.0.0.1:11434",
                      models_dir=None, stall_s=120):
    """Pull with streamed notebook progress and per-layer byte-stall detection."""
    print(f"[Ollama] Pulling {model} from {api_url(host)}", flush=True)
    if models_dir:
        print(f"[Ollama] Model storage for a locally managed server: {models_dir}", flush=True)
    started = time.monotonic()
    last_print = started - 5
    last_status = None
    layers = {}
    try:
        with requests.post(f"{api_url(host)}/api/pull", json={"model": model, "stream": True},
                           stream=True, timeout=(10, stall_s)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                now = time.monotonic()
                status = event.get("status", "")
                total, done = event.get("total", 0), event.get("completed", 0)
                digest = event.get("digest", status)
                if total:
                    previous, changed = layers.get(digest, (done, now))
                    if done > previous:
                        changed = now
                    layers[digest] = (done, changed)
                    if done < total and now - changed >= stall_s:
                        raise RuntimeError(f"Layer {digest} stalled for {stall_s}s without new bytes")
                    if now - last_print >= 5 or done == total:
                        print(f"  {status}: {done / 1e9:.2f}/{total / 1e9:.2f} GB ({done / total:.1%})", flush=True)
                        last_print = now
                elif status != last_status:
                    print(f"  {status}", flush=True)
                last_status = status
                if status == "success":
                    print(f"[Ollama] {model} ready after {(now - started) / 60:.1f} min", flush=True)
                    return True
        raise RuntimeError("Pull stream ended without a success event")
    except Exception as exc:
        print(f"[Ollama] Pull failed for {model}: {exc}", flush=True)
        return False
