"""Cloudflare Quick Tunnel (no account): exposes the viewer page at https://*.trycloudflare.com."""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from typing import Optional

from . import paths

log = logging.getLogger("tunnel")

DOWNLOAD_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


# ----------------------------------------------------------------------------- job object
_job = None


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """Put the child in a Windows Job object so it dies when this app dies (even if killed)."""
    global _job
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if _job is None:
            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class BASIC(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class EXT(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k32.CreateJobObjectW.restype = wintypes.HANDLE
            job = k32.CreateJobObjectW(None, None)
            info = EXT()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            _job = job
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(0x0100 | 0x0001, False, proc.pid)  # SET_QUOTA | TERMINATE
        k32.AssignProcessToJobObject(wintypes.HANDLE(_job), wintypes.HANDLE(h))
        k32.CloseHandle(wintypes.HANDLE(h))
    except Exception:
        log.debug("job object setup failed", exc_info=True)


# ----------------------------------------------------------------------------- tunnel

class Tunnel:
    def __init__(self, port: int):
        self.port = port
        self.url = ""
        self.state = "off"  # off | downloading | starting | ok | error
        self.message = ""
        self.download_pct = 0
        self.proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def ensure_binary(self) -> bool:
        exe = paths.CLOUDFLARED_EXE
        if exe.exists() and exe.stat().st_size > 1_000_000:
            return True
        self.state = "downloading"
        self.message = "Завантаження cloudflared…"
        import httpx

        tmp = exe.with_suffix(".part")
        try:
            with httpx.stream("GET", DOWNLOAD_URL, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or 1
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 16):
                        f.write(chunk)
                        done += len(chunk)
                        self.download_pct = int(done * 100 / total)
            tmp.replace(exe)
            return True
        except Exception as e:
            self.state = "error"
            self.message = f"Не вдалося завантажити cloudflared (перевірте інтернет): {e}"
            log.error(self.message)
            return False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tunnel")
        self._thread.start()

    def _run(self) -> None:
        backoff = 3
        while not self._stop.is_set():
            if not self.ensure_binary():
                self._stop.wait(15)
                continue
            self.state = "starting"
            self.message = "Створення публічного посилання…"
            self.url = ""
            try:
                self.proc = subprocess.Popen(
                    [str(paths.CLOUDFLARED_EXE), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                _kill_with_parent(self.proc)
            except Exception as e:
                self.state = "error"
                self.message = f"Не вдалося запустити cloudflared: {e}"
                self._stop.wait(backoff)
                continue
            started = time.time()
            for line in self.proc.stdout:  # type: ignore[union-attr]
                if self._stop.is_set():
                    break
                m = URL_RE.search(line)
                if m and not self.url:
                    self.url = m.group(0)
                    self.state = "ok"
                    self.message = "Посилання активне"
                    backoff = 3
                    log.info("public URL: %s", self.url)
                if "failed to" in line.lower() and not self.url:
                    self.message = line.strip()[-200:]
            if self._stop.is_set():
                break
            self.state = "error"
            self.url = ""
            self.message = "Тунель обірвався — перезапуск (посилання зміниться!)" if time.time() - started > 30 \
                else "Не вдалося створити тунель (перевірте інтернет / антивірус). Повтор…"
            self._stop.wait(backoff)
            backoff = min(60, backoff * 2)

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.state = "off"
        self.url = ""
