"""Make pip-installed NVIDIA DLLs (cuBLAS, cuDNN) visible to CTranslate2 on Windows,
and detect hardware."""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_done = False


def register_cuda_dlls() -> list[str]:
    global _done
    added: list[str] = []
    if _done or sys.platform != "win32":
        return added
    _done = True
    roots = []
    try:
        roots += site.getsitepackages()
    except Exception:
        pass
    roots.append(str(Path(sys.prefix) / "Lib" / "site-packages"))
    seen = set()
    for r in roots:
        nv = Path(r) / "nvidia"
        if not nv.is_dir():
            continue
        for sub in nv.iterdir():
            b = sub / "bin"
            if b.is_dir() and str(b) not in seen:
                seen.add(str(b))
                try:
                    os.add_dll_directory(str(b))
                except Exception:
                    pass
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
                added.append(str(b))
    return added


def detect_hardware() -> dict:
    """Return {'cuda': bool, 'gpu_name': str, 'vram_mb': int}."""
    register_cuda_dlls()
    info = {"cuda": False, "gpu_name": "", "vram_mb": 0}
    try:
        import ctranslate2

        info["cuda"] = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        info["cuda"] = False
    if info["cuda"]:
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.strip().splitlines()
            if out:
                name, mem = out[0].rsplit(",", 1)
                info["gpu_name"] = name.strip()
                info["vram_mb"] = int(float(mem.strip()))
        except Exception:
            pass
    return info


def choose_whisper_model(requested: str, hw: dict) -> tuple[str, str, str]:
    """(model, device, compute_type)."""
    if hw.get("cuda"):
        if requested and requested != "auto":
            return requested, "cuda", "float16"
        # large-v3 needs ~3.5 GB VRAM in fp16; turbo is ~8x faster at near-identical English accuracy.
        model = "large-v3" if hw.get("vram_mb", 0) >= 8000 else "large-v3-turbo"
        return model, "cuda", "float16"
    if requested and requested != "auto" and not requested.startswith("large"):
        return requested, "cpu", "int8"
    # medium.en is often slower than real time on laptop CPUs; small.en keeps up.
    return "small.en", "cpu", "int8"
