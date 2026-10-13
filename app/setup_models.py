"""First-run model download with a visible progress bar (called by START.bat)."""
from __future__ import annotations

import sys

from . import paths
from .cuda_setup import choose_whisper_model, detect_hardware


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    from .config import Config

    cfg = Config()
    hw = detect_hardware()
    if hw["cuda"]:
        print(f"  Відеокарта: {hw['gpu_name'] or 'NVIDIA'} ({hw['vram_mb']} МБ) — розпізнавання на GPU.")
    else:
        print("  ⚠ Відеокарту NVIDIA не знайдено — розпізнавання працюватиме на процесорі (повільніше).")
    if cfg.get("stt_engine") == "deepgram" and cfg.get("deepgram_api_key"):
        print("  Використовується Deepgram — локальна модель не потрібна.")
    names = [choose_whisper_model(cfg.get("whisper_model", "auto"), hw)[0]]
    if hw["cuda"] and names[0].startswith("large"):
        names.append("small.en")  # small CPU fallback in case the GPU fails during a lecture
    from faster_whisper.utils import _MODELS
    from huggingface_hub import snapshot_download

    allow = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
    for name in names:
        repo = _MODELS.get(name, name)
        print(f"  Модель розпізнавання мовлення: {name}")
        try:
            snapshot_download(repo, cache_dir=str(paths.MODELS_DIR / "whisper"), allow_patterns=allow,
                              local_files_only=True)
            print("    вже завантажена ✓")
            continue
        except Exception:
            pass
        print("    завантаження (один раз, може тривати кілька хвилин)…")
        try:
            snapshot_download(repo, cache_dir=str(paths.MODELS_DIR / "whisper"), allow_patterns=allow)
            print("    готово ✓")
        except Exception as e:
            print(f"  ❌ Не вдалося завантажити модель: {e}")
            print("     Перевірте підключення до інтернету та запустіть START.bat ще раз.")
            if name == names[0]:
                return 1
    ensure_cloudflared()
    return 0


def ensure_cloudflared() -> None:
    exe = paths.CLOUDFLARED_EXE
    if exe.exists() and exe.stat().st_size > 1_000_000:
        return
    print("  Завантаження cloudflared (для публічного посилання)…")
    import httpx

    from .tunnel import DOWNLOAD_URL

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
                    bar = "#" * int(30 * done / total)
                    print(f"\r    [{bar:<30}] {done * 100 // total:3d}%", end="", flush=True)
        tmp.replace(exe)
        print("\n    готово ✓")
    except Exception as e:
        print(f"\n  ⚠ Не вдалося завантажити cloudflared: {e}. Програма спробує ще раз під час роботи.")


if __name__ == "__main__":
    sys.exit(main())
