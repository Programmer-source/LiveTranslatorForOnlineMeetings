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
    return 0


if __name__ == "__main__":
    sys.exit(main())
