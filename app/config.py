"""Local configuration (config.json) and glossary (glossary.json)."""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from . import paths

DEFAULTS: dict[str, Any] = {
    "port": 8765,
    "setup_done": False,
    "gemini_api_key": "",
    # Tried in order. First = best quality, later ones = fallback when quota runs out.
    "gemini_models": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
    "gemini_rpm": 10,
    "deepl_api_key": "",
    "deepgram_api_key": "",
    # Last-resort keyless translator (Google Translate public web endpoint).
    "free_fallback_translator": True,
    "stt_engine": "whisper",  # whisper | deepgram
    "whisper_model": "auto",  # auto | large-v3 | large-v3-turbo | medium.en | small.en ...
    "audio_device": "default",
    "zoom_caption_url": "",
    "zoom_captions_enabled": True,
    "zoom_seq_state": {},  # caption token -> next seq (survives restarts)
    "tunnel_enabled": True,
    "live_partials": True,
}

# Keys that must never leave the control panel.
SECRET_KEYS = ("gemini_api_key", "deepl_api_key", "deepgram_api_key")

_lock = threading.RLock()


def _atomic_write(path, text: str) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Config:
    def __init__(self, path=paths.CONFIG_PATH):
        self.path = path
        self.data: dict[str, Any] = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        with _lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, encoding="utf-8") as f:
                        self.data.update(json.load(f))
                except Exception:
                    pass  # corrupted file -> defaults; will be rewritten on save
            for k, v in DEFAULTS.items():
                self.data.setdefault(k, v)

    def save(self) -> None:
        with _lock:
            _atomic_write(self.path, json.dumps(self.data, ensure_ascii=False, indent=2))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def update(self, values: dict[str, Any]) -> None:
        with _lock:
            for k, v in values.items():
                if k in DEFAULTS:
                    self.data[k] = v
            self.save()

    def public_view(self) -> dict[str, Any]:
        """Config for the (local) control panel: secrets are masked, never echoed in full."""
        out = {k: v for k, v in self.data.items() if k not in SECRET_KEYS and k != "zoom_seq_state"}
        for k in SECRET_KEYS:
            v = self.data.get(k) or ""
            out[k + "_set"] = bool(v)
            out[k + "_hint"] = ("…" + v[-4:]) if len(v) > 8 else ""
        return out


class Glossary:
    """English term -> preferred Ukrainian. Empty value or same as term = keep in English."""

    def __init__(self, path=paths.GLOSSARY_PATH):
        self.path = path
        self.items: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        with _lock:
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.items = {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip()}
                elif isinstance(raw, list):
                    self.items = {str(r[0]).strip(): str(r[1]).strip() for r in raw if r and str(r[0]).strip()}
            except FileNotFoundError:
                self.items = {}
            except Exception:
                self.items = {}

    def save(self, items: dict[str, str]) -> None:
        with _lock:
            self.items = {k.strip(): (v or "").strip() for k, v in items.items() if k and k.strip()}
            _atomic_write(self.path, json.dumps(self.items, ensure_ascii=False, indent=2))

    @staticmethod
    def keep_english(term: str, value: str) -> bool:
        v = (value or "").strip().lower()
        return not v or v == term.strip().lower() or v in ("keep", "keep in english", "=", "-")

    def terms(self) -> list[str]:
        return list(self.items.keys())

    def prompt_lines(self) -> str:
        lines = []
        for term, uk in self.items.items():
            if self.keep_english(term, uk):
                lines.append(f'- "{term}" → keep in English as "{term}"')
            else:
                lines.append(f'- "{term}" → "{uk}"')
        return "\n".join(lines)
