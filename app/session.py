"""Incremental, crash-safe transcript files for a session."""
from __future__ import annotations

import os
import time
from pathlib import Path

from . import paths


def fmt_srt_time(sec: float) -> str:
    sec = max(0.0, sec)
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_clock(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_entry(index: int, t0: float, t1: float, text: str) -> str:
    if t1 <= t0:
        t1 = t0 + 1.0
    return f"{index}\n{fmt_srt_time(t0)} --> {fmt_srt_time(t1)}\n{text.strip()}\n\n"


def build_srt(lines: list[dict], key: str = "uk") -> str:
    out, n = [], 0
    for ln in lines:
        text = ln.get(key)
        if not text:
            continue
        n += 1
        out.append(srt_entry(n, ln["t0"], ln["t1"], text))
    return "".join(out)


class SessionWriter:
    def __init__(self, started_at: float | None = None, base: Path | None = None):
        started_at = started_at or time.time()
        name = time.strftime("%Y-%m-%d_%H-%M", time.localtime(started_at))
        base = base or paths.TRANSCRIPTS_DIR
        d = base / name
        i = 2
        while d.exists() and any(d.iterdir()):
            d = base / f"{name}_{i}"
            i += 1
        d.mkdir(parents=True, exist_ok=True)
        self.dir = d
        self.srt_index = 0
        self._files = {
            "en": open(d / "transcript_en.txt", "a", encoding="utf-8"),
            "uk": open(d / "transcript_uk.txt", "a", encoding="utf-8"),
            "bi": open(d / "bilingual.txt", "a", encoding="utf-8"),
            "srt": open(d / "subtitles_uk.srt", "a", encoding="utf-8"),
        }

    def _w(self, key: str, text: str) -> None:
        f = self._files[key]
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    def write_en(self, line: dict) -> None:
        self._w("en", f"[{fmt_clock(line['t0'])}] {line['en']}\n")

    def write_translated(self, line: dict) -> None:
        self._w("uk", f"[{fmt_clock(line['t0'])}] {line['uk']}\n")
        self._w("bi", f"[{fmt_clock(line['t0'])}]\nEN: {line['en']}\nUK: {line['uk']}\n\n")
        self.srt_index += 1
        self._w("srt", srt_entry(self.srt_index, line["t0"], line["t1"], line["uk"]))

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
