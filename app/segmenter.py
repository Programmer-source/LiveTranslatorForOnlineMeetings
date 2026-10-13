"""VAD-based segmentation of a continuous 16 kHz stream into 3–15 s speech segments.

Guarantees:
* every sample that VAD considers speech ends up in exactly one segment (plus padding);
* forced cuts (speech > 15 s without a pause) happen at the lowest-energy frame of the
  last 2 s, with ±200 ms overlap; the STT stage de-duplicates words at that seam.
"""
from __future__ import annotations

import bisect
import itertools
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .vad import FRAME

SR = 16000


@dataclass
class Segment:
    seq: int
    audio: np.ndarray
    abs_start: float  # seconds on the captured-sample timeline
    abs_end: float
    wall_start: float  # time.time() of first sample
    wall_end: float
    speech_end_wall: float  # wall time when the last speech frame ended (for latency)
    kind: str = "pause"  # pause | forced | eof
    cut_start: Optional[float] = None  # abs time of forced cut at our start (overlap region)
    cut_end: Optional[float] = None  # abs time of forced cut at our end
    speech_frames: int = 0
    total_frames: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.abs_end - self.abs_start


class Segmenter:
    START_TH = 0.5
    KEEP_TH = 0.35
    PRE_PAD = 0.3
    POST_PAD = 0.2
    OVERLAP = 0.2
    MAX_LEN = 15.0
    FORCE_WINDOW = 2.0
    END_SILENCE = 1.2  # always close a segment after this much silence

    def __init__(self, vad, emit: Callable[[Segment], None], seq_start: int = 0):
        self.vad = vad
        self.emit = emit
        self._seq = itertools.count(seq_start)
        self.lock = threading.Lock()
        self.buf = np.zeros(0, np.float32)
        self.buf_start = 0  # absolute sample index of buf[0]
        self.total = 0  # absolute samples received
        self.vad_pos = 0  # absolute sample index of next frame to run VAD on
        self.blocks_idx: list[int] = []  # block end sample index
        self.blocks_wall: list[float] = []  # block end wall time
        self.in_seg = False
        self.seg_start = 0
        self.seg_cut_start: Optional[float] = None
        self.last_speech_end = 0
        self.silence = 0
        self.min_next_start = 0
        self.frames: list[tuple[int, float, float]] = []  # (frame start idx, prob, energy) inside open segment
        self.seg_speech_frames = 0
        self.seg_total_frames = 0

    # ---------------------------------------------------------------- helpers
    def _wall(self, idx: int) -> float:
        if not self.blocks_idx:
            return 0.0
        i = bisect.bisect_left(self.blocks_idx, idx)
        if i >= len(self.blocks_idx):
            i = len(self.blocks_idx) - 1
        return self.blocks_wall[i] - (self.blocks_idx[i] - idx) / SR

    def _slice(self, a: int, b: int) -> np.ndarray:
        a = max(a, self.buf_start)
        return self.buf[a - self.buf_start:b - self.buf_start].copy()

    def _threshold(self, seg_len: float) -> float:
        """Silence needed to cut, as a function of current segment length (target 3–12 s)."""
        if seg_len < 3.0:
            return 0.8
        if seg_len < 8.0:
            return 0.45
        if seg_len < 12.0:
            return 0.25
        return 0.15

    def _make(self, a: int, b: int, kind: str, cut_end: Optional[float]) -> Segment:
        speech_end = min(self.last_speech_end, b)
        return Segment(
            seq=next(self._seq), audio=self._slice(a, b), abs_start=a / SR, abs_end=b / SR,
            wall_start=self._wall(a), wall_end=self._wall(b), speech_end_wall=self._wall(speech_end),
            kind=kind, cut_start=self.seg_cut_start, cut_end=cut_end,
            speech_frames=self.seg_speech_frames, total_frames=self.seg_total_frames,
        )

    def _trim(self) -> None:
        keep_from = (self.seg_start if self.in_seg else self.vad_pos) - int((self.PRE_PAD + 1.0) * SR)
        keep_from = max(self.buf_start, min(keep_from, self.vad_pos - int(self.PRE_PAD * SR) - FRAME))
        if keep_from - self.buf_start > SR * 5:
            self.buf = self.buf[keep_from - self.buf_start:]
            self.buf_start = keep_from
        if len(self.blocks_idx) > 2000:
            cut = bisect.bisect_left(self.blocks_idx, self.buf_start) - 1
            if cut > 0:
                del self.blocks_idx[:cut]
                del self.blocks_wall[:cut]

    # ---------------------------------------------------------------- public
    def feed(self, x: np.ndarray, wall_end: float) -> None:
        with self.lock:
            self.buf = np.concatenate([self.buf, x.astype(np.float32, copy=False)])
            self.total += len(x)
            self.blocks_idx.append(self.total)
            self.blocks_wall.append(wall_end)
            n = (self.total - self.vad_pos) // FRAME
            if n <= 0:
                return
            a = self.vad_pos - self.buf_start
            frames = self.buf[a:a + n * FRAME].reshape(n, FRAME)
            probs = self.vad(frames)
            energies = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
            for k in range(n):
                self._step(self.vad_pos + k * FRAME, float(probs[k]), float(energies[k]))
            self.vad_pos += n * FRAME
            self._trim()

    def _step(self, fstart: int, p: float, energy: float) -> None:
        fend = fstart + FRAME
        if not self.in_seg:
            if p >= self.START_TH:
                self.in_seg = True
                self.seg_start = max(self.min_next_start, self.buf_start, fstart - int(self.PRE_PAD * SR))
                self.seg_cut_start = None
                self.last_speech_end = fend
                self.silence = 0
                self.frames = []
                self.seg_speech_frames = 0
                self.seg_total_frames = 0
            else:
                return
        self.frames.append((fstart, p, energy))
        self.seg_total_frames += 1
        if p >= self.KEEP_TH:
            self.silence = 0
            self.last_speech_end = fend
            self.seg_speech_frames += 1
        else:
            self.silence += FRAME
        seg_len = (fend - self.seg_start) / SR
        sil = self.silence / SR
        if self.silence and (sil >= self._threshold(seg_len) or sil >= self.END_SILENCE):
            end = self.last_speech_end + min(int(self.POST_PAD * SR), self.silence)
            self.emit(self._make(self.seg_start, end, "pause", None))
            self.in_seg = False
            self.min_next_start = end
            return
        if seg_len >= self.MAX_LEN:
            self._force_cut(fend)

    def _force_cut(self, now: int) -> None:
        window_from = now - int(self.FORCE_WINDOW * SR)
        cands = [f for f in self.frames if f[0] >= window_from] or self.frames[-1:]
        best = min(cands, key=lambda f: (f[2], f[1]))
        cut = best[0] + FRAME // 2
        ov = int(self.OVERLAP * SR)
        end = min(cut + ov, now)
        seg = self._make(self.seg_start, end, "forced", cut / SR)
        self.emit(seg)
        # continue the same speech run in a new segment that overlaps the cut
        self.seg_start = max(self.buf_start, cut - ov)
        self.seg_cut_start = cut / SR
        self.frames = [f for f in self.frames if f[0] >= self.seg_start]
        self.seg_speech_frames = sum(1 for f in self.frames if f[1] >= self.KEEP_TH)
        self.seg_total_frames = len(self.frames)

    def flush(self) -> None:
        """End of stream: emit whatever is open."""
        with self.lock:
            if self.in_seg:
                end = min(self.total, self.last_speech_end + int(self.POST_PAD * SR))
                end = max(end, self.seg_start + FRAME)
                self.emit(self._make(self.seg_start, end, "eof", None))
                self.in_seg = False
                self.min_next_start = end

    def open_segment_audio(self) -> tuple[Optional[int], Optional[np.ndarray]]:
        """For live partials: (seq that the open segment will get, its audio so far)."""
        with self.lock:
            if not self.in_seg:
                return None, None
            nxt = next(self._seq)
            self._seq = itertools.count(nxt)  # peek without consuming
            return nxt, self._slice(self.seg_start, self.vad_pos)
