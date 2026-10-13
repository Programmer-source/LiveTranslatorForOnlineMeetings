"""Speech-to-text: faster-whisper (local, GPU/CPU) or Deepgram (cloud, optional).

Includes hallucination guards and forced-cut seam de-duplication."""
from __future__ import annotations

import io
import logging
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .seam import Word, join_words, merge_seam, split_at_cut
from .segmenter import Segment

log = logging.getLogger("stt")

HALLUCINATIONS = [
    "thank you for watching", "thanks for watching", "please subscribe", "subscribe to my channel",
    "like and subscribe", "subtitles by", "subtitled by", "transcribed by", "amara.org",
    "thank you so much for watching", "see you in the next video", "i'll see you next time",
    "captions by", "translated by", "www.", ".com",
]


@dataclass
class Transcript:
    seq: int
    text: str
    words: list[Word]
    seg: Segment
    stt_ms: float = 0.0
    dropped: list[str] = field(default_factory=list)  # for diagnostics

    @property
    def wall_start(self) -> float:
        if self.words:
            return self.seg.wall_start + (self.words[0].start - self.seg.abs_start)
        return self.seg.wall_start

    @property
    def wall_end(self) -> float:
        if self.words:
            return self.seg.wall_start + (self.words[-1].end - self.seg.abs_start)
        return self.seg.wall_end


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()


def is_hallucination(text: str, avg_logprob: float, no_speech_prob: float, speech_ratio: float) -> bool:
    """Conservative: only drops text that is both suspicious in content AND weakly supported."""
    n = _norm(text)
    if not n:
        return True
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    if any(h.replace(".", "") in n for h in HALLUCINATIONS if len(h) > 4):
        if avg_logprob < -0.6 or no_speech_prob > 0.3 or speech_ratio < 0.3:
            return True
    return False


def collapse_loops(words: list[Word], min_repeats: int = 5) -> list[Word]:
    """Whisper decoding loops ("the the the the the …"): collapse runs of an n-gram repeated
    ≥ min_repeats times down to 2 copies. Real speech virtually never repeats 5+ times."""
    norms = [w.norm for w in words]
    i = 0
    out: list[Word] = []
    while i < len(words):
        collapsed = False
        for n in range(1, 6):
            gram = norms[i:i + n]
            if len(gram) < n or not all(gram):
                break
            reps = 1
            while norms[i + reps * n:i + (reps + 1) * n] == gram:
                reps += 1
            if reps >= min_repeats:
                out.extend(words[i:i + 2 * n])
                i += reps * n
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return out


class SeamState:
    """Carries A's words beyond a forced cut until B is transcribed."""

    def __init__(self):
        self.kept: list[Word] = []
        self.carry: list[Word] = []
        self.cut: Optional[float] = None

    def apply(self, seg: Segment, words: list[Word]) -> list[Word]:
        if seg.cut_start is not None and self.cut is not None and abs(seg.cut_start - self.cut) < 1e-6:
            words = merge_seam(self.kept, self.carry, words, self.cut)
        elif self.carry:
            # unexpected: previous carry has no continuation; emit it rather than lose it
            words = sorted(self.carry + words, key=lambda w: w.start)
        self.carry, self.cut = [], None
        if seg.cut_end is not None:
            kept, carry = split_at_cut(words, seg.cut_end)
            self.kept, self.carry, self.cut = kept, carry, seg.cut_end
            return kept
        self.kept = words
        return words

    def flush(self) -> list[Word]:
        c, self.carry, self.cut = self.carry, [], None
        return c


# ----------------------------------------------------------------------------- engines

class WhisperEngine:
    def __init__(self, model_name: str, device: str, compute_type: str, download_root: str):
        from faster_whisper import WhisperModel

        self.name = model_name
        self.device = device
        self.compute_type = compute_type
        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type,
                                      download_root=download_root)
        except Exception:
            if device != "cuda":
                raise
            log.exception("CUDA model load failed; falling back to CPU")
            self.device, self.compute_type = "cpu", "int8"
            self.name = model_name if not model_name.startswith("large") else "small.en"
            self.model = WhisperModel(self.name, device="cpu", compute_type="int8", download_root=download_root)

    def transcribe(self, audio: np.ndarray, prompt: str, hotwords: str, speech_ratio: float) -> tuple[list[Word], list[str]]:
        segments, _info = self.model.transcribe(
            audio, language="en", task="transcribe", beam_size=5, best_of=5,
            word_timestamps=True, condition_on_previous_text=True,
            initial_prompt=prompt or None, hotwords=hotwords or None,
            vad_filter=False, no_speech_threshold=0.6, log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        words: list[Word] = []
        dropped: list[str] = []
        for s in segments:
            if is_hallucination(s.text, s.avg_logprob, s.no_speech_prob, speech_ratio):
                dropped.append(s.text.strip())
                continue
            for w in (s.words or []):
                words.append(Word(w.word, float(w.start), float(w.end), float(w.probability)))
            if not s.words and s.text.strip():  # no word timings (should not happen) — keep text anyway
                words.append(Word(" " + s.text.strip(), float(s.start), float(s.end)))
        return words, dropped

    def partial(self, audio: np.ndarray, prompt: str) -> str:
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, best_of=1, temperature=0.0, word_timestamps=False,
            condition_on_previous_text=False, initial_prompt=prompt or None, vad_filter=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments
                        if not is_hallucination(s.text, s.avg_logprob, s.no_speech_prob, 1.0)).strip()


def _wav_bytes(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return bio.getvalue()


class DeepgramEngine:
    name = "deepgram nova-3"
    device = "cloud"
    compute_type = ""

    def __init__(self, api_key: str):
        import httpx

        self.key = api_key
        self.client = httpx.Client(timeout=30)

    def transcribe(self, audio: np.ndarray, prompt: str, hotwords: str, speech_ratio: float) -> tuple[list[Word], list[str]]:
        params = [("model", "nova-3"), ("language", "en"), ("smart_format", "true"), ("punctuate", "true")]
        for t in [h.strip() for h in hotwords.split(",") if h.strip()][:50]:
            params.append(("keyterm", t))
        r = self.client.post("https://api.deepgram.com/v1/listen", params=params, content=_wav_bytes(audio),
                             headers={"Authorization": "Token " + self.key, "Content-Type": "audio/wav"})
        r.raise_for_status()
        alts = r.json()["results"]["channels"][0]["alternatives"]
        words = [Word(" " + (w.get("punctuated_word") or w["word"]), float(w["start"]), float(w["end"]),
                      float(w.get("confidence", 1.0))) for w in (alts[0].get("words", []) if alts else [])]
        return words, []

    def partial(self, audio, prompt):  # not supported (would double the cost)
        return ""


# ----------------------------------------------------------------------------- stage

class STTStage:
    """Converts segments (in seq order) into transcripts, keeping order and seams."""

    def __init__(self, engine, glossary_terms_fn):
        self.engine = engine
        self.glossary_terms_fn = glossary_terms_fn
        self.history = ""  # transcript so far, for the prompt
        self.seam = SeamState()
        self.last_text = ""

    def _prompt(self) -> tuple[str, str]:
        terms = [t for t in self.glossary_terms_fn() if t]
        hot = ", ".join(terms)
        tail = self.history[-200:]
        if tail and " " in tail:
            tail = tail[tail.index(" ") + 1:]
        prompt = (f"Glossary: {hot}. " if hot else "") + tail
        return prompt.strip(), hot

    def process(self, seg: Segment) -> Transcript:
        t0 = time.perf_counter()
        speech_ratio = seg.speech_frames / max(1, seg.total_frames)
        prompt, hot = self._prompt()
        words: list[Word] = []
        dropped: list[str] = []
        if seg.speech_frames > 0 and seg.audio.size > 0:
            raw, dropped = self.engine.transcribe(seg.audio, prompt, hot, speech_ratio)
            words = [Word(w.text, w.start + seg.abs_start, w.end + seg.abs_start, w.prob) for w in raw]
            words = collapse_loops(words)
        words = self.seam.apply(seg, words)
        if seg.kind == "eof":
            words += self.seam.flush()
        text = join_words(words)
        # Identical repeat of the previous segment with weak evidence = classic Whisper echo.
        if text and _norm(text) == _norm(self.last_text) and words and np.mean([w.prob for w in words]) < 0.4:
            dropped.append(text)
            text, words = "", []
        if text:
            self.last_text = text
            self.history = (self.history + " " + text)[-1000:]
        return Transcript(seg.seq, text, words, seg, (time.perf_counter() - t0) * 1000, dropped)
