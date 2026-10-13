"""Pipeline orchestration:

audio source ─► segmenter (VAD) ─► STT ─► unit builder ─► translator ─► hub (web) + Zoom + files

Every stage is connected by an asyncio queue and processes items strictly in order."""
from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

from . import paths
from .audio import SR, AudioError, FileSource, LevelMeter, WasapiSource
from .config import Config, Glossary
from .cuda_setup import choose_whisper_model, detect_hardware
from .hub import Hub
from .segmenter import Segment, Segmenter
from .session import SessionWriter
from .stt import DeepgramEngine, STTStage, Transcript, WhisperEngine
from .translate import Ctx, DeepLProvider, FreeGoogleProvider, GeminiProvider, TranslatorManager
from .zoom import ZoomCaptions

log = logging.getLogger("pipeline")

EOF = object()
SENT_END = re.compile(r"[.!?…][\"'”»)\]]*$")


class Pipeline:
    HOLD_IDLE = 2.0  # close a mid-sentence unit after this long if nobody is speaking
    HOLD_MAX = 8.0  # …and in any case after this long
    UNIT_MAX_WORDS = 45
    UNIT_MAX_SEC = 20.0
    BATCH_MAX = 3
    BATCH_MAX_WORDS = 70
    CONTEXT_PAIRS = 4

    def __init__(self, config: Config, glossary: Glossary, hub: Hub, *, engine=None, translator=None,
                 write_files: bool = True):
        self.config = config
        self.glossary = glossary
        self.hub = hub
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.http = httpx.AsyncClient(timeout=45, follow_redirects=True)
        self.zoom = ZoomCaptions(config)
        self.meter = LevelMeter()
        self.write_files = write_files
        self.session: Optional[SessionWriter] = None

        self.hw: dict = {}
        self.engine = engine
        self.engine_state = "ready" if engine else "loading"  # loading | ready | error
        self.engine_msg = ""
        self.engine_ready = asyncio.Event()
        self.stt: Optional[STTStage] = STTStage(engine, glossary.terms) if engine else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self.stt_busy = False
        self.stt_errors = 0

        self._translator_override = translator
        self.translator = translator or self._make_translator()

        self.seg_q: asyncio.Queue = asyncio.Queue()
        self.tr_q: asyncio.Queue = asyncio.Queue()
        self.line_q: asyncio.Queue = asyncio.Queue()

        self.source = None
        self.segmenter: Optional[Segmenter] = None
        self.seg_thread: Optional[threading.Thread] = None
        self.audio_q: "queue.Queue" = queue.Queue()
        self.running = False
        self.mode = ""  # live | test
        self.next_seq = 0
        self.last_transcribed_seq = -1
        self.pending: list[Transcript] = []
        self.pending_since = 0.0
        self.untranslated = 0
        self.latencies: deque = deque(maxlen=10)
        self.all_latencies: list[float] = []
        self.ctx_pairs: deque = deque(maxlen=self.CONTEXT_PAIRS)
        self.tasks: list[asyncio.Task] = []
        self.audio_error = ""
        self.preview: deque = deque(maxlen=6)

    # ------------------------------------------------------------------ setup
    def _make_translator(self) -> TranslatorManager:
        providers = [GeminiProvider(self.config.get("gemini_api_key", ""), self.config.get("gemini_models", []),
                                    int(self.config.get("gemini_rpm", 10) or 10), self.http)]
        if self.config.get("deepl_api_key"):
            providers.append(DeepLProvider(self.config["deepl_api_key"], self.http, Glossary.keep_english))
        if self.config.get("free_fallback_translator", True):
            providers.append(FreeGoogleProvider(self.http))
        return TranslatorManager(providers)

    def reload_translator(self) -> None:
        if not self._translator_override:
            self.translator = self._make_translator()

    async def start_background(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.engine_ready = asyncio.Event()
        if self.engine:
            self.engine_ready.set()
        else:
            threading.Thread(target=self._load_engine, daemon=True, name="model-load").start()
        self.zoom.start()
        for coro in (self._stt_task(), self._unit_task(), self._translate_task(), self._partial_task()):
            self.tasks.append(asyncio.create_task(coro))

    def _load_engine(self) -> None:
        try:
            self.engine_state = "loading"
            self.hw = detect_hardware()
            if self.config.get("stt_engine") == "deepgram" and self.config.get("deepgram_api_key"):
                engine = DeepgramEngine(self.config["deepgram_api_key"])
            else:
                name, device, ctype = choose_whisper_model(self.config.get("whisper_model", "auto"), self.hw)
                self.engine_msg = f"Завантаження моделі {name} ({'GPU' if device == 'cuda' else 'CPU'})…"
                engine = WhisperEngine(name, device, ctype, str(paths.MODELS_DIR / "whisper"))
                # warm-up so the first real segment is not slow
                import numpy as np

                engine.transcribe(np.zeros(SR, np.float32), "", "", 0.0)
            self.engine = engine
            self.stt = STTStage(engine, self.glossary.terms)
            self.engine_state = "ready"
            self.engine_msg = ""
            log.info("STT ready: %s on %s", engine.name, engine.device)
        except Exception as e:
            log.exception("model load failed")
            self.engine_state = "error"
            self.engine_msg = f"Не вдалося завантажити модель розпізнавання: {e}"
        finally:
            if self.engine and self.loop:
                self.loop.call_soon_threadsafe(self.engine_ready.set)

    # ------------------------------------------------------------------ control
    def start(self, device: Optional[str] = None, file: Optional[str] = None, speed: float = 1.0) -> None:
        if self.running:
            self.stop()
        if self.write_files and self.session is None:
            self.session = SessionWriter(self.hub.started_at)
        self.audio_q = queue.Queue()
        if file:
            self.source = FileSource(self.audio_q, self.meter, file, speed=speed)
            self.mode = "test"
        else:
            self.source = WasapiSource(self.audio_q, self.meter, device or self.config.get("audio_device", "default"))
            self.mode = "live"
        from .vad import SileroVAD

        self.segmenter = Segmenter(SileroVAD(), self._emit_segment, seq_start=self.next_seq)
        self.running = True
        self.audio_error = ""
        self.source.start()
        self.seg_thread = threading.Thread(target=self._segment_loop, args=(self.source, self.segmenter),
                                           daemon=True, name="segmenter")
        self.seg_thread.start()
        self.hub.set_state("live")

    def stop(self) -> None:
        """Stop capturing. Everything already captured is still transcribed and translated."""
        src = self.source
        self.running = False
        if src:
            src.stop()
        if self.seg_thread:
            self.seg_thread.join(timeout=5)
        self.source = None
        self.hub.set_state("stopped" if self.hub.lines else "idle")

    def new_session(self) -> None:
        self.stop()
        if self.session:
            self.session.close()
            self.session = None
        self.hub.new_session()
        self.ctx_pairs.clear()
        self.preview.clear()
        self.hub.set_state("idle")
        for ws, q in list(self.hub.clients.items()):
            q.put_nowait(None)  # force clients to resync to the new session

    def _segment_loop(self, source, seg: Segmenter) -> None:
        while True:
            blk = self.audio_q.get()
            if blk.eof:
                seg.flush()
                break
            seg.feed(blk.samples, blk.wall_end)
        if source.error:
            self.audio_error = source.error
        self._put(self.seg_q, EOF)
        if self.loop:
            self.loop.call_soon_threadsafe(self._source_finished, source)

    def _source_finished(self, source) -> None:
        if self.source is source:  # ended by itself (end of test file)
            self.running = False
            self.source = None
            self.hub.set_state("stopped" if self.hub.lines else "idle")

    def _put(self, q: asyncio.Queue, item) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(q.put_nowait, item)

    def _emit_segment(self, s: Segment) -> None:
        self.next_seq = s.seq + 1
        self._put(self.seg_q, s)

    # ------------------------------------------------------------------ stages
    async def _stt_task(self) -> None:
        await self.engine_ready.wait()
        while True:
            item = await self.seg_q.get()
            if item is EOF:
                await self.tr_q.put(EOF)
                continue
            seg: Segment = item
            backoff = 1.0
            while True:  # never drop a segment because of an STT error
                try:
                    self.stt_busy = True
                    tr = await self.loop.run_in_executor(self.executor, self.stt.process, seg)
                    self.stt_errors = 0
                    break
                except Exception as e:
                    self.stt_errors += 1
                    self.engine_msg = f"Помилка розпізнавання (повтор): {e}"
                    log.exception("STT failed for seq %s", seg.seq)
                    await asyncio.sleep(backoff)
                    backoff = min(10, backoff * 2)
                finally:
                    self.stt_busy = False
            self.engine_msg = "" if self.engine_state == "ready" else self.engine_msg
            self.last_transcribed_seq = seg.seq
            if tr.dropped:
                log.info("seq %d: dropped as hallucination: %s", seg.seq, tr.dropped)
            await self.tr_q.put(tr)

    def _held_text(self) -> str:
        return " ".join(t.text for t in self.pending)

    def _should_close(self) -> bool:
        if not self.pending:
            return False
        text = self._held_text()
        words = len(text.split())
        dur = self.pending[-1].wall_end - self.pending[0].wall_start
        if SENT_END.search(text) and words >= 3:
            return True
        return words >= self.UNIT_MAX_WORDS or dur >= self.UNIT_MAX_SEC

    def _speech_in_progress(self) -> bool:
        s = self.segmenter
        return bool(self.running and s and s.in_seg) or not self.seg_q.empty() or self.stt_busy

    async def _unit_task(self) -> None:
        while True:
            timeout = None
            if self.pending:
                waited = time.time() - self.pending_since
                limit = self.HOLD_MAX if self._speech_in_progress() else self.HOLD_IDLE
                timeout = max(0.05, min(limit - waited, 0.5))
            try:
                item = await asyncio.wait_for(self.tr_q.get(), timeout) if timeout else await self.tr_q.get()
            except asyncio.TimeoutError:
                waited = time.time() - self.pending_since
                if waited >= self.HOLD_MAX or (waited >= self.HOLD_IDLE and not self._speech_in_progress()):
                    self._close_unit()
                continue
            if item is EOF:
                self._close_unit()
                continue
            if item.text:
                self.pending.append(item)
                self.pending_since = time.time()
                if self._should_close():
                    self._close_unit()
                else:
                    self.hub.set_partial(self._held_text())

    def _close_unit(self) -> None:
        if not self.pending:
            return
        units, self.pending = self.pending, []
        en = " ".join(t.text for t in units)
        t0 = units[0].wall_start - self.hub.started_at
        t1 = units[-1].wall_end - self.hub.started_at
        line = self.hub.add_line(en, t0, t1, [t.seq for t in units])
        line_meta = {"id": line["id"], "speech_end": units[-1].seg.speech_end_wall, "created": time.time()}
        if self.session:
            self.session.write_en(line)
        self.untranslated += 1
        self.line_q.put_nowait(line_meta)

    async def _translate_task(self) -> None:
        while True:
            batch = [await self.line_q.get()]
            # backlog → translate a few short lines in one request (saves free-tier quota)
            while not self.line_q.empty() and len(batch) < self.BATCH_MAX:
                nxt = self.line_q._queue[0]  # peek
                words = sum(len(self.hub.lines[b["id"]]["en"].split()) for b in batch)
                if words + len(self.hub.lines[nxt["id"]]["en"].split()) > self.BATCH_MAX_WORDS:
                    break
                batch.append(self.line_q.get_nowait())
            items = [self.hub.lines[b["id"]]["en"] for b in batch]
            ctx = Ctx(list(self.ctx_pairs), dict(self.glossary.items),
                      [t for t, v in self.glossary.items.items() if Glossary.keep_english(t, v)],
                      self.glossary.prompt_lines())

            def on_retry(err, attempt, batch=batch):
                for b in batch:
                    self.hub.mark_retrying(b["id"], err)

            results = await self.translator.translate(items, ctx, on_retry=on_retry)
            now = time.time()
            for b, en, uk in zip(batch, items, results):
                lat = now - b["speech_end"]
                line = self.hub.set_translation(b["id"], uk, self.translator.last_provider, lat)
                self.ctx_pairs.append((en, uk))
                self.untranslated -= 1
                self.latencies.append(lat)
                self.all_latencies.append(lat)
                self.preview.append({"id": line["id"], "en": en, "uk": uk})
                if self.session:
                    self.session.write_translated(line)
                self.zoom.push_line(uk)

    async def _partial_task(self) -> None:
        last_len, last_seq = 0, -1
        while True:
            await asyncio.sleep(0.6)
            try:
                eng = self.engine
                if (not self.running or not eng or not self.config.get("live_partials", True)
                        or getattr(eng, "device", "") != "cuda" or self.stt_busy or not self.seg_q.empty()
                        or not self.segmenter):
                    continue
                seq, audio = self.segmenter.open_segment_audio()
                if audio is None or len(audio) < SR:
                    continue
                if seq == last_seq and len(audio) - last_len < SR // 2:
                    continue
                last_seq, last_len = seq, len(audio)
                n_lines = len(self.hub.lines)
                prompt = self.stt.history[-200:] if self.stt else ""
                text = await self.loop.run_in_executor(self.executor, eng.partial, audio, prompt)
                # discard if the final transcript overtook us
                if len(self.hub.lines) != n_lines or self.last_transcribed_seq >= seq:
                    continue
                held = self._held_text()
                self.hub.set_partial((held + " " + text).strip() if text else held)
            except Exception:
                log.debug("partial failed", exc_info=True)

    # ------------------------------------------------------------------ status
    def backlog(self) -> int:
        return self.seg_q.qsize() + self.tr_q.qsize() + self.untranslated + (1 if self.stt_busy else 0)

    def delay(self) -> float:
        lat = list(self.latencies)[-5:]
        cur = sum(lat) / len(lat) if lat else 0.0
        # if the oldest untranslated line is already older than that, report the real lag
        if self.untranslated and self.line_q.qsize():
            try:
                oldest = self.line_q._queue[0]
                cur = max(cur, time.time() - oldest["speech_end"])
            except Exception:
                pass
        return round(cur, 1)
