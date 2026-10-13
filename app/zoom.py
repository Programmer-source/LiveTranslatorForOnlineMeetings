"""Zoom closed-caption API (3rd-party captioning token).

POST text/plain to <URL>&seq=<n>&lang=uk-UA. seq starts at 0 (or the seq already in the URL)
and increases by 1 per request. Runs independently: failures never block the web output."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time

import httpx

log = logging.getLogger("zoom")

MAX_CHARS = 120


def split_caption(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split into readable chunks: sentence ends, then commas/dashes, then spaces. Never loses text."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    out: list[str] = []
    for s in sentences:
        pieces = [s] if len(s) <= max_chars else re.split(r"(?<=[,;:—–])\s+", s)
        for p in pieces:
            while len(p) > max_chars:
                cut = p.rfind(" ", 0, max_chars + 1)
                if cut <= max_chars // 3:
                    cut = max_chars
                out.append(p[:cut].strip())
                p = p[cut:].strip()
            if p:
                out.append(p)
    # re-pack small neighbours so captions are not choppy
    packed: list[str] = []
    for p in out:
        if packed and len(packed[-1]) + 1 + len(p) <= max_chars:
            packed[-1] += " " + p
        else:
            packed.append(p)
    return packed


def parse_url(url: str) -> tuple[str, int]:
    """Return (base URL without seq/lang, starting seq). The rest of the query is kept byte-for-byte
    (the token's signature must not be re-encoded)."""
    url = (url or "").strip()
    base, _, frag = url.partition("#")
    path, sep, query = base.partition("?")
    seq = 0
    kept = []
    for item in query.split("&") if query else []:
        k = item.split("=", 1)[0]
        if k == "seq":
            try:
                seq = int(item.split("=", 1)[1])
            except (IndexError, ValueError):
                seq = 0
        elif k != "lang" and item:
            kept.append(item)
    out = path + ("?" + "&".join(kept) if kept else "")
    return out, seq


def url_key(url: str) -> str:
    return hashlib.sha1(parse_url(url)[0].encode()).hexdigest()[:16]


class ZoomCaptions:
    def __init__(self, config, client: httpx.AsyncClient | None = None):
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=15)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.sent = 0
        self.failed = 0
        self.last_error = ""
        self.last_ok = 0.0
        self.state = "off"  # off | ok | error | retrying
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def url(self) -> str:
        return (self.config.get("zoom_caption_url") or "").strip()

    def enabled(self) -> bool:
        return bool(self.url) and bool(self.config.get("zoom_captions_enabled", True))

    def start(self) -> None:
        if not self._task:
            self._task = asyncio.create_task(self._worker())

    def push_line(self, uk: str) -> None:
        if not self.enabled():
            return
        for chunk in split_caption(uk):
            self.queue.put_nowait(chunk)

    def _next_seq(self) -> tuple[str, int]:
        base, url_seq = parse_url(self.url)
        key = url_key(self.url)
        state = dict(self.config.get("zoom_seq_state") or {})
        seq = max(url_seq, int(state.get(key, url_seq)))
        return base, seq

    def _commit_seq(self, seq: int) -> None:
        key = url_key(self.url)
        state = dict(self.config.get("zoom_seq_state") or {})
        state = {k: v for k, v in state.items() if k == key or len(state) < 20}
        state[key] = seq + 1
        self.config.update({"zoom_seq_state": state})

    async def post(self, text: str) -> None:
        async with self._lock:
            base, seq = self._next_seq()
            sep = "&" if "?" in base else "?"
            full = f"{base}{sep}seq={seq}&lang=uk-UA"
            r = await self.client.post(full, content=text.encode("utf-8"),
                                       headers={"Content-Type": "text/plain; charset=utf-8"})
            if r.status_code >= 400:
                body = r.text[:120].strip()
                if r.status_code in (400, 403) and "seq" in body.lower():
                    self._commit_seq(seq)  # seq rejected (already used) — skip ahead next time
                raise RuntimeError(f"HTTP {r.status_code} {body}")
            self._commit_seq(seq)

    async def test(self) -> tuple[bool, str]:
        if not self.url:
            return False, "Вставте URL (API-токен) субтитрів Zoom"
        if not self.url.startswith("http"):
            return False, "Це не схоже на посилання: має починатися з https://"
        try:
            await self.post("✅ Тест: українські субтитри працюють")
            self.state = "ok"
            self.last_error = ""
            return True, "Субтитри Zoom працюють — перевірте рядок субтитрів у Zoom"
        except Exception as e:
            self.last_error = str(e)
            self.state = "error"
            return False, f"Не вдалося: {e}"

    async def _worker(self) -> None:
        prev_len = 0
        while True:
            chunk = await self.queue.get()
            # pacing: give viewers time to read the previous chunk, unless we are behind
            if prev_len and self.queue.qsize() < 2:
                await asyncio.sleep(min(3.0, max(0.8, prev_len / 20)))
            backoff = 1.0
            for attempt in range(8):
                if not self.enabled():
                    break
                try:
                    await self.post(chunk)
                    self.sent += 1
                    self.last_ok = time.time()
                    self.state = "ok"
                    self.last_error = ""
                    break
                except Exception as e:
                    self.last_error = str(e)[:200]
                    self.state = "retrying"
                    log.warning("zoom caption failed: %s", e)
                    await asyncio.sleep(backoff)
                    backoff = min(10, backoff * 2)
            else:
                # Captions are time-sensitive: after ~1 min of failures skip this chunk
                # (the web page keeps everything).
                self.failed += 1
                self.state = "error"
            prev_len = len(chunk)
