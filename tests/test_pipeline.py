import asyncio
import json
import random
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Config, Glossary
from app.hub import Hub
from app.pipeline import Pipeline
from app.seam import Word
from app.segmenter import Segment
from app.server import AppState, create_app
from app.translate import ProviderError, TranslatorManager


class FakeEngine:
    """Segment audio is constant = seq/1000; returns two words per segment."""
    name = "fake"
    device = "cpu"

    def transcribe(self, audio, prompt, hotwords, speech_ratio):
        seq = int(round(float(audio[0]) * 1000))
        end = "." if seq % 3 == 2 else ""  # some segments end mid-sentence -> merged units
        return [Word(f" alpha{seq}", 0.1, 0.5), Word(f" beta{seq}{end}", 0.6, 0.9)], []

    def partial(self, audio, prompt):
        return ""


class ChaosProvider:
    """Fails ~50% of calls and is sometimes rate limited; uppercases as 'translation'."""
    label = "chaos"

    def __init__(self, seed=1):
        self.rng = random.Random(seed)
        self.calls = 0

    def available_in(self):
        return 0.0

    async def translate(self, items, ctx):
        self.calls += 1
        await asyncio.sleep(self.rng.random() * 0.01)
        if self.rng.random() < 0.5:
            raise ProviderError("simulated outage", self.rng.choice(["rate", "other"]))
        return ["УКР " + t for t in items]


def make_cfg(tmp_path):
    cfg = Config(tmp_path / "config.json")
    cfg.data["free_fallback_translator"] = False
    return cfg


def test_pipeline_order_and_no_loss_under_translator_failures(tmp_path):
    async def run():
        hub = Hub()
        tm = TranslatorManager([ChaosProvider()], max_backoff=0.01)
        p = Pipeline(make_cfg(tmp_path), Glossary(tmp_path / "g.json"), hub, engine=FakeEngine(),
                     translator=tm, write_files=False)
        p.HOLD_IDLE = 0.05
        await p.start_background()
        n = 40
        now = time.time()
        for seq in range(n):
            seg = Segment(seq, np.full(1600, seq / 1000, np.float32), seq, seq + 1, now + seq, now + seq + 1,
                          now + seq + 0.9, speech_frames=10, total_frames=10)
            p._emit_segment(seg)
        from app.pipeline import EOF
        p._put(p.seg_q, EOF)
        for _ in range(400):
            await asyncio.sleep(0.05)
            if hub.lines and all(l["uk"] for l in hub.lines) and p.backlog() == 0 and not p.pending:
                break
        assert all(l["uk"] for l in hub.lines), "some lines never got translated"
        en_words = " ".join(l["en"] for l in hub.lines).split()
        expected = []
        for seq in range(n):
            expected += [f"alpha{seq}", f"beta{seq}" + ("." if seq % 3 == 2 else "")]
        assert en_words == expected  # every word, in order, exactly once
        uk_words = " ".join(l["uk"] for l in hub.lines).replace("УКР ", "").split()
        assert uk_words == expected
        seqs = [s for l in hub.lines for s in l["seqs"]]
        assert seqs == list(range(n))
        assert [l["id"] for l in hub.lines] == list(range(len(hub.lines)))
        # update events were emitted in line order
        upd = [e["line"]["id"] for e in hub.events if e["kind"] == "update" and e["line"]["status"] == "done"]
        assert upd == sorted(upd)
        for t in p.tasks:
            t.cancel()

    asyncio.run(run())


def test_websocket_reconnect_replay(tmp_path):
    cfg = make_cfg(tmp_path)
    state = AppState(cfg, Glossary(tmp_path / "g.json"), enable_tunnel=False,
                     pipeline_kwargs={"engine": FakeEngine(), "write_files": False})
    app = create_app(state)
    hub = state.hub
    with TestClient(app) as client:
        for i in range(3):
            ln = hub.add_line(f"line {i}", i, i + 1, [i])
            hub.set_translation(ln["id"], f"рядок {i}")
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "hello", "session": None, "last_ev": -1}))
            msgs = [json.loads(ws.receive_text()) for _ in range(3)]
            snap = [m for m in msgs if m["type"] == "snapshot"][0]
            assert [l["uk"] for l in snap["lines"]] == ["рядок 0", "рядок 1", "рядок 2"]
            last_ev, session = snap["ev"], snap["session"]
        # while disconnected, more lines arrive
        ln = hub.add_line("line 3", 3, 4, [3])
        hub.set_translation(ln["id"], "рядок 3")
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "hello", "session": session, "last_ev": last_ev}))
            msgs = [json.loads(ws.receive_text()) for _ in range(3)]
            rep = [m for m in msgs if m["type"] == "replay"][0]
            assert [e["kind"] for e in rep["events"]] == ["line", "update"]
            assert rep["events"][-1]["line"]["uk"] == "рядок 3"
            # live event after replay
            ln = hub.add_line("line 4", 4, 5, [4])
            got = json.loads(ws.receive_text())
            while got["type"] != "ev":
                got = json.loads(ws.receive_text())
            assert got["ev"] == rep["ev"] + 1 and got["line"]["en"] == "line 4"


def test_control_panel_is_local_only(tmp_path):
    state = AppState(make_cfg(tmp_path), Glossary(tmp_path / "g.json"), enable_tunnel=False,
                     pipeline_kwargs={"engine": FakeEngine(), "write_files": False})
    with TestClient(create_app(state)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/state").status_code == 200
        # through the Cloudflare tunnel
        assert client.get("/api/state", headers={"cf-connecting-ip": "1.2.3.4"}).status_code == 404
        assert client.get("/control", headers={"host": "abc.trycloudflare.com"}).status_code == 404
        # CSRF: write calls need the custom header
        assert client.post("/api/config", json={"gemini_api_key": "x"}).status_code == 403
        r = client.post("/api/config", json={"gemini_api_key": "secret-key-123456"}, headers={"x-control": "1"})
        assert r.status_code == 200
        assert "secret-key-123456" not in r.text
        assert "secret-key-123456" not in client.get("/").text
