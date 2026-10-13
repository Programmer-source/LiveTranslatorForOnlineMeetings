"""Broadcast hub: ordered history of subtitle lines + event log for reconnect replay."""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from typing import Any, Optional

log = logging.getLogger("hub")


class Hub:
    MAX_CLIENT_BACKLOG = 2000

    def __init__(self):
        self.new_session()
        self.clients: dict[Any, asyncio.Queue] = {}
        self.state = "idle"  # idle | live | stopped
        self.partial: dict = {"type": "partial", "text": "", "upto": -1}

    def new_session(self) -> None:
        self.session = secrets.token_hex(6)
        self.started_at = time.time()
        self.lines: list[dict] = []
        self.events: list[dict] = []  # events[i]["ev"] == i
        self.partial = {"type": "partial", "text": "", "upto": -1}

    # ------------------------------------------------------------------ lines
    def _publish(self, kind: str, line: dict) -> dict:
        ev = {"type": "ev", "ev": len(self.events), "kind": kind, "line": dict(line)}
        self.events.append(ev)
        self._fanout(ev)
        return ev

    def add_line(self, en: str, t0: float, t1: float, seqs: list[int]) -> dict:
        line = {"id": len(self.lines), "en": en, "uk": None, "t0": round(t0, 2), "t1": round(t1, 2),
                "seqs": seqs, "status": "translating", "err": ""}
        self.lines.append(line)
        self._publish("line", line)
        self.set_partial("", line["id"])
        return line

    def set_translation(self, line_id: int, uk: str, provider: str = "") -> dict:
        line = self.lines[line_id]
        line["uk"] = uk
        line["status"] = "done"
        line["err"] = ""
        line["prov"] = provider
        self._publish("update", line)
        return line

    def mark_retrying(self, line_id: int, err: str) -> None:
        line = self.lines[line_id]
        if line["status"] != "retrying":
            line["status"] = "retrying"
            line["err"] = err[:200]
            self._publish("update", line)

    # ------------------------------------------------------------------ ephemeral
    def set_partial(self, text: str, upto: Optional[int] = None) -> None:
        if upto is None:
            upto = len(self.lines) - 1
        msg = {"type": "partial", "text": text, "upto": upto}
        if msg != self.partial:
            self.partial = msg
            self._fanout(msg)

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self._fanout(self.status_msg())

    def status_msg(self) -> dict:
        return {"type": "status", "state": self.state, "session": self.session, "has_lines": bool(self.lines)}

    # ------------------------------------------------------------------ clients
    def _fanout(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False)
        for ws, q in list(self.clients.items()):
            if q.qsize() > self.MAX_CLIENT_BACKLOG:
                q.put_nowait(None)  # tell writer to close; client will reconnect + replay
                self.clients.pop(ws, None)
            else:
                q.put_nowait(data)

    def register(self, ws) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.clients[ws] = q
        return q

    def unregister(self, ws) -> None:
        self.clients.pop(ws, None)

    def sync_messages(self, session: Optional[str], last_ev: int) -> list[dict]:
        """What a (re)connecting client needs: missed events, or a full snapshot."""
        msgs: list[dict] = [self.status_msg()]
        if session == self.session and 0 <= last_ev + 1 <= len(self.events) and last_ev >= -1 and session:
            missed = self.events[last_ev + 1:]
            msgs.append({"type": "replay", "session": self.session, "events": missed, "ev": len(self.events) - 1})
        else:
            msgs.append({"type": "snapshot", "session": self.session, "started_at": self.started_at,
                         "lines": self.lines, "ev": len(self.events) - 1})
        msgs.append(self.partial)
        return msgs
