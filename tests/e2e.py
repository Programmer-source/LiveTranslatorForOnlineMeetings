"""End-to-end test against the real app (python -m app must NOT be running).

Starts the app, uploads tests/data/lecture.wav, plays it through the pipeline in real time,
listens like a viewer over WebSocket and checks that every line arrives translated, in order,
and is saved to the transcript files. Prints average latency (speech end -> UA line shown).

    .runtime\venv\Scripts\python.exe -m tests.e2e
"""
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
H = {"X-Control": "1"}


def words(x):
    return re.sub(r"[^a-z0-9' ]", " ", x.lower().replace("-", " ")).split()


async def main():
    proc = subprocess.Popen([sys.executable, "-m", "app", "--no-browser"], cwd=ROOT)
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            for _ in range(120):
                try:
                    if (await c.get(BASE + "/healthz")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
            for _ in range(240):  # wait for the speech model
                st = (await c.get(BASE + "/api/state")).json()["status"]
                if st["stt"]["model"]:
                    break
                await asyncio.sleep(0.5)
            print("STT:", st["stt"]["model"], st["stt"]["device"], "| translator gemini key:", st["translator"]["has_gemini"])
            r = await c.post(BASE + "/api/session/new", headers=H, json={})
            up = await c.post(BASE + "/api/upload", headers=H,
                              files={"file": ("lecture.wav", open(ROOT / "tests/data/lecture.wav", "rb"), "audio/wav")})
            name = up.json()["file"]
            lines = {}
            recv_at = {}
            async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws") as ws:
                await ws.send(json.dumps({"type": "hello", "session": None, "last_ev": -1}))
                await c.post(BASE + "/api/test", headers=H, json={"file": name})
                t0 = time.time()
                started = False
                while True:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    except asyncio.TimeoutError:
                        msg = {}
                    if msg.get("type") == "ev":
                        ln = msg["line"]
                        lines[ln["id"]] = ln
                        if ln.get("uk") and ln["id"] not in recv_at:
                            recv_at[ln["id"]] = time.time()
                            print(f"  [{ln['id']:2d}] {ln['uk']}")
                    if msg.get("type") == "status" and msg["state"] == "live":
                        started = True
                    st = (await c.get(BASE + "/api/state")).json()["status"]
                    if started and not st["running"] and st["backlog"] == 0 and lines and all(l.get("uk") for l in lines.values()):
                        break
                    if time.time() - t0 > 600:
                        raise SystemExit("TIMEOUT")
            ids = sorted(lines)
            assert ids == list(range(len(ids))), "line ids not contiguous"
            en = " ".join(lines[i]["en"] for i in ids)
            ref = words((ROOT / "tests/data/lecture.txt").read_text(encoding="utf-8"))
            import difflib
            sm = difflib.SequenceMatcher(a=ref, b=words(en), autojunk=False)
            errs = sum(max(b - a, d - cc) for op, a, b, cc, d in sm.get_opcodes() if op != "equal")
            lat = [lines[i]["lat"] for i in ids if "lat" in lines[i]]
            print(f"\nlines: {len(ids)}, all translated in order: True")
            print(f"English words vs script: {len(words(en))}/{len(ref)}, WER≈{100 * errs / len(ref):.1f}%")
            print(f"latency speech-end -> UA line: avg {sum(lat) / len(lat):.2f}s, max {max(lat):.2f}s")
            print("providers used:", sorted({lines[i].get("prov", "") for i in ids}))
            # transcript files
            d = sorted((ROOT / "transcripts").iterdir(), key=lambda p: p.stat().st_mtime)[-1]
            uk_file = (d / "transcript_uk.txt").read_text(encoding="utf-8").strip().splitlines()
            srt = (d / "subtitles_uk.srt").read_text(encoding="utf-8")
            assert len(uk_file) == len(ids), (len(uk_file), len(ids))
            for i, row in zip(ids, uk_file):
                assert row.endswith(lines[i]["uk"]), (row, lines[i]["uk"])
            assert srt.count(" --> ") == len(ids)
            print("saved transcript:", d, "- OK, matches the viewer page line-for-line")
            await c.post(BASE + "/api/shutdown", headers=H, json={})
    finally:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
