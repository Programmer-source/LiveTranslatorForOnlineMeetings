"""FastAPI app: public viewer (/ and /ws) + local-only host control panel (/control, /api/*, /ws/control)."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import __version__, audio, paths
from .config import Config, Glossary
from .hub import Hub
from .pipeline import Pipeline
from .tunnel import Tunnel

log = logging.getLogger("server")


def _no_cache(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-cache"
    return resp


class AppState:
    def __init__(self, config: Config, glossary: Glossary, *, enable_tunnel: bool = True, pipeline_kwargs=None):
        self.config = config
        self.glossary = glossary
        self.hub = Hub()
        self.pipeline = Pipeline(config, glossary, self.hub, **(pipeline_kwargs or {}))
        self.tunnel = Tunnel(int(config.get("port", 8765)))
        self.enable_tunnel = enable_tunnel and config.get("tunnel_enabled", True)
        self.shutdown_event = asyncio.Event()


def create_app(state: AppState) -> FastAPI:
    port = int(state.config.get("port", 8765))
    local_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost", "testserver"}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.pipeline.start_background()
        if state.enable_tunnel:
            state.tunnel.start()
        yield
        state.pipeline.stop()
        state.tunnel.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def is_local(headers, client_host: Optional[str]) -> bool:
        """Requests through the tunnel also come from 127.0.0.1, so check Host and Cloudflare headers too."""
        if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            return False
        if any(h in headers for h in ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "cf-ipcountry")):
            return False
        return headers.get("host", "") in local_hosts

    def require_local(request: Request, write: bool = False) -> None:
        if not is_local(request.headers, request.client.host if request.client else None):
            raise HTTPException(status_code=404)
        # CSRF guard: custom header forces a CORS pre-flight that we never approve.
        if write and request.headers.get("x-control") != "1":
            raise HTTPException(status_code=403)

    # ------------------------------------------------------------------ viewer (public)
    @app.get("/", response_class=HTMLResponse)
    async def viewer():
        return _no_cache(FileResponse(paths.WEB_DIR / "viewer.html", media_type="text/html; charset=utf-8"))

    @app.get("/favicon.ico")
    async def favicon():
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="16" fill="#0057b7"/>'
               '<rect y="16" width="32" height="16" fill="#ffd700"/></svg>')
        return Response(svg, media_type="image/svg+xml")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_viewer(ws: WebSocket):
        await ws.accept()
        q = state.hub.register(ws)

        async def writer():
            while True:
                msg = await q.get()
                if msg is None:
                    await ws.close()
                    return
                await ws.send_text(msg)

        wtask = asyncio.create_task(writer())
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("type") == "hello":
                    for m in state.hub.sync_messages(msg.get("session"), int(msg.get("last_ev", -1))):
                        q.put_nowait(json.dumps(m, ensure_ascii=False))
                elif msg.get("type") == "ping":
                    q.put_nowait('{"type":"pong"}')
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state.hub.unregister(ws)
            wtask.cancel()

    # ------------------------------------------------------------------ control (local only)
    @app.get("/control", response_class=HTMLResponse)
    async def control(request: Request):
        require_local(request)
        return _no_cache(FileResponse(paths.WEB_DIR / "control.html", media_type="text/html; charset=utf-8"))

    def status() -> dict:
        p, t, z = state.pipeline, state.tunnel, state.pipeline.zoom
        tr = p.translator
        eng = p.engine
        src = p.source
        audio_err = p.audio_error or (src.error if src else "")
        silent_for = time.time() - p.meter.last_data_time if p.meter.last_data_time else None
        if not p.running:
            a_state = "idle"
        elif audio_err:
            a_state = "error"
        elif silent_for is None or silent_for > 10:
            a_state = "warn"
        else:
            a_state = "ok"
        test_info = None
        if src is not None and hasattr(src, "duration"):
            test_info = {"pos": round(src.position, 1), "dur": round(src.duration, 1)}
        return {
            "type": "status",
            "version": __version__,
            "running": p.running,
            "mode": p.mode,
            "level": round(p.meter.current(), 3),
            "audio": {"state": a_state, "device": src.name if src else "", "error": audio_err,
                      "silent_for": round(silent_for, 1) if silent_for is not None else None},
            "stt": {"state": {"ready": "ok", "loading": "warn", "error": "error"}[p.engine_state]
                    if not (p.stt_errors and p.engine_state == "ready") else "warn",
                    "model": getattr(eng, "name", ""), "device": getattr(eng, "device", ""),
                    "gpu": p.hw.get("gpu_name", ""), "message": p.engine_msg},
            "translator": {"state": {"idle": "ok", "ok": "ok", "retrying": "warn", "error": "error"}[tr.state],
                           "provider": tr.last_provider, "error": tr.last_error,
                           "has_gemini": bool(state.config.get("gemini_api_key"))},
            "tunnel": {"state": {"off": "idle", "downloading": "warn", "starting": "warn", "ok": "ok", "error": "error"}[t.state],
                       "url": t.url, "message": t.message, "pct": t.download_pct, "enabled": state.enable_tunnel},
            "zoom": {"state": z.state if z.enabled() else "off", "sent": z.sent, "failed": z.failed,
                     "error": z.last_error, "queued": z.queue.qsize(), "configured": bool(z.url)},
            "delay": p.delay(),
            "backlog": p.backlog(),
            "lines": len(state.hub.lines),
            "viewers": len(state.hub.clients),
            "session": state.hub.session,
            "test": test_info,
            "preview": list(p.preview),
            "partial": state.hub.partial.get("text", ""),
            "pending_lines": [{"id": l["id"], "en": l["en"], "status": l["status"], "err": l["err"]}
                              for l in state.hub.lines[-6:] if l["status"] != "done"],
        }

    @app.get("/api/state")
    async def api_state(request: Request):
        require_local(request)
        return {"config": state.config.public_view(), "glossary": state.glossary.items,
                "status": status(), "local_url": f"http://127.0.0.1:{port}/"}

    @app.websocket("/ws/control")
    async def ws_control(ws: WebSocket):
        origin = ws.headers.get("origin", "")
        client = ws.client.host if ws.client else None
        if not is_local(ws.headers, client) or (origin and not re.match(rf"^http://(127\.0\.0\.1|localhost):{port}$", origin)):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            while True:
                await ws.send_text(json.dumps(status(), ensure_ascii=False))
                await asyncio.sleep(0.2)
        except Exception:
            pass

    @app.get("/api/devices")
    async def api_devices(request: Request):
        require_local(request)
        return await asyncio.to_thread(audio.list_devices)

    @app.post("/api/start")
    async def api_start(request: Request):
        require_local(request, write=True)
        body = await request.json() if await request.body() else {}
        device = body.get("device") or state.config.get("audio_device", "default")
        state.config.update({"audio_device": device})
        await asyncio.to_thread(state.pipeline.start, device)
        return {"ok": True}

    @app.post("/api/stop")
    async def api_stop(request: Request):
        require_local(request, write=True)
        await asyncio.to_thread(state.pipeline.stop)
        return {"ok": True}

    @app.post("/api/test")
    async def api_test(request: Request):
        require_local(request, write=True)
        body = await request.json() if await request.body() else {}
        name = body.get("file")
        if name:
            path = paths.UPLOADS_DIR / os.path.basename(name)
            if not path.exists():
                raise HTTPException(404, "файл не знайдено")
        else:
            path = paths.TEST_SAMPLE
        await asyncio.to_thread(state.pipeline.start, None, str(path))
        return {"ok": True}

    @app.post("/api/upload")
    async def api_upload(request: Request, file: UploadFile = File(...)):
        require_local(request, write=True)
        ext = os.path.splitext(file.filename or "")[1][:8] or ".bin"
        name = uuid.uuid4().hex + re.sub(r"[^.\w]", "", ext)
        dest = paths.UPLOADS_DIR / name
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        return {"ok": True, "file": name}

    @app.post("/api/config")
    async def api_config(request: Request):
        require_local(request, write=True)
        body = await request.json()
        allowed = {k: v for k, v in body.items() if k in state.config.data and k != "zoom_seq_state" and k != "port"}
        for k in ("gemini_api_key", "deepl_api_key", "deepgram_api_key", "zoom_caption_url"):
            if k in allowed and isinstance(allowed[k], str):
                allowed[k] = allowed[k].strip()
        if isinstance(allowed.get("gemini_models"), str):
            allowed["gemini_models"] = [m.strip() for m in allowed["gemini_models"].split(",") if m.strip()]
        state.config.update(allowed)
        if any(k in allowed for k in ("gemini_api_key", "deepl_api_key", "gemini_models", "gemini_rpm",
                                      "free_fallback_translator")):
            state.pipeline.reload_translator()
        return {"ok": True, "config": state.config.public_view()}

    @app.get("/api/glossary")
    async def api_glossary_get(request: Request):
        require_local(request)
        return state.glossary.items

    @app.post("/api/glossary")
    async def api_glossary(request: Request):
        require_local(request, write=True)
        body = await request.json()
        rows = body.get("items", body)
        if isinstance(rows, list):
            rows = {str(r[0]): str(r[1]) for r in rows if r and str(r[0]).strip()}
        state.glossary.save(rows)
        return {"ok": True, "glossary": state.glossary.items}

    @app.post("/api/zoom/test")
    async def api_zoom_test(request: Request):
        require_local(request, write=True)
        body = await request.json() if await request.body() else {}
        if "url" in body:
            state.config.update({"zoom_caption_url": (body.get("url") or "").strip()})
        ok, msg = await state.pipeline.zoom.test()
        return {"ok": ok, "message": msg}

    @app.post("/api/translator/test")
    async def api_translator_test(request: Request):
        require_local(request, write=True)
        from .translate import Ctx, GeminiProvider, ProviderError

        key = state.config.get("gemini_api_key")
        if not key:
            return {"ok": False, "message": "Ключ Gemini не введено"}
        g = GeminiProvider(key, state.config.get("gemini_models", []), 60, state.pipeline.http)
        try:
            out = await g.translate(["Good morning, everyone! Welcome to the webinar."], Ctx([], {}, [], ""))
            return {"ok": True, "message": f"✅ Gemini працює ({g.last_model}): {out[0]}"}
        except ProviderError as e:
            return {"ok": False, "message": f"❌ {e}"}
        except Exception as e:
            return {"ok": False, "message": f"❌ {e}"}

    @app.post("/api/session/new")
    async def api_new_session(request: Request):
        require_local(request, write=True)
        p = state.pipeline
        if p.backlog() or p.pending:
            return {"ok": False, "message": "Зачекайте: ще перекладаються попередні рядки."}
        await asyncio.to_thread(p.new_session)
        return {"ok": True}

    @app.post("/api/open-folder")
    async def api_open_folder(request: Request):
        require_local(request, write=True)
        p = state.pipeline
        target = p.session.dir if p.session else paths.TRANSCRIPTS_DIR
        try:
            os.startfile(str(target))  # type: ignore[attr-defined]
        except Exception:
            pass
        return {"ok": True, "path": str(target)}

    @app.get("/api/qr.svg")
    async def api_qr(request: Request, url: str):
        require_local(request)
        import qrcode
        import qrcode.image.svg

        img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
        buf = io.BytesIO()
        img.save(buf)
        return Response(buf.getvalue(), media_type="image/svg+xml")

    @app.post("/api/shutdown")
    async def api_shutdown(request: Request):
        require_local(request, write=True)
        state.shutdown_event.set()
        return {"ok": True}

    return app
