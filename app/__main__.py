"""Entry point: python -m app"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import threading
import time
import webbrowser

from . import paths  # noqa: F401  (sets HF_HOME before anything imports huggingface)
from .cuda_setup import register_cuda_dlls


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(paths.LOGS_DIR / time.strftime("app_%Y-%m-%d.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    root.addHandler(sh)
    for noisy in ("httpx", "httpcore", "uvicorn.access", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def our_app_running(port: int) -> bool:
    try:
        import httpx

        return httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.5).json().get("ok") is True
    except Exception:
        return False


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    setup_logging()
    register_cuda_dlls()
    from .config import Config, Glossary

    config = Config()
    if not paths.CONFIG_PATH.exists():
        config.save()
    glossary = Glossary()
    port = int(config.get("port", 8765))
    no_browser = "--no-browser" in sys.argv

    if port_in_use(port):
        if our_app_running(port):
            print("Програма вже запущена — відкриваю панель керування.")
            if not no_browser:
                webbrowser.open(f"http://127.0.0.1:{port}/control")
            return
        for p in range(port + 1, port + 50):
            if not port_in_use(p):
                port = p
                break
        config.data["port"] = port  # this run only

    import uvicorn

    from .server import AppState, create_app

    state = AppState(config, glossary)
    state.tunnel.port = port
    app = create_app(state)
    uv_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False,
                               ws_ping_interval=20, ws_ping_timeout=30)
    server = uvicorn.Server(uv_config)

    async def watch_shutdown():
        await state.shutdown_event.wait()
        server.should_exit = True

    async def run():
        asyncio.create_task(watch_shutdown())
        await server.serve()

    def open_browser():
        for _ in range(100):
            if port_in_use(port):
                break
            time.sleep(0.1)
        if not no_browser:
            webbrowser.open(f"http://127.0.0.1:{port}/control")

    # Closing the console window: stop the tunnel cleanly (the Job object kills it anyway).
    if sys.platform == "win32":
        try:
            import ctypes

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
            def handler(ctrl):
                try:
                    state.tunnel.stop()
                    if state.pipeline.session:
                        state.pipeline.session.close()
                except Exception:
                    pass
                os._exit(0)

            ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
            main._handler = handler  # keep a reference
        except Exception:
            pass

    print("=" * 64)
    print("  Zoom Live UA Subtitles запущено")
    print(f"  Панель керування: http://127.0.0.1:{port}/control")
    print("  Щоб зупинити програму — просто закрийте це вікно.")
    print("=" * 64)
    threading.Thread(target=open_browser, daemon=True).start()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        state.tunnel.stop()
        if state.pipeline.session:
            state.pipeline.session.close()
    print("Програму зупинено.")


if __name__ == "__main__":
    main()
