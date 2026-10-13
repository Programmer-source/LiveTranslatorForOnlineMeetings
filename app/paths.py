"""All filesystem locations. Everything lives inside the project folder."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):  # PyInstaller build
    ROOT = Path(sys.executable).resolve().parent
    BUNDLE = Path(getattr(sys, "_MEIPASS", ROOT))
else:
    ROOT = Path(__file__).resolve().parent.parent
    BUNDLE = ROOT

WEB_DIR = BUNDLE / "web"
ASSETS_DIR = BUNDLE / "app" / "assets"
RUNTIME_DIR = ROOT / ".runtime"
MODELS_DIR = RUNTIME_DIR / "models"
TOOLS_DIR = RUNTIME_DIR / "tools"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
TRANSCRIPTS_DIR = ROOT / "transcripts"
LOGS_DIR = ROOT / "logs"
CONFIG_PATH = ROOT / "config.json"
CONFIG_EXAMPLE_PATH = BUNDLE / "config.example.json"
GLOSSARY_PATH = ROOT / "glossary.json"
CLOUDFLARED_EXE = TOOLS_DIR / "cloudflared.exe"
TEST_SAMPLE = ASSETS_DIR / "test_sample.wav"

# Keep Hugging Face downloads inside the project.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

for _d in (RUNTIME_DIR, MODELS_DIR, TOOLS_DIR, UPLOADS_DIR, TRANSCRIPTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
