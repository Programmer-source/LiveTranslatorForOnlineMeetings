"""Offline STT accuracy check: segment + transcribe an audio file and diff against its script.

Usage:  .runtime\\venv\\Scripts\\python.exe -m tests.stt_accuracy [audio.wav] [script.txt]
"""
import difflib
import re
import sys
import time
from pathlib import Path

from app import paths
from app.cuda_setup import choose_whisper_model, detect_hardware

ROOT = Path(__file__).resolve().parent


def words(x: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", x.lower().replace("-", " ")).split()


def main() -> None:
    wav = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "lecture.wav")
    ref_path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data" / "lecture.txt")
    from app.audio import decode_file

    from app.segmenter import Segmenter
    from app.stt import STTStage, WhisperEngine
    from app.vad import SileroVAD

    audio = decode_file(wav)
    segs = []
    sg = Segmenter(SileroVAD(), segs.append)
    for i in range(0, len(audio), 1600):
        sg.feed(audio[i:i + 1600], 1000 + (i + 1600) / 16000)
    sg.flush()
    print(f"segments: {len(segs)}  durations: {[round(s.duration, 1) for s in segs]}")
    hw = detect_hardware()
    name, device, ctype = choose_whisper_model("auto", hw)
    t = time.time()
    eng = WhisperEngine(name, device, ctype, str(paths.MODELS_DIR / "whisper"))
    print(f"model {eng.name} on {eng.device} loaded in {time.time() - t:.1f}s")
    st = STTStage(eng, lambda: [])
    out = []
    t = time.time()
    for s in segs:
        tr = st.process(s)
        out.append(tr.text)
        if tr.dropped:
            print("  dropped:", tr.dropped)
    el = time.time() - t
    print(f"STT: {el:.1f}s for {len(audio) / 16000:.1f}s of audio (RTF {el / (len(audio) / 16000):.2f})")
    ref = words(Path(ref_path).read_text(encoding="utf-8"))
    hyp = words(" ".join(out))
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    diffs = [(op, " ".join(ref[a:b]), " ".join(hyp[c:d])) for op, a, b, c, d in sm.get_opcodes() if op != "equal"]
    errs = sum(max(b - a, d - c) for op, a, b, c, d in sm.get_opcodes() if op != "equal")
    print(f"reference words: {len(ref)}  recognised: {len(hyp)}  WER≈{100 * errs / len(ref):.1f}%")
    for d in diffs:
        print("  ", d)


if __name__ == "__main__":
    main()
