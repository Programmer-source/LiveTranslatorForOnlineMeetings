"""Audio sources. Every source delivers 16 kHz mono float32 blocks, stamped with the
wall-clock time of the block end, into an unbounded queue (nothing is ever dropped)."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("audio")

SR = 16000


@dataclass
class AudioBlock:
    samples: np.ndarray  # float32, 16 kHz mono
    wall_end: float  # time.time() when the last sample was captured
    eof: bool = False


class AudioError(Exception):
    """Error with a plain-Ukrainian message for the host."""


class LevelMeter:
    def __init__(self):
        self.level = 0.0  # 0..1 (peak-ish, decaying)
        self.last_data_time = 0.0

    def feed(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        db = 20 * np.log10(max(rms, 1e-6))
        lvl = min(1.0, max(0.0, (db + 60) / 60))  # -60 dBFS .. 0 dBFS -> 0..1
        self.level = max(lvl, self.level * 0.7)
        self.last_data_time = time.time()

    def current(self) -> float:
        if time.time() - self.last_data_time > 0.5:
            self.level *= 0.5
        return self.level


class _Resampler:
    def __init__(self, in_rate: int, channels: int):
        self.in_rate = in_rate
        self.channels = channels
        self._rs = None
        if in_rate != SR:
            import soxr

            self._rs = soxr.ResampleStream(in_rate, SR, 1, dtype="float32", quality="HQ")

    def __call__(self, interleaved: np.ndarray, last: bool = False) -> np.ndarray:
        x = interleaved
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self._rs is not None:
            x = self._rs.resample_chunk(x, last=last)
        return x.astype(np.float32, copy=False)


# ----------------------------------------------------------------------------- devices

def list_devices() -> list[dict]:
    """Loopback (what you hear) devices first, then microphones."""
    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        return [{"id": "default", "name": "Системний звук (типовий пристрій)", "kind": "loopback"}]
    out = [{"id": "default", "name": "🔊 Системний звук — типовий пристрій виводу (рекомендовано)", "kind": "loopback"}]
    p = pyaudio.PyAudio()
    try:
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return out
        loops, mics = [], []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get("hostApi") != wasapi["index"]:
                continue
            if d.get("isLoopbackDevice"):
                name = d["name"].replace(" [Loopback]", "")
                loops.append({"id": "loop:" + name, "name": "🔊 " + name, "kind": "loopback"})
            elif d.get("maxInputChannels", 0) > 0:
                mics.append({"id": "mic:" + d["name"], "name": "🎤 " + d["name"] + " (мікрофон, для тесту)", "kind": "mic"})
        return out + loops + mics
    finally:
        p.terminate()


def _resolve_device(p, pyaudio, device_id: str) -> dict:
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        raise AudioError("WASAPI недоступний у цій системі Windows.")
    if device_id in ("", "default", None):
        try:
            spk = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        except Exception:
            raise AudioError("Не знайдено пристрій виводу звуку. Підключіть навушники або динаміки.")
        if spk.get("isLoopbackDevice"):
            return spk
        for lb in p.get_loopback_device_info_generator():
            if spk["name"] in lb["name"]:
                return lb
        raise AudioError("Не вдалося під'єднатися до системного звуку (loopback). Оберіть інший пристрій.")
    kind, _, name = device_id.partition(":")
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("hostApi") != wasapi["index"]:
            continue
        if kind == "loop" and d.get("isLoopbackDevice") and d["name"].replace(" [Loopback]", "") == name:
            return d
        if kind == "mic" and not d.get("isLoopbackDevice") and d["name"] == name and d.get("maxInputChannels", 0) > 0:
            return d
    raise AudioError("Обраний аудіопристрій не знайдено (можливо, відключено). Оберіть інший у списку.")


def default_output_name() -> str:
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            return p.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
        finally:
            p.terminate()
    except Exception:
        return ""


# ----------------------------------------------------------------------------- sources

class BaseSource:
    def __init__(self, out: "queue.Queue[AudioBlock]", meter: LevelMeter):
        self.out = out
        self.meter = meter
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str = ""
        self.name = ""

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run_safe, daemon=True, name=type(self).__name__)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _emit(self, x: np.ndarray, wall_end: float) -> None:
        if x.size:
            self.meter.feed(x)
            self.out.put(AudioBlock(x, wall_end))

    def _run_safe(self) -> None:
        try:
            self._run()
        except AudioError as e:
            self.error = str(e)
            log.error("audio: %s", e)
        except Exception as e:  # pragma: no cover - hardware specific
            log.exception("audio source crashed")
            self.error = f"Помилка захоплення звуку: {e}"
        finally:
            self.out.put(AudioBlock(np.zeros(0, np.float32), time.time(), eof=True))

    def _run(self) -> None:
        raise NotImplementedError


class WasapiSource(BaseSource):
    """WASAPI loopback (system audio) or microphone via pyaudiowpatch.

    The PortAudio callback only copies raw bytes into an unbounded queue; conversion
    happens in this thread, so a slow consumer can never cause dropped audio."""

    def __init__(self, out, meter, device_id: str = "default"):
        super().__init__(out, meter)
        self.device_id = device_id

    def _run(self) -> None:
        import pyaudiowpatch as pyaudio

        while not self._stop.is_set():
            p = pyaudio.PyAudio()
            raw: "queue.Queue[tuple[bytes, float]]" = queue.Queue()
            stream = None
            try:
                dev = _resolve_device(p, pyaudio, self.device_id)
                channels = max(1, int(dev["maxInputChannels"]))
                rate = int(dev["defaultSampleRate"])
                self.name = dev["name"]
                rs = _Resampler(rate, channels)

                def cb(in_data, frame_count, time_info, status):
                    raw.put((in_data, time.time()))
                    return (None, pyaudio.paContinue)

                stream = p.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                                input_device_index=dev["index"], frames_per_buffer=int(rate * 0.05),
                                stream_callback=cb)
                stream.start_stream()
                self.error = ""
                log.info("capturing from %s (%d Hz, %d ch)", self.name, rate, channels)
                default_name = default_output_name() if self.device_id == "default" else ""
                last_check = time.time()
                while not self._stop.is_set():
                    try:
                        data, t = raw.get(timeout=0.2)
                        self._emit(rs(np.frombuffer(data, dtype=np.float32)), t)
                    except queue.Empty:
                        pass
                    if not stream.is_active():
                        raise AudioError("Аудіопотік зупинився (пристрій відключено?). Перепідключаюсь…")
                    # Follow the Windows default output device if the host switches headphones/speakers.
                    if self.device_id == "default" and time.time() - last_check > 3:
                        last_check = time.time()
                        now = default_output_name()
                        if now and default_name and now != default_name:
                            log.info("default output changed: %s -> %s", default_name, now)
                            break
                # drain whatever the callback already delivered — never lose audio
                while True:
                    try:
                        data, t = raw.get_nowait()
                        self._emit(rs(np.frombuffer(data, dtype=np.float32)), t)
                    except queue.Empty:
                        break
            except AudioError as e:
                self.error = str(e)
                log.warning("%s", e)
                self._stop.wait(2.0)
            except OSError as e:
                self.error = "Не вдалося відкрити аудіопристрій: " + str(e)
                log.warning("%s", e)
                self._stop.wait(2.0)
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                p.terminate()


class FileSource(BaseSource):
    """Feeds an audio/video file into the pipeline (test mode). `speed` > 1 feeds faster than real time."""

    def __init__(self, out, meter, path: str, speed: float = 1.0, tail_silence: float = 2.0):
        super().__init__(out, meter)
        self.path = path
        self.speed = speed
        self.tail_silence = tail_silence
        self.name = "файл: " + str(path).replace("\\", "/").rsplit("/", 1)[-1]
        self.duration = 0.0
        self.position = 0.0

    def _run(self) -> None:
        try:
            from faster_whisper.audio import decode_audio

            audio = decode_audio(self.path, sampling_rate=SR)
        except Exception as e:
            raise AudioError(f"Не вдалося прочитати файл (формат не підтримується?): {e}")
        audio = np.concatenate([audio.astype(np.float32), np.zeros(int(self.tail_silence * SR), np.float32)])
        self.duration = len(audio) / SR
        block = int(0.1 * SR)
        t0 = time.time()
        for i in range(0, len(audio), block):
            if self._stop.is_set():
                break
            chunk = audio[i:i + block]
            due = t0 + (i + len(chunk)) / SR / self.speed
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            self.position = (i + len(chunk)) / SR
            self._emit(chunk, time.time())
