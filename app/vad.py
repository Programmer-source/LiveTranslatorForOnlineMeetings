"""Streaming Silero VAD (ONNX model shipped with faster-whisper; no torch needed)."""
from __future__ import annotations

import os

import numpy as np

FRAME = 512  # samples @ 16 kHz = 32 ms
CONTEXT = 64


def _model_path() -> str:
    import faster_whisper

    assets = os.path.join(os.path.dirname(faster_whisper.__file__), "assets")
    for name in sorted(os.listdir(assets), reverse=True):
        if name.startswith("silero_vad") and name.endswith(".onnx"):
            return os.path.join(assets, name)
    raise FileNotFoundError("Silero VAD model not found in faster_whisper/assets")


class SileroVAD:
    def __init__(self):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(_model_path(), providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """frames: (n, 512) float32 -> speech probabilities (n,)."""
        if frames.shape[0] == 0:
            return np.zeros(0, np.float32)
        ctx = np.empty((frames.shape[0], CONTEXT), np.float32)
        ctx[0] = self.context
        if frames.shape[0] > 1:
            ctx[1:] = frames[:-1, -CONTEXT:]
        x = np.concatenate([ctx, frames], axis=1).astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frames[-1, -CONTEXT:].copy()
        return np.asarray(out, np.float32).reshape(-1)
