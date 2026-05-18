#!/usr/bin/env python3
"""Inference wrapper for the TensorFlow Lite keyword spotting model."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import kws_native
except ImportError as exc:
    raise RuntimeError(
        "kws_native is not built yet. Run catkin_make/catkin build from the workspace first."
    ) from exc

SAMPLE_RATE = int(kws_native.SAMPLE_RATE)
EXPECTED_SAMPLES = int(kws_native.EXPECTED_SAMPLES)
HOP_LENGTH = int(kws_native.HOP_LENGTH)
DEFAULT_LABELS = ("background", "marvin")


def _load_tflite_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            return Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow Lite inference requires either `tflite_runtime` or `tensorflow`."
            ) from exc


def _pcm_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (audio - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483647.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def load_wav_mono(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        source_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if source_rate != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz WAV, got {source_rate} Hz")

    audio = _pcm_to_float32(raw, sample_width)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False)


def get_spectrogram(audio: np.ndarray, normalize: bool = True) -> np.ndarray:
    return kws_native.get_spectrogram(np.asarray(audio, dtype=np.float32).reshape(-1), normalize)


def to_model_input(spectrogram: np.ndarray) -> np.ndarray:
    return kws_native.to_model_input(np.asarray(spectrogram, dtype=np.float32))


class StreamingSpectrogram:
    def __init__(self, window_samples: int = EXPECTED_SAMPLES, emit_hop_samples: int = HOP_LENGTH) -> None:
        self.emit_hop_samples = int(emit_hop_samples)
        self.ring = kws_native.RingBuffer(int(window_samples))
        self._samples_since_emit = 0

    @property
    def ready(self) -> bool:
        return bool(self.ring.is_full())

    def reset(self) -> None:
        self.ring.reset()
        self._samples_since_emit = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        emitted = []
        for start in range(0, samples.size, self.emit_hop_samples):
            part = samples[start : start + self.emit_hop_samples]
            self.ring.push(part)
            self._samples_since_emit += part.size
            if self.ready and self._samples_since_emit >= self.emit_hop_samples:
                emitted.append(self.current_spectrogram())
                self._samples_since_emit = 0
        return emitted

    def current_audio(self) -> np.ndarray:
        return self.ring.current_audio()

    def current_spectrogram(self) -> np.ndarray:
        return get_spectrogram(self.current_audio())

    def current_model_input(self) -> np.ndarray:
        return to_model_input(self.current_spectrogram())


class KeywordSpotter:
    def __init__(
        self,
        model_path: str | Path,
        labels: Iterable[str] = DEFAULT_LABELS,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = tuple(labels)
        self.threshold = float(threshold)

        Interpreter = _load_tflite_interpreter()
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self.input_dtype = self.input_details[0]["dtype"]

    def _prepare_input(self, spectrogram: np.ndarray) -> np.ndarray:
        model_input = to_model_input(spectrogram)
        if self.input_dtype == np.float32:
            return model_input.astype(np.float32, copy=False)

        quant = self.input_details[0].get("quantization", (0.0, 0))
        scale, zero_point = quant
        if scale and scale > 0:
            model_input = np.round(model_input / scale + zero_point)
        return model_input.astype(self.input_dtype)

    def _read_score(self) -> float:
        output = self.interpreter.get_tensor(self.output_index)
        output = np.asarray(output)

        if output.dtype != np.float32:
            quant = self.output_details[0].get("quantization", (0.0, 0))
            scale, zero_point = quant
            if scale and scale > 0:
                output = (output.astype(np.float32) - zero_point) * scale

        scores = output.reshape(-1).astype(np.float32)
        if scores.size == 1:
            return float(scores[0])
        if scores.size >= 2:
            return float(scores[1])
        raise RuntimeError("TFLite model returned an empty output tensor")

    def predict_spectrogram(self, spectrogram: np.ndarray) -> dict[str, float | str]:
        model_input = self._prepare_input(spectrogram)
        self.interpreter.set_tensor(self.input_index, model_input)
        self.interpreter.invoke()
        score = self._read_score()
        label = self.labels[1] if score >= self.threshold else self.labels[0]
        return {"label": label, "score": score}

    def predict_audio(self, audio: np.ndarray) -> dict[str, float | str]:
        return self.predict_spectrogram(get_spectrogram(audio))

    def predict_file(self, wav_path: str | Path) -> dict[str, float | str]:
        return self.predict_audio(load_wav_mono(wav_path))

    def stream_audio(self, audio: np.ndarray, chunk_samples: int = HOP_LENGTH) -> list[dict[str, float | str | int]]:
        streamer = StreamingSpectrogram(emit_hop_samples=chunk_samples)
        results = []
        window_index = 0
        for start in range(0, audio.size, chunk_samples):
            for spec in streamer.push(audio[start : start + chunk_samples]):
                result = self.predict_spectrogram(spec)
                result["window"] = window_index
                results.append(result)
                window_index += 1
        return results

    def stream_file(self, wav_path: str | Path, chunk_samples: int = HOP_LENGTH) -> list[dict[str, float | str | int]]:
        return self.stream_audio(load_wav_mono(wav_path), chunk_samples=chunk_samples)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run TFLite keyword inference on a 16 kHz WAV file.")
    parser.add_argument("wav_path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    spotter = KeywordSpotter(args.model, threshold=args.threshold)
    if args.stream:
        for result in spotter.stream_file(args.wav_path):
            print(result)
    else:
        print(spotter.predict_file(args.wav_path))


if __name__ == "__main__":
    main()
