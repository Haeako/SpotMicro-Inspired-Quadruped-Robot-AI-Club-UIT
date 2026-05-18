"""Inference wrapper for the streaming keyword model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from spectrogram import (
    EXPECTED_SAMPLES,
    HOP_LENGTH,
    SAMPLE_RATE,
    StreamingSpectrogram,
    get_spectrogram,
    load_wav_mono,
    to_model_input,
)


DEFAULT_LABELS = ("background", "marvin")


def merge_keyword_segments(
    windows: list[dict[str, float | str | int]],
    threshold: float = 0.5,
    min_duration_s: float = 0.05,
    merge_gap_s: float = 0.25,
) -> list[dict[str, float]]:
    """Merge adjacent keyword windows into visualizable time segments."""
    raw_segments = []
    active = None

    for window in windows:
        score = float(window["score"])
        is_keyword = score >= threshold
        if is_keyword and active is None:
            active = {
                "start": float(window["time_start"]),
                "end": float(window["time_end"]),
                "peak_score": score,
            }
        elif is_keyword and active is not None:
            active["end"] = float(window["time_end"])
            active["peak_score"] = max(active["peak_score"], score)
        elif active is not None:
            raw_segments.append(active)
            active = None

    if active is not None:
        raw_segments.append(active)

    merged = []
    for segment in raw_segments:
        if segment["end"] - segment["start"] < min_duration_s:
            continue
        if merged and segment["start"] - merged[-1]["end"] <= merge_gap_s:
            merged[-1]["end"] = segment["end"]
            merged[-1]["peak_score"] = max(
                merged[-1]["peak_score"], segment["peak_score"]
            )
        else:
            merged.append(segment)
    return merged


def pick_keyword_segments(
    windows: list[dict[str, float | str | int]],
    threshold: float = 0.9,
    segment_duration_s: float = 0.8,
    min_peak_distance_s: float = 0.8,
) -> list[dict[str, float]]:
    """Pick local score peaks instead of merging every overlapping hot window.

    The model sees a full 1-second window. A single spoken keyword can therefore
    trigger many neighboring windows. Peak picking gives one visual segment per
    activation burst, which is closer to what you expect to see on the waveform.
    """
    if not windows:
        return []

    scores = np.asarray([float(window["score"]) for window in windows], dtype=np.float32)
    centers = np.asarray(
        [
            (float(window["time_start"]) + float(window["time_end"])) / 2.0
            for window in windows
        ],
        dtype=np.float32,
    )
    duration = max(segment_duration_s, 0.01)
    half_duration = duration / 2.0

    candidates = []
    for index, score in enumerate(scores):
        if score < threshold:
            continue
        left = scores[index - 1] if index > 0 else -np.inf
        right = scores[index + 1] if index + 1 < scores.size else -np.inf
        if score >= left and score >= right:
            candidates.append((float(score), float(centers[index]), index))

    candidates.sort(reverse=True)
    chosen = []
    for score, center, index in candidates:
        if any(abs(center - item["center"]) < min_peak_distance_s for item in chosen):
            continue
        chosen.append(
            {
                "start": max(0.0, center - half_duration),
                "end": center + half_duration,
                "center": center,
                "peak_score": score,
                "peak_window": float(index),
            }
        )

    chosen.sort(key=lambda item: item["start"])
    return chosen


class KeywordSpotter:
    """Load an ONNX model and run file or streaming inference."""

    def __init__(
        self,
        model_path: str | Path = "checkpoint.onnx",
        labels: Iterable[str] = DEFAULT_LABELS,
        threshold: float = 0.5,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for inference. Install it with "
                "`pip install onnxruntime` inside your environment."
            ) from exc

        self.model_path = Path(model_path)
        self.labels = tuple(labels)
        self.threshold = float(threshold)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_spectrogram(self, spectrogram: np.ndarray) -> dict[str, float | str]:
        model_input = to_model_input(spectrogram)
        output = self.session.run([self.output_name], {self.input_name: model_input})[0]
        score = float(np.asarray(output).reshape(-1)[0])
        label = self.labels[1] if score >= self.threshold else self.labels[0]
        return {"label": label, "score": score}

    def predict_audio(self, audio: np.ndarray) -> dict[str, float | str]:
        return self.predict_spectrogram(get_spectrogram(audio))


    def detect_segments(
        self,
        wav_path: str | Path,
        chunk_samples: int = HOP_LENGTH,
        emit_hop_samples: int = HOP_LENGTH,
        min_duration_s: float = 0.05,
        merge_gap_s: float = 0.25,
        method: str = "peaks",
        segment_duration_s: float = 0.8,
        min_peak_distance_s: float = 0.8,
    ) -> dict[str, object]:
        windows = self.stream_file(wav_path, chunk_samples, emit_hop_samples)
        if method == "merge":
            segments = merge_keyword_segments(
                windows,
                threshold=self.threshold,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
            )
        elif method == "peaks":
            segments = pick_keyword_segments(
                windows,
                threshold=self.threshold,
                segment_duration_s=segment_duration_s,
                min_peak_distance_s=min_peak_distance_s,
            )
        else:
            raise ValueError("method must be either 'peaks' or 'merge'.")
        return {"windows": windows, "segments": segments}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run keyword inference on a WAV file.")
    parser.add_argument("wav_path", help="Path to the WAV file to score.")
    parser.add_argument("--model", default="checkpoint.onnx", help="ONNX model path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stream", action="store_true", help="Use streaming windows.")
    args = parser.parse_args()

    spotter = KeywordSpotter(args.model, threshold=args.threshold)
    if args.stream:
        for result in spotter.stream_file(args.wav_path):
            print(result)
    else:
        print(spotter.predict_file(args.wav_path))


if __name__ == "__main__":
    main()
