#!/usr/bin/env python3

import argparse
import importlib.util
import wave
from pathlib import Path

import numpy as np
import rospkg

SAMPLE_RATE = 16000
def parse_args():
    parser = argparse.ArgumentParser(description="Run keyword spotting inference on a WAV file.")
    parser.add_argument("wav_path")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--window-hop", type=float, default=0.2)
    parser.add_argument("--channel", type=int, default=0)
    return parser.parse_args()


def load_voice_node(package_root):
    source_path = package_root / "scripts" / "voice_node.py"
    spec = importlib.util.spec_from_file_location("spot_kws_voice_node_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_wav(path, channel):
    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width: {} byte(s)".format(sample_width))

    if channels > 1:
        audio = audio.reshape(-1, channels)[:, channel]

    if sample_rate != SAMPLE_RATE:
        duration = audio.size / float(sample_rate)
        output_size = max(1, int(round(duration * SAMPLE_RATE)))
        source_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        target_x = np.linspace(0.0, duration, num=output_size, endpoint=False)
        audio = np.interp(target_x, source_x, audio).astype(np.float32)

    return audio


def main():
    args = parse_args()
    package_root = Path(rospkg.RosPack().get_path("spot_keyword_spotting"))
    voice_node = load_voice_node(package_root)
    model_path = args.model_path

    if model_path is None:
        model_path = package_root / "keyword_spotting" / "weights" / "checkpoint.tflite"

    audio = read_wav(args.wav_path, args.channel)
    spotter = voice_node.KeywordSpotter(model_path, threshold=args.threshold)
    hop = max(1, int(args.window_hop * voice_node.SAMPLE_RATE))

    best_score = -1.0
    best_time = 0.0
    best_label = "background"

    if audio.size < voice_node.EXPECTED_SAMPLES:
        audio = np.pad(audio, (0, voice_node.EXPECTED_SAMPLES - audio.size))

    for start in range(0, audio.size - voice_node.EXPECTED_SAMPLES + 1, hop):
        window = audio[start:start + voice_node.EXPECTED_SAMPLES]
        result = spotter.predict_spectrogram(voice_node.get_spectrogram(window))
        score = float(result["score"])

        if score > best_score:
            best_score = score
            best_time = start / float(voice_node.SAMPLE_RATE)
            best_label = str(result["label"])

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    print(
        "wav={}, samples={}, peak={:.6f}, rms={:.6f}, best_label={}, best_score={:.6f}, best_time={:.2f}s".format(
            args.wav_path,
            audio.size,
            peak,
            rms,
            best_label,
            best_score,
            best_time,
        )
    )


if __name__ == "__main__":
    main()
