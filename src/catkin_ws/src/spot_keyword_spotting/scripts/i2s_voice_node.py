#!/usr/bin/env python3
# spot_keyword_spotting/scripts/i2s_voice_node.py

from __future__ import annotations

import errno
import importlib.util
import threading
import sys
import time
import wave
from pathlib import Path
from typing import Iterable

import alsaaudio
import numpy as np
import rospy
import rospkg

from spot_keyword_spotting.msg import VoiceCommand
def _load_voice_node_helpers():
    script_path = Path(__file__).resolve().parent / "voice_node.py"
    spec = importlib.util.spec_from_file_location("spot_kws_voice_node_source", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load voice_node.py from {}".format(script_path))

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_voice_node = _load_voice_node_helpers()
DEFAULT_LABELS = _voice_node.DEFAULT_LABELS
EXPECTED_SAMPLES = _voice_node.EXPECTED_SAMPLES
SAMPLE_RATE = _voice_node.SAMPLE_RATE
FixedRatioAveragingResampler = _voice_node.FixedRatioAveragingResampler
KeywordSpotter = _voice_node.KeywordSpotter
StreamingSpectrogram = _voice_node.StreamingSpectrogram
_get_private_param = _voice_node._get_private_param
_resolve_package_path = _voice_node._resolve_package_path


SAMPLE_FORMATS = {
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, 2, np.dtype("<i2"), 32768.0),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, 4, np.dtype("<i4"), 2147483648.0),
}


class DirectI2SVoiceNode:
    def __init__(self) -> None:
        package_root = Path(rospkg.RosPack().get_path("spot_keyword_spotting"))
        default_model = package_root / "keyword_spotting" / "weights" / "checkpoint.tflite"

        self.device = _get_private_param(("device", "i2s/device"), "hw:1,0")
        self.input_sample_rate = int(_get_private_param(("input_sample_rate", "i2s/sample_rate", "ros/input_sample_rate"), 48000))
        self.channels = int(_get_private_param(("channels", "i2s/channels", "ros/audio_channels"), 2))
        self.period_size = int(_get_private_param(("period_size", "i2s/period_size"), 256))
        self.sample_format = str(_get_private_param(("sample_format", "i2s/sample_format"), "S32_LE")).upper()
        self.channel_index = int(_get_private_param(("channel_index", "i2s/channel_index", "ros/audio_channel_index"), 0))
        self.audio_gain = float(_get_private_param(("audio_gain", "ros/audio_gain"), 1.0))
        self.infer_hop_samples = int(_get_private_param(("infer_hop_samples", "model/infer_hop_samples"), EXPECTED_SAMPLES // 5))
        self.threshold = float(_get_private_param(("confidence", "model/confidence"), 0.95))
        self.labels = tuple(_get_private_param(("labels", "model/labels"), DEFAULT_LABELS))
        self.num_threads = int(_get_private_param(("num_threads", "model/num_threads"), 2))
        self.inference_rate = float(_get_private_param(("inference_rate", "model/inference_rate"), 5.0))
        self.publish_inference = bool(_get_private_param(("publish_inference", "ros/publish_inference"), True))
        self.command_topic = _get_private_param(("command_topic", "ros/publish_topic"), "/voice/command")
        self.inference_topic = _get_private_param(("inference_topic", "ros/inference_topic"), "/voice/inference")
        self.record_audio_chunks = bool(_get_private_param(("record_audio_chunks", "debug/record_audio_chunks"), False))
        self.record_raw_audio_chunks = bool(_get_private_param(("record_raw_audio_chunks", "debug/record_raw_audio_chunks"), False))
        self.audio_chunk_seconds = float(_get_private_param(("audio_chunk_seconds", "debug/audio_chunk_seconds"), 5.0))
        self.raw_audio_chunk_seconds = float(_get_private_param(("raw_audio_chunk_seconds", "debug/raw_audio_chunk_seconds"), self.audio_chunk_seconds))
        self.audio_chunk_dir = _resolve_package_path(package_root, _get_private_param(("audio_chunk_dir", "debug/audio_chunk_dir"), "audio_chunks"))
        self.raw_audio_chunk_dir = _resolve_package_path(package_root, _get_private_param(("raw_audio_chunk_dir", "debug/raw_audio_chunk_dir"), "raw_audio_chunks"))
        self.detected_chunk_dir = _resolve_package_path(package_root, _get_private_param(("detected_chunk_dir", "debug/detected_chunk_dir"), "detected_chunks"))
        self.save_detected_chunks = bool(_get_private_param(("save_detected_chunks", "debug/save_detected_chunks"), True))

        if self.sample_format not in SAMPLE_FORMATS:
            supported = ", ".join(sorted(SAMPLE_FORMATS))
            raise ValueError("Unsupported sample_format '{}'. Supported values: {}".format(self.sample_format, supported))

        self.alsa_format, self.bytes_per_sample, self.dtype, self.scale = SAMPLE_FORMATS[self.sample_format]
        model_path = _resolve_package_path(package_root, _get_private_param(("model_path", "model/path"), str(default_model)))

        self.spotter = KeywordSpotter(
            model_path=model_path,
            labels=self.labels,
            threshold=self.threshold,
            num_threads=self.num_threads,
        )
        self.streamer = StreamingSpectrogram(emit_hop_samples=self.infer_hop_samples)
        self.resampler = FixedRatioAveragingResampler(self.input_sample_rate, SAMPLE_RATE)
        self.publisher = rospy.Publisher(self.command_topic, VoiceCommand, queue_size=10)
        self.inference_publisher = None
        if self.publish_inference:
            self.inference_publisher = rospy.Publisher(self.inference_topic, VoiceCommand, queue_size=10)

        self.audio_chunk_samples = max(1, int(self.audio_chunk_seconds * SAMPLE_RATE))
        self.raw_audio_chunk_samples = max(1, int(self.raw_audio_chunk_seconds * self.input_sample_rate))
        self.audio_chunk_buffer: list[np.ndarray] = []
        self.raw_audio_chunk_buffer: list[np.ndarray] = []
        self.audio_chunk_count = 0
        self.raw_audio_chunk_count = 0
        self.saved_audio_chunks = 0
        self.saved_raw_chunks = 0
        self.saved_detected_chunks = 0
        self.inference_count = 0
        self.best_score = 0.0
        self.best_label = "background"
        self.last_inferred_audio_seen = 0
        self.alsa_overruns = 0
        self._lock = threading.Lock()

        for directory, enabled in (
            (self.audio_chunk_dir, self.record_audio_chunks),
            (self.raw_audio_chunk_dir, self.record_raw_audio_chunks),
            (self.detected_chunk_dir, self.save_detected_chunks),
        ):
            if enabled:
                directory.mkdir(parents=True, exist_ok=True)

        self.pcm = None
        self._open_pcm()

        rospy.loginfo(
            "Direct I2S KWS started: device=%s, rate=%d, channels=%d, format=%s, period=%d, channel_index=%d",
            self.device,
            self.input_sample_rate,
            self.channels,
            self.sample_format,
            self.period_size,
            self.channel_index,
        )
        rospy.loginfo("Model path: %s", model_path)
        rospy.loginfo("Threshold: %.3f, expected samples: %d, inference hop samples: %d", self.threshold, EXPECTED_SAMPLES, self.infer_hop_samples)
        rospy.loginfo("Audio gain: %.3f", self.audio_gain)
        rospy.loginfo("Inference timer rate: %.3f Hz", self.inference_rate)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(0.1, self.inference_rate)),
            self.on_inference_timer,
        )

    def _open_pcm(self) -> None:
        self.pcm = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
        self.pcm.setchannels(self.channels)
        self.pcm.setrate(self.input_sample_rate)
        self.pcm.setformat(self.alsa_format)
        self.pcm.setperiodsize(self.period_size)

    def _recover_pcm(self) -> None:
        try:
            if self.pcm is not None:
                self.pcm.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass

        self._open_pcm()
    def _write_wav(self, path: Path, audio: np.ndarray, sample_rate: int) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sample_rate))
            wav.writeframes(pcm.tobytes())

    def _record_chunk(self, audio: np.ndarray, sample_rate: int, is_raw: bool) -> None:
        if audio.size == 0:
            return

        if is_raw:
            if not self.record_raw_audio_chunks:
                return
            buffer = self.raw_audio_chunk_buffer
            target = self.raw_audio_chunk_samples
            directory = self.raw_audio_chunk_dir
            count_attr = "raw_audio_chunk_count"
            saved_attr = "saved_raw_chunks"
            tag = "direct_raw"
        else:
            if not self.record_audio_chunks:
                return
            buffer = self.audio_chunk_buffer
            target = self.audio_chunk_samples
            directory = self.audio_chunk_dir
            count_attr = "audio_chunk_count"
            saved_attr = "saved_audio_chunks"
            tag = "direct_stream"

        buffer.append(audio.copy())
        setattr(self, count_attr, getattr(self, count_attr) + int(audio.size))

        while getattr(self, count_attr) >= target:
            merged = np.concatenate(buffer)
            chunk = merged[:target]
            remaining = merged[target:]
            buffer[:] = [remaining] if remaining.size else []
            setattr(self, count_attr, int(remaining.size))
            setattr(self, saved_attr, getattr(self, saved_attr) + 1)
            filename = "{stamp}_{tag}_{rate}hz_{idx:04d}.wav".format(
                stamp=time.strftime("%Y%m%d_%H%M%S"),
                tag=tag,
                rate=sample_rate,
                idx=getattr(self, saved_attr),
            )
            path = directory / filename
            self._write_wav(path, chunk, sample_rate)
            peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            rospy.loginfo("Saved %s chunk: %s (peak=%.6f, rms=%.6f)", tag, path, peak, rms)

    def _save_detected_chunk(self, audio: np.ndarray, label: str, score: float) -> None:
        if not self.save_detected_chunks:
            return
        self.saved_detected_chunks += 1
        path = self.detected_chunk_dir / "{stamp}_{idx:04d}_{label}_{score:.3f}.wav".format(
            stamp=time.strftime("%Y%m%d_%H%M%S"),
            idx=self.saved_detected_chunks,
            label=label,
            score=score,
        )
        self._write_wav(path, audio, SAMPLE_RATE)
        rospy.loginfo("Saved detected chunk: %s", path)

    def _read_mono(self) -> np.ndarray | None:
        length, data = self.pcm.read()
        if length <= 0:
            if length == -errno.EPIPE:
                self.alsa_overruns += 1
                rospy.logwarn_throttle(
                    2.0,
                    "ALSA overrun while reading I2S (length=%d, count=%d); recovering capture stream. Try _period_size:=1024 or 2048 if this keeps happening.",
                    length,
                    self.alsa_overruns,
                )
                try:
                    self._recover_pcm()
                except alsaaudio.ALSAAudioError as exc:
                    rospy.logwarn_throttle(2.0, "Failed to recover ALSA capture stream: %s", exc)
                return None

            rospy.logwarn_throttle(2.0, "ALSA read returned length=%d", length)
            return None

        samples = np.frombuffer(data, dtype=self.dtype).astype(np.float32)
        expected_values = int(length) * self.channels
        if samples.size < expected_values:
            rospy.logwarn_throttle(2.0, "Short ALSA block: expected %d values, got %d", expected_values, samples.size)
            return None

        samples = samples[:expected_values].reshape(int(length), self.channels)
        if 0 <= self.channel_index < self.channels:
            selected_channel = self.channel_index
        else:
            rms_by_channel = np.sqrt(np.mean(samples * samples, axis=0))
            selected_channel = int(np.argmax(rms_by_channel))

        mono = samples[:, selected_channel] / self.scale
        mono = np.clip(mono * self.audio_gain, -1.0, 1.0).astype(np.float32)
        rospy.loginfo_throttle(
            5.0,
            "Direct audio alive: frames=%d bytes=%d selected_channel=%d peak=%.6f rms=%.6f",
            length,
            len(data),
            selected_channel,
            float(np.max(np.abs(mono))) if mono.size else 0.0,
            float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0,
        )
        return mono

    def on_inference_timer(self, _event=None) -> None:
        with self._lock:
            if not self.streamer.ready:
                return

            samples_seen = self.streamer.ring.samples_seen()
            new_samples = samples_seen - self.last_inferred_audio_seen
            if new_samples < self.infer_hop_samples:
                return

            self.last_inferred_audio_seen = samples_seen
            audio = self.streamer.current_audio()
            spectrogram = self.streamer.current_spectrogram()

        result = self.spotter.predict_spectrogram(spectrogram)
        label = str(result["label"])
        score = float(result["score"])
        self.inference_count += 1

        if score >= self.best_score:
            self.best_score = score
            self.best_label = label

        if self.inference_publisher is not None:
            msg = VoiceCommand()
            msg.command = label
            msg.confidence = score
            self.inference_publisher.publish(msg)

        rospy.loginfo_throttle(
            2.0,
            "Direct inference alive: count=%d, label=%s, score=%.3f, best_label=%s, best_score=%.3f, threshold=%.3f",
            self.inference_count,
            label,
            score,
            self.best_label,
            self.best_score,
            self.spotter.threshold,
        )

        if score < self.spotter.threshold:
            return

        msg = VoiceCommand()
        msg.command = label
        msg.confidence = score
        self.publisher.publish(msg)
        self._save_detected_chunk(audio, label, score)
        rospy.loginfo("Direct detected: %s (%.3f)", label, score)

    def run(self) -> None:
        while not rospy.is_shutdown():
            raw_48k = self._read_mono()
            if raw_48k is None:
                continue

            self._record_chunk(raw_48k, self.input_sample_rate, is_raw=True)
            audio_16k = self.resampler.process(raw_48k)
            self._record_chunk(audio_16k, SAMPLE_RATE, is_raw=False)
            with self._lock:
                self.streamer.push_audio(audio_16k)
            rospy.loginfo_throttle(
                5.0,
                "Direct resampler: in_total=%d out_total=%d latest_in=%d latest_out=%d pending=%d",
                self.resampler.input_count,
                self.resampler.output_count,
                raw_48k.size,
                audio_16k.size,
                self.resampler.pending_samples,
            )


def main() -> None:
    rospy.init_node("spot_keyword_spotting")
    DirectI2SVoiceNode().run()


if __name__ == "__main__":
    main()