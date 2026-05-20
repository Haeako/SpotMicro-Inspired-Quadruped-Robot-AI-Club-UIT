#!/usr/bin/env python3
# spot_keyword_spotting/scripts/i2s_voice_node.py

from __future__ import annotations

import errno
import importlib.util
import sys
import threading
import time
import wave
from pathlib import Path

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
        self.period_size = int(_get_private_param(("period_size", "i2s/period_size"), 512))
        self.sample_format = str(_get_private_param(("sample_format", "i2s/sample_format"), "S32_LE")).upper()
        self.channel_index = int(_get_private_param(("channel_index", "i2s/channel_index", "ros/audio_channel_index"), 0))
        self.audio_gain = float(_get_private_param(("audio_gain", "ros/audio_gain"), 1.0))
        self.infer_hop_samples = int(_get_private_param(("infer_hop_samples", "model/infer_hop_samples"), EXPECTED_SAMPLES // 5))
        self.inference_rate = float(_get_private_param(("inference_rate", "model/inference_rate"), 2.0))
        self.threshold = float(_get_private_param(("confidence", "model/confidence"), 0.95))
        self.labels = tuple(_get_private_param(("labels", "model/labels"), DEFAULT_LABELS))
        self.num_threads = int(_get_private_param(("num_threads", "model/num_threads"), 1))
        self.command_topic = _get_private_param(("command_topic", "ros/publish_topic"), "/voice/command")
        self.save_detected_chunks = bool(_get_private_param(("save_detected_chunks", "debug/save_detected_chunks"), False))
        self.detected_chunk_dir = _resolve_package_path(
            package_root,
            _get_private_param(("detected_chunk_dir", "debug/detected_chunk_dir"), "detected_chunks"),
        )

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

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pcm_lock = threading.Lock()
        self.pcm = None
        self.audio_messages = 0
        self.audio_bytes = 0
        self.alsa_overruns = 0
        self.alsa_errors = 0
        self.inference_count = 0
        self.best_score = 0.0
        self.best_label = "background"
        self.last_inferred_audio_seen = 0
        self.saved_detected_chunks = 0

        if self.save_detected_chunks:
            self.detected_chunk_dir.mkdir(parents=True, exist_ok=True)

        self._open_pcm()
        self.capture_thread = threading.Thread(target=self._capture_loop, name="i2s_capture", daemon=True)
        self.capture_thread.start()
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(0.1, self.inference_rate)),
            self.on_inference_timer,
        )

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
        rospy.loginfo(
            "Threshold: %.3f, expected samples: %d, inference hop samples: %d, inference rate: %.3f Hz",
            self.threshold,
            EXPECTED_SAMPLES,
            self.infer_hop_samples,
            self.inference_rate,
        )
        rospy.loginfo("Audio gain: %.3f, save detected chunks: %s", self.audio_gain, self.save_detected_chunks)
        rospy.loginfo("Publishing command topic only: %s", self.command_topic)

    def _open_pcm(self) -> None:
        with self._pcm_lock:
            self.pcm = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
            self.pcm.setchannels(self.channels)
            self.pcm.setrate(self.input_sample_rate)
            self.pcm.setformat(self.alsa_format)
            self.pcm.setperiodsize(self.period_size)

    def _recover_pcm(self) -> None:
        with self._pcm_lock:
            try:
                if self.pcm is not None:
                    self.pcm.close()
            except (AttributeError, alsaaudio.ALSAAudioError):
                pass

            self.pcm = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
            self.pcm.setchannels(self.channels)
            self.pcm.setrate(self.input_sample_rate)
            self.pcm.setformat(self.alsa_format)
            self.pcm.setperiodsize(self.period_size)

    def _write_wav(self, path: Path, audio: np.ndarray, sample_rate: int) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sample_rate))
            wav.writeframes(pcm.tobytes())

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
        try:
            with self._pcm_lock:
                length, data = self.pcm.read()
        except alsaaudio.ALSAAudioError as exc:
            self.alsa_errors += 1
            rospy.logwarn_throttle(2.0, "ALSA read failed in capture thread: %s (count=%d)", exc, self.alsa_errors)
            try:
                self._recover_pcm()
            except alsaaudio.ALSAAudioError as recover_exc:
                rospy.logwarn_throttle(2.0, "Failed to recover ALSA capture stream: %s", recover_exc)
            return None

        if length <= 0:
            if length == -errno.EPIPE:
                self.alsa_overruns += 1
                rospy.logwarn_throttle(2.0, "ALSA overrun in capture thread (count=%d); recovering", self.alsa_overruns)
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
        self.audio_messages += 1
        self.audio_bytes += len(data)
        rospy.loginfo_throttle(
            5.0,
            "Capture alive: messages=%d bytes=%d frames=%d selected_channel=%d peak=%.6f rms=%.6f",
            self.audio_messages,
            self.audio_bytes,
            length,
            selected_channel,
            float(np.max(np.abs(mono))) if mono.size else 0.0,
            float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0,
        )
        return mono

    def _capture_loop(self) -> None:
        while not rospy.is_shutdown() and not self._stop_event.is_set():
            raw_48k = self._read_mono()
            if raw_48k is None:
                continue

            audio_16k = self.resampler.process(raw_48k)
            if audio_16k.size == 0:
                continue

            with self._lock:
                self.streamer.push_audio(audio_16k)

            rospy.loginfo_throttle(
                5.0,
                "Capture resampler: in_total=%d out_total=%d latest_in=%d latest_out=%d pending=%d",
                self.resampler.input_count,
                self.resampler.output_count,
                raw_48k.size,
                audio_16k.size,
                self.resampler.pending_samples,
            )

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

        rospy.loginfo_throttle(
            2.0,
            "Inference alive: count=%d, label=%s, score=%.3f, best_label=%s, best_score=%.3f, threshold=%.3f",
            self.inference_count,
            label,
            score,
            self.best_label,
            self.best_score,
            self.spotter.threshold,
        )

        if score < self.spotter.threshold:
            return

        command = VoiceCommand()
        command.command = label
        command.confidence = score
        self.publisher.publish(command)
        self._save_detected_chunk(audio, label, score)
        rospy.loginfo("Detected: %s (%.3f)", label, score)

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            if self.pcm is not None:
                self.pcm.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass


def main() -> None:
    rospy.init_node("spot_keyword_spotting")
    node = DirectI2SVoiceNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()