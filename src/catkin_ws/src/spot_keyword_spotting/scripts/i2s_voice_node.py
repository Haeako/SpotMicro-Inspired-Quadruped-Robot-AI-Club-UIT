#!/usr/bin/env python3
# spot_keyword_spotting/scripts/i2s_voice_node.py

from __future__ import annotations

import errno
import threading
import time
import wave
from pathlib import Path

import alsaaudio
import numpy as np
import roslaunch
import rospy
import rospkg

from std_msgs.msg import Bool

try:
    import kws_native
except ImportError as exc:
    raise RuntimeError(
        "kws_native is not built yet. Run catkin_make/catkin build from the workspace first."
    ) from exc


SAMPLE_RATE = int(kws_native.SAMPLE_RATE)
EXPECTED_SAMPLES = int(kws_native.EXPECTED_SAMPLES)
DEFAULT_LABELS = ("background", "marvin")


def _get_private_param(names, default):
    if isinstance(names, str):
        names = (names,)

    for name in names:
        private_name = name if name.startswith("~") else "~" + name
        if rospy.has_param(private_name):
            return rospy.get_param(private_name)

    return default


def _resolve_package_path(package_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return package_root / path


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


def get_spectrogram(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return kws_native.get_spectrogram(audio, True)


def to_model_input(spectrogram: np.ndarray) -> np.ndarray:
    spectrogram = np.asarray(spectrogram, dtype=np.float32)
    return kws_native.to_model_input(spectrogram)


class StreamingSpectrogram:
    def __init__(self, window_samples: int = EXPECTED_SAMPLES, emit_hop_samples: int = EXPECTED_SAMPLES // 2) -> None:
        self.emit_hop_samples = int(emit_hop_samples)
        self.ring = kws_native.RingBuffer(int(window_samples))

    @property
    def ready(self) -> bool:
        return bool(self.ring.is_full())

    def push_audio(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self.ring.push(samples)

    def current_audio(self) -> np.ndarray:
        return self.ring.current_audio()


class RatioAveragingResampler:
    def __init__(self, input_sample_rate: int, output_sample_rate: int) -> None:
        self.input_sample_rate = int(input_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        if self.input_sample_rate <= 0 or self.output_sample_rate <= 0:
            raise ValueError("Sample rates must be positive")
        if self.input_sample_rate % self.output_sample_rate != 0:
            raise ValueError("RatioAveragingResampler requires an integer sample-rate ratio")
        self.factor = self.input_sample_rate // self.output_sample_rate
        self._pending = np.empty(0, dtype=np.float32)
        self._input_count = 0
        self._output_count = 0

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)

        self._input_count += int(audio.size)
        if self.factor == 1:
            self._output_count += int(audio.size)
            return audio.astype(np.float32, copy=False)

        if self._pending.size:
            audio = np.concatenate((self._pending, audio))

        usable = audio.size - (audio.size % self.factor)
        self._pending = audio[usable:].copy()
        if usable <= 0:
            return np.empty(0, dtype=np.float32)

        out = audio[:usable].reshape(-1, self.factor).mean(axis=1).astype(np.float32)
        self._output_count += int(out.size)
        return out

    @property
    def pending_samples(self) -> int:
        return int(self._pending.size)

    @property
    def input_count(self) -> int:
        return int(self._input_count)

    @property
    def output_count(self) -> int:
        return int(self._output_count)


class KeywordSpotter:
    def __init__(self, model_path: str | Path, labels=DEFAULT_LABELS, threshold: float = 0.5, num_threads: int = 1) -> None:
        self.model_path = Path(model_path)
        self.labels = tuple(labels)
        self.threshold = float(threshold)
        Interpreter = _load_tflite_interpreter()
        self.interpreter = Interpreter(model_path=str(self.model_path), num_threads=int(num_threads))
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

        scale, zero_point = self.input_details[0].get("quantization", (0.0, 0))
        if scale and scale > 0:
            model_input = np.round(model_input / scale + zero_point)
        return model_input.astype(self.input_dtype)

    def _read_score(self) -> float:
        output = np.asarray(self.interpreter.get_tensor(self.output_index))
        if output.dtype != np.float32:
            scale, zero_point = self.output_details[0].get("quantization", (0.0, 0))
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

SAMPLE_FORMATS = {
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, 2, np.dtype("<i2"), 32768.0),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, 4, np.dtype("<i4"), 2147483648.0),
}


class I2SVoiceNode:
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
        self.capture_stats = bool(_get_private_param(("capture_stats", "debug/capture_stats"), False))
        self.resampler_stats = bool(_get_private_param(("resampler_stats", "debug/resampler_stats"), False))
        self.infer_hop_samples = int(_get_private_param(("infer_hop_samples", "model/infer_hop_samples"), EXPECTED_SAMPLES // 5))
        self.inference_rate = float(_get_private_param(("inference_rate", "model/inference_rate"), 2.0))
        self.threshold = float(_get_private_param(("confidence", "model/confidence"), 0.95))
        self.labels = tuple(_get_private_param(("labels", "model/labels"), DEFAULT_LABELS))
        self.num_threads = int(_get_private_param(("num_threads", "model/num_threads"), 1))        
        self.launch_on_detect = bool(_get_private_param(("launch_on_detect", "launch/enabled"), True))
        self.launch_once = bool(_get_private_param(("launch_once", "launch/once"), True))
        self.launch_cooldown = float(_get_private_param(("launch_cooldown", "launch/cooldown"), 10.0))
        self.launch_target_label = str(_get_private_param(("launch_target_label", "launch/target_label"), "marvin"))
        self.launch_package = str(_get_private_param(("launch_package", "launch/package"), "spot_micro_joy"))
        self.launch_file = str(_get_private_param(("launch_file", "launch/file"), "everything.launch"))
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
        self.resampler = RatioAveragingResampler(self.input_sample_rate, SAMPLE_RATE)
        self.publisher = rospy.Publisher("/voice_cmd", Bool, queue_size=10)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.pcm = None
        self.fixed_channel = self.channel_index if 0 <= self.channel_index < self.channels else None
        self.audio_messages = 0
        self.audio_bytes = 0
        self.alsa_overruns = 0
        self.alsa_errors = 0
        self.inference_count = 0
        self.best_score = 0.0
        self.best_label = "background"
        self.last_inferred_audio_seen = 0
        self.saved_detected_chunks = 0
        self.launch_parent = None
        self.launch_started = False
        self.last_launch_time = 0.0

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
        rospy.loginfo(
            "Publishing command topic: %s, launch_on_detect=%s target_label=%s launch=%s/%s",
            self.publisher.name,
            self.launch_on_detect,
            self.launch_target_label,
            self.launch_package,
            self.launch_file,
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

    def _launch_everything(self) -> None:
        if not self.launch_on_detect:
            return
        if self.launch_once and self.launch_started:
            rospy.loginfo_throttle(5.0, "Launch already started; ignoring repeated detection.")
            return

        now = time.time()
        if now - self.last_launch_time < self.launch_cooldown:
            rospy.loginfo_throttle(2.0, "Launch trigger is cooling down.")
            return

        launch_path = Path(rospkg.RosPack().get_path(self.launch_package)) / "launch" / self.launch_file
        if not launch_path.exists():
            rospy.logerr("Launch file not found: %s", launch_path)
            self.last_launch_time = now
            return

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch_parent = roslaunch.parent.ROSLaunchParent(uuid, [str(launch_path)])
        self.launch_parent.start()
        self.launch_started = True
        self.last_launch_time = now
        rospy.loginfo("Started launch file after voice detection: %s", launch_path)

    def _read_mono(self) -> np.ndarray | None:
        try:
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

        samples = np.frombuffer(data, dtype=self.dtype)
        expected_values = int(length) * self.channels
        if samples.size < expected_values:
            rospy.logwarn_throttle(2.0, "Short ALSA block: expected %d values, got %d", expected_values, samples.size)
            return None

        samples = samples[:expected_values].reshape(int(length), self.channels)
        if self.fixed_channel is not None:
            selected_channel = self.fixed_channel
        else:
            float_samples = samples.astype(np.float32)
            rms_by_channel = np.sqrt(np.mean(float_samples * float_samples, axis=0))
            selected_channel = int(np.argmax(rms_by_channel))

        mono = samples[:, selected_channel].astype(np.float32) / self.scale
        if self.audio_gain != 1.0:
            mono *= self.audio_gain
            np.clip(mono, -1.0, 1.0, out=mono)

        self.audio_messages += 1
        self.audio_bytes += len(data)
        if self.capture_stats:
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

            if self.resampler_stats:
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

        spectrogram = get_spectrogram(audio)

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
        self.publisher.publish(True)
        rospy.loginfo("Detected: %s (%.3f)", label, score)
        self._save_detected_chunk(audio, label, score)
        if label == self.launch_target_label:
            self._launch_everything()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self.launch_parent is not None:
            try:
                self.launch_parent.shutdown()
            except Exception as exc:
                rospy.logwarn("Failed to shutdown launched processes cleanly: %s", exc)
        try:
            if self.pcm is not None:
                self.pcm.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass


def main() -> None:
    rospy.init_node("spot_keyword_spotting")
    node = I2SVoiceNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
