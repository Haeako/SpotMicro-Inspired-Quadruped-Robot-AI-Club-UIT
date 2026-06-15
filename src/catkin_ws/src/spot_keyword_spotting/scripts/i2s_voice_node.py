#!/usr/bin/env python3
# spot_keyword_spotting/scripts/i2s_voice_node.py

from __future__ import annotations

import errno
import threading
import time
from pathlib import Path

import alsaaudio
import numpy as np
import rospy
import rospkg

from std_msgs.msg import Bool, String

try:
    import kws_native
except ImportError as exc:
    raise RuntimeError(
        "kws_native is not built yet. Run catkin_make/catkin build from the workspace first."
    ) from exc


SAMPLE_RATE = int(kws_native.SAMPLE_RATE)
EXPECTED_SAMPLES = int(kws_native.EXPECTED_SAMPLES)
DEFAULT_LABELS = ("background", "marvin")


def _get_private_param(name, default):
    private_name = name if name.startswith("~") else "~" + name
    return rospy.get_param(private_name, default)


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

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)

        if self.factor == 1:
            return audio.astype(np.float32, copy=False)

        if self._pending.size:
            audio = np.concatenate((self._pending, audio))

        usable = audio.size - (audio.size % self.factor)
        self._pending = audio[usable:].copy()
        if usable <= 0:
            return np.empty(0, dtype=np.float32)

        return audio[:usable].reshape(-1, self.factor).mean(axis=1).astype(np.float32)


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
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, np.dtype("<i2"), 32768.0),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, np.dtype("<i4"), 2147483648.0),
}


class I2SVoiceNode:
    INPUT_SAMPLE_RATE = 48000
    CHANNELS = 2
    PERIOD_SIZE = 128
    SAMPLE_FORMAT = "S32_LE"
    CHANNEL_INDEX = 0
    INFERENCE_RATE = 2.0
    INFER_HOP_SAMPLES = EXPECTED_SAMPLES // 5
    NUM_THREADS = 1
    ACTIVE_STATE = "Idle"
    STATE_TOPIC = "/lcd_state"
    STAND_CMD_TOPIC = "/stand_cmd"
    COMMAND_COOLDOWN = 2.0

    def __init__(self) -> None:
        package_root = Path(rospkg.RosPack().get_path("spot_keyword_spotting"))
        default_model = package_root / "keyword_spotting" / "weights" / "model_int8.tflite"

        self.device = _get_private_param("device", "hw:1,0")
        self.audio_gain = float(_get_private_param("audio_gain", 5.0))
        self.threshold = float(_get_private_param("confidence", 0.7))
        self.target_label = str(_get_private_param("target_label", DEFAULT_LABELS[1]))

        self.input_sample_rate = self.INPUT_SAMPLE_RATE
        self.channels = self.CHANNELS
        self.period_size = self.PERIOD_SIZE
        self.sample_format = self.SAMPLE_FORMAT
        self.channel_index = self.CHANNEL_INDEX
        self.infer_hop_samples = self.INFER_HOP_SAMPLES
        self.inference_rate = self.INFERENCE_RATE

        if self.sample_format not in SAMPLE_FORMATS:
            supported = ", ".join(sorted(SAMPLE_FORMATS))
            raise ValueError("Unsupported sample_format '{}'. Supported values: {}".format(self.sample_format, supported))

        self.alsa_format, self.dtype, self.scale = SAMPLE_FORMATS[self.sample_format]
        model_path = default_model

        self.spotter = KeywordSpotter(
            model_path=model_path,
            labels=DEFAULT_LABELS,
            threshold=self.threshold,
            num_threads=self.NUM_THREADS,
        )
        self.streamer = StreamingSpectrogram(emit_hop_samples=self.infer_hop_samples)
        self.resampler = RatioAveragingResampler(self.input_sample_rate, SAMPLE_RATE)
        self.stand_publisher = rospy.Publisher(self.STAND_CMD_TOPIC, Bool, queue_size=1)
        self.state_subscriber = rospy.Subscriber(self.STATE_TOPIC, String, self.on_state, queue_size=1)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.pcm = None
        self.current_state = self.ACTIVE_STATE
        self.inference_enabled = True
        self.waiting_for_state_exit = False
        self.fixed_channel = self.channel_index if 0 <= self.channel_index < self.channels else None
        self.audio_messages = 0
        self.audio_bytes = 0
        self.alsa_overruns = 0
        self.alsa_errors = 0
        self.inference_count = 0
        self.best_score = 0.0
        self.best_label = "background"
        self.last_inferred_audio_seen = 0
        self.last_command_time = 0.0

        self._open_pcm()
        self.capture_thread = threading.Thread(target=self._capture_loop, name="i2s_capture", daemon=True)
        self.capture_thread.start()
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(0.1, self.inference_rate)),
            self.on_inference_timer,
        )

        rospy.loginfo(
            "I2S voice node started: device=%s, rate=%d, channels=%d, format=%s, period=%d, channel_index=%d",
            self.device,
            self.input_sample_rate,
            self.channels,
            self.sample_format,
            self.period_size,
            self.channel_index,
        )
        rospy.loginfo("Model path: %s", model_path)
        rospy.loginfo(
            "Wake flow: target_label=%s threshold=%.3f audio_gain=%.3f active_state=%s stand_topic=%s",
            self.target_label,
            self.threshold,
            self.audio_gain,
            self.ACTIVE_STATE,
            self.STAND_CMD_TOPIC,
        )

    def on_state(self, msg: String) -> None:
        state = msg.data.strip()
        if self.waiting_for_state_exit and state == self.ACTIVE_STATE:
            self.current_state = state
            self.inference_enabled = False
            return

        if self.waiting_for_state_exit and state != self.ACTIVE_STATE:
            self.waiting_for_state_exit = False

        enabled = state == self.ACTIVE_STATE
        if enabled != self.inference_enabled:
            rospy.loginfo(
                "Voice inference %s because robot state is %s",
                "enabled" if enabled else "paused",
                state,
            )
        self.current_state = state
        self.inference_enabled = enabled

    # FIX 2: PCM_NONBLOCK — at period_size=128 (2.67 ms) the capture thread
    # must service reads extremely fast.  Blocking mode stalls on any hiccup
    # and immediately causes an overrun.  Non-blocking returns length=0 when
    # no data is ready, and we simply sleep 1 ms and retry.
    def _open_pcm(self) -> None:
        self.pcm = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NONBLOCK,
            device=self.device,
        )
        self.pcm.setchannels(self.channels)
        self.pcm.setrate(self.input_sample_rate)
        self.pcm.setformat(self.alsa_format)
        self.pcm.setperiodsize(self.period_size)

    # FIX 3: nullify self.pcm before closing so that a concurrent shutdown()
    # call cannot double-close the same handle (which caused the
    # "Assertion `pcm' failed" crash).
    def _close_pcm_safe(self) -> None:
        pcm, self.pcm = self.pcm, None   # atomic swap in CPython
        if pcm is not None:
            try:
                pcm.close()
            except (AttributeError, alsaaudio.ALSAAudioError):
                pass

    def _recover_pcm(self) -> None:
        self._close_pcm_safe()
        self._open_pcm()

    def _publish_stand_command(self) -> None:
        now = time.time()
        if now - self.last_command_time < self.COMMAND_COOLDOWN:
            rospy.loginfo_throttle(2.0, "Voice stand command is cooling down.")
            return

        self.stand_publisher.publish(Bool(data=True))
        self.last_command_time = now
        self.inference_enabled = False
        self.waiting_for_state_exit = True
        self.current_state = "Voice Stand Commanded"
        rospy.loginfo("Voice command issued: /stand_cmd=True")

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

        if length == 0:
            # FIX 2 (cont.): PCM_NONBLOCK returns 0 when the hardware buffer
            # has no period ready yet.  Sleep briefly to avoid a busy-loop.
            time.sleep(0.001)
            return None

        if length < 0:
            if length == -errno.EPIPE:
                self.alsa_overruns += 1
                rospy.logwarn_throttle(2.0, "ALSA overrun in capture thread (count=%d); recovering", self.alsa_overruns)
            else:
                rospy.logwarn_throttle(2.0, "ALSA read returned length=%d", length)
            try:
                self._recover_pcm()
            except alsaaudio.ALSAAudioError as exc:
                rospy.logwarn_throttle(2.0, "Failed to recover ALSA capture stream: %s", exc)
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

    def on_inference_timer(self, _event=None) -> None:
        if not self.inference_enabled:
            rospy.loginfo_throttle(5.0, "Voice inference paused while robot state is %s", self.current_state)
            return

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
        rospy.loginfo("Detected: %s (%.3f)", label, score)
        if label == self.target_label:
            self._publish_stand_command()

    # FIX 3 (cont.): join the capture thread before closing the PCM handle
    # so the thread cannot call pcm.read() on a handle we just closed.
    def shutdown(self) -> None:
        self._stop_event.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self._close_pcm_safe()


def main() -> None:
    rospy.init_node("spot_keyword_spotting")
    node = I2SVoiceNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
