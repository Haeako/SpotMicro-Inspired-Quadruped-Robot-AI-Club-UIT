#!/usr/bin/env python3
# spot_keyword_spotting/scripts/voice_node.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rospy
import rospkg

from std_msgs.msg import UInt8MultiArray
from spot_keyword_spotting.msg import VoiceCommand

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


def _get_private_param(names: str | Iterable[str], default):
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


def get_spectrogram(audio: np.ndarray, normalize: bool = True) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return kws_native.get_spectrogram(audio, normalize)


def to_model_input(spectrogram: np.ndarray) -> np.ndarray:
    spectrogram = np.asarray(spectrogram, dtype=np.float32)
    return kws_native.to_model_input(spectrogram)


class StreamingSpectrogram:
    def __init__(
        self,
        window_samples: int = EXPECTED_SAMPLES,
        emit_hop_samples: int = EXPECTED_SAMPLES // 2,
    ) -> None:
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
            part = samples[start:start + self.emit_hop_samples]

            if part.size == 0:
                continue

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


class KeywordSpotter:
    def __init__(
        self,
        model_path: str | Path,
        labels: Iterable[str] = DEFAULT_LABELS,
        threshold: float = 0.5,
        num_threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = tuple(labels)
        self.threshold = float(threshold)

        Interpreter = _load_tflite_interpreter()

        self.interpreter = Interpreter(
            model_path=str(self.model_path),
            num_threads=int(num_threads),
        )

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
        output = self.interpreter.get_tensor(self.output_index)
        output = np.asarray(output)

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

        return {
            "label": label,
            "score": score,
        }


class VoiceNode:
    def __init__(self) -> None:
        package_root = Path(
            rospkg.RosPack().get_path("spot_keyword_spotting")
        )

        default_model = (
            package_root
            / "keyword_spotting"
            / "weights"
            / "checkpoint.tflite"
        )

        model_path = _resolve_package_path(
            package_root,
            _get_private_param(("model_path", "model/path"), str(default_model)),
        )

        threshold = float(
            _get_private_param(("threshold", "model/confidence"), 0.95)
        )

        labels = _get_private_param(
            ("labels", "model/labels"),
            ["background", "marvin"]
        )

        num_threads = int(
            _get_private_param("num_threads", 2)
        )

        infer_hop_samples = int(
            _get_private_param(("infer_hop_samples", "model/infer_hop_samples"), EXPECTED_SAMPLES // 2)
        )

        fallback_audio_channels = int(
            _get_private_param(("audio_channels", "ros/audio_channels"), 1)
        )

        fallback_sample_width_bytes = int(
            _get_private_param(("sample_width_bytes", "ros/sample_width_bytes"), 4)
        )

        audio_channel_index = int(
            _get_private_param(("audio_channel_index", "ros/audio_channel_index"), -1)
        )

        audio_topic = _get_private_param(
            ("audio_topic", "ros/audio_topic"),
            "/audio/raw"
        )

        command_topic = _get_private_param(
            ("command_topic", "ros/publish_topic"),
            "/voice/command"
        )

        inference_topic = _get_private_param(
            ("inference_topic", "ros/inference_topic"),
            "/voice/inference"
        )

        publish_inference = bool(
            _get_private_param(("publish_inference", "ros/publish_inference"), True)
        )

        self.spotter = KeywordSpotter(
            model_path=model_path,
            labels=labels,
            threshold=threshold,
            num_threads=num_threads,
        )

        self.streamer = StreamingSpectrogram(
            emit_hop_samples=infer_hop_samples,
        )
        self.fallback_audio_channels = max(1, fallback_audio_channels)
        self.fallback_sample_width_bytes = fallback_sample_width_bytes
        self.audio_channel_index = audio_channel_index
        self._audio_messages = 0
        self._audio_bytes = 0
        self._inference_count = 0

        self.publisher = rospy.Publisher(
            command_topic,
            VoiceCommand,
            queue_size=10,
        )

        self.inference_publisher = None

        if publish_inference:
            self.inference_publisher = rospy.Publisher(
                inference_topic,
                VoiceCommand,
                queue_size=10,
            )

        self.subscriber = rospy.Subscriber(
            audio_topic,
            UInt8MultiArray,
            self.on_audio,
            queue_size=1,
            buff_size=2**16,
        )

        rospy.loginfo("Keyword spotting node started")
        rospy.loginfo("Subscribing audio topic: %s", audio_topic)
        rospy.loginfo("Publishing command topic: %s", command_topic)
        if publish_inference:
            rospy.loginfo("Publishing inference topic: %s", inference_topic)
        rospy.loginfo("Model path: %s", model_path)
        rospy.loginfo(
            "Threshold: %.3f, expected samples: %d, inference hop samples: %d",
            self.spotter.threshold,
            EXPECTED_SAMPLES,
            self.streamer.emit_hop_samples,
        )

    def _layout_dim_size(self, msg: UInt8MultiArray, label: str) -> int | None:
        for dim in msg.layout.dim:
            if dim.label == label:
                return int(dim.size)

        return None

    def on_audio(self, msg: UInt8MultiArray) -> None:
        raw = np.frombuffer(
            bytearray(msg.data),
            dtype=np.uint8,
        )

        self._audio_messages += 1
        self._audio_bytes += int(raw.size)
        rospy.loginfo_throttle(
            5.0,
            "Audio input alive: messages=%d, total_bytes=%d, latest_bytes=%d",
            self._audio_messages,
            self._audio_bytes,
            raw.size,
        )

        if raw.size < 2:
            rospy.logwarn_throttle(
                5.0,
                "Ignoring too-short audio packet: %d byte(s)",
                raw.size,
            )
            return

        sample_width_bytes = self._layout_dim_size(msg, "sample_width_bytes")

        if sample_width_bytes is None:
            sample_width_bytes = self.fallback_sample_width_bytes

        if sample_width_bytes == 4:
            dtype = np.dtype("<i4")
            scale = 2147483648.0
        elif sample_width_bytes == 2:
            dtype = np.dtype("<i2")
            scale = 32768.0
        else:
            rospy.logwarn_throttle(
                5.0,
                "Unsupported audio sample width: %d byte(s)",
                sample_width_bytes,
            )
            return

        if raw.size % sample_width_bytes != 0:
            rospy.logwarn_throttle(
                5.0,
                "Audio packet byte count %d is not divisible by sample width %d; trimming packet",
                raw.size,
                sample_width_bytes,
            )
            raw = raw[: raw.size - (raw.size % sample_width_bytes)]

        audio = raw.view(dtype).astype(np.float32)

        channels = self._layout_dim_size(msg, "channels")

        if channels is None or channels <= 0:
            channels = self.fallback_audio_channels

        if channels > 1:
            if audio.size % channels != 0:
                rospy.logwarn_throttle(
                    5.0,
                    "Audio sample count %d is not divisible by channel count %d; trimming packet",
                    audio.size,
                    channels,
                )
                audio = audio[: audio.size - (audio.size % channels)]

            audio_by_channel = audio.reshape(-1, channels)
            channel_rms = np.sqrt(np.mean(audio_by_channel * audio_by_channel, axis=0))

            if 0 <= self.audio_channel_index < channels:
                selected_channel = self.audio_channel_index
            else:
                selected_channel = int(np.argmax(channel_rms))

            audio = audio_by_channel[:, selected_channel]
            rospy.loginfo_throttle(
                5.0,
                "Selected channel %d/%d for inference, channel_rms=%s",
                selected_channel,
                channels,
                np.array2string(channel_rms, precision=2, separator=","),
            )

        audio /= scale

        if audio.size:
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio * audio)))
            rospy.loginfo_throttle(
                5.0,
                "Audio stats: samples=%d, channels=%d, sample_width=%d, peak=%.6f, rms=%.6f",
                audio.size,
                channels,
                sample_width_bytes,
                peak,
                rms,
            )

        for spectrogram in self.streamer.push(audio):
            result = self.spotter.predict_spectrogram(spectrogram)

            score = float(result["score"])
            label = str(result["label"])
            self._inference_count += 1

            if self.inference_publisher is not None:
                inference = VoiceCommand()
                inference.command = label
                inference.confidence = score
                self.inference_publisher.publish(inference)

            rospy.loginfo_throttle(
                2.0,
                "Inference alive: count=%d, label=%s, score=%.3f, threshold=%.3f",
                self._inference_count,
                label,
                score,
                self.spotter.threshold,
            )

            if score < self.spotter.threshold:
                continue

            command = VoiceCommand()
            command.command = label
            command.confidence = score

            self.publisher.publish(command)

            rospy.loginfo(
                "Detected: %s (%.3f)",
                command.command,
                command.confidence,
            )


def main() -> None:
    rospy.init_node("spot_keyword_spotting")
    VoiceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
