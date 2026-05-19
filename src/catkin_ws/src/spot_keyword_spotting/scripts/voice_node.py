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
        emit_hop_samples: int = HOP_LENGTH,
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

        model_path = Path(
            rospy.get_param("~model_path", str(default_model))
        )

        threshold = float(
            rospy.get_param("~threshold", 0.95)
        )

        labels = rospy.get_param(
            "~labels",
            ["background", "marvin"]
        )

        num_threads = int(
            rospy.get_param("~num_threads", 2)
        )

        audio_topic = rospy.get_param(
            "~audio_topic",
            "/audio/raw"
        )

        command_topic = rospy.get_param(
            "~command_topic",
            "/voice/command"
        )

        self.spotter = KeywordSpotter(
            model_path=model_path,
            labels=labels,
            threshold=threshold,
            num_threads=num_threads,
        )

        self.streamer = StreamingSpectrogram()

        self.publisher = rospy.Publisher(
            command_topic,
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
        rospy.loginfo("Model path: %s", model_path)

    def on_audio(self, msg: UInt8MultiArray) -> None:
        raw = np.frombuffer(
            bytearray(msg.data),
            dtype=np.uint8,
        )

        if raw.size < 2:
            return

        if raw.size % 2 != 0:
            raw = raw[:-1]

        audio = raw.view(np.int16).astype(np.float32)
        audio /= 32768.0

        for spectrogram in self.streamer.push(audio):
            result = self.spotter.predict_spectrogram(spectrogram)

            score = float(result["score"])

            if score < self.spotter.threshold:
                continue

            command = VoiceCommand()
            command.command = str(result["label"])
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