#!/usr/bin/env python3
# spot_keyword_spotting/scripts/window_infer_node.py

from __future__ import annotations

import importlib.util
import sys
import time
import wave
from pathlib import Path

import numpy as np
import rospy
import rospkg

from std_msgs.msg import Float32MultiArray
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
KeywordSpotter = _voice_node.KeywordSpotter
_get_private_param = _voice_node._get_private_param
_resolve_package_path = _voice_node._resolve_package_path
get_spectrogram = _voice_node.get_spectrogram


class WindowInferNode:
    def __init__(self) -> None:
        package_root = Path(rospkg.RosPack().get_path("spot_keyword_spotting"))
        default_model = package_root / "keyword_spotting" / "weights" / "checkpoint.tflite"

        model_path = _resolve_package_path(package_root, _get_private_param(("model_path", "model/path"), str(default_model)))
        labels = tuple(_get_private_param(("labels", "model/labels"), DEFAULT_LABELS))
        threshold = float(_get_private_param(("confidence", "model/confidence"), 0.95))
        num_threads = int(_get_private_param(("num_threads", "model/num_threads"), 1))
        window_topic = _get_private_param(("window_topic", "ros/window_topic"), "/voice/audio_window")
        command_topic = _get_private_param(("command_topic", "ros/publish_topic"), "/voice/command")
        inference_topic = _get_private_param(("inference_topic", "ros/inference_topic"), "/voice/inference")
        publish_inference = bool(_get_private_param(("publish_inference", "ros/publish_inference"), True))
        self.save_detected_chunks = bool(_get_private_param(("save_detected_chunks", "debug/save_detected_chunks"), False))
        self.detected_chunk_dir = _resolve_package_path(
            package_root,
            _get_private_param(("detected_chunk_dir", "debug/detected_chunk_dir"), "detected_chunks"),
        )

        if self.save_detected_chunks:
            self.detected_chunk_dir.mkdir(parents=True, exist_ok=True)

        self.spotter = KeywordSpotter(model_path=model_path, labels=labels, threshold=threshold, num_threads=num_threads)
        self.publisher = rospy.Publisher(command_topic, VoiceCommand, queue_size=10)
        self.inference_publisher = None
        if publish_inference:
            self.inference_publisher = rospy.Publisher(inference_topic, VoiceCommand, queue_size=10)

        self.subscriber = rospy.Subscriber(window_topic, Float32MultiArray, self.on_window, queue_size=1)
        self.inference_count = 0
        self.best_score = 0.0
        self.best_label = "background"
        self.saved_detected_chunks = 0

        rospy.loginfo("Window infer node started: topic=%s, model=%s, threshold=%.3f", window_topic, model_path, threshold)

    def _write_wav(self, path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
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

    def on_window(self, msg: Float32MultiArray) -> None:
        audio = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if audio.size != EXPECTED_SAMPLES:
            rospy.logwarn_throttle(
                2.0,
                "Ignoring audio window with %d samples; expected %d",
                audio.size,
                EXPECTED_SAMPLES,
            )
            return

        spectrogram = get_spectrogram(audio)
        result = self.spotter.predict_spectrogram(spectrogram)
        label = str(result["label"])
        score = float(result["score"])
        self.inference_count += 1

        if score >= self.best_score:
            self.best_score = score
            self.best_label = label

        if self.inference_publisher is not None:
            inference = VoiceCommand()
            inference.command = label
            inference.confidence = score
            self.inference_publisher.publish(inference)

        rospy.loginfo_throttle(
            2.0,
            "Window inference alive: count=%d, label=%s, score=%.3f, best_label=%s, best_score=%.3f, threshold=%.3f",
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
        rospy.loginfo("Window detected: %s (%.3f)", label, score)


def main() -> None:
    rospy.init_node("window_infer_node")
    WindowInferNode()
    rospy.spin()


if __name__ == "__main__":
    main()