#!/usr/bin/env python3
"""ROS node that publishes keyword spotting results."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray

from spot_keyword_spotting.msg import VoiceCommand

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import KeywordSpotter  # noqa: E402


class VoiceNode:
    def __init__(self) -> None:
        package_root = Path(rospkg.RosPack().get_path("spot_keyword_spotting"))
        default_model = package_root / "keyword_spotting" / "weights" / "checkpoint.tflite"

        model_path = Path(rospy.get_param("~model_path", str(default_model)))
        threshold = float(rospy.get_param("~confidence", rospy.get_param("~threshold", 0.95)))
        labels = rospy.get_param("~labels", ["background", "marvin"])

        self.spotter = KeywordSpotter(model_path, labels=labels, threshold=threshold)
        self.publisher = rospy.Publisher("~command", VoiceCommand, queue_size=10)
        self.subscriber = rospy.Subscriber("~audio", Float32MultiArray, self.on_audio, queue_size=1)

    def on_audio(self, msg: Float32MultiArray) -> None:
        audio = np.asarray(msg.data, dtype=np.float32)
        if audio.size == 0:
            return

        result = self.spotter.predict_audio(audio)
        command = VoiceCommand()
        command.command = str(result["label"])
        command.confidence = float(result["score"])
        self.publisher.publish(command)


def main() -> None:
    rospy.init_node("spot_keyword_spotting")
    VoiceNode()
    rospy.spin()


if __name__ == "__main__":
    main()


