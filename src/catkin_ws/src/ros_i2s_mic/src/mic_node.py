#!/usr/bin/env python3

import rospy
import alsaaudio

from std_msgs.msg import UInt8MultiArray

class I2SMicNode:
    def __init__(self):

        self.device = rospy.get_param("~device", "hw:1,0")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))
        self.period_size = int(rospy.get_param("~period_size", 512))
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/raw")
        self._published_messages = 0
        self._published_bytes = 0

        self.publisher = rospy.Publisher(
            self.audio_topic,
            UInt8MultiArray,
            queue_size=2
        )

        try:
            self.mic = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device=self.device
            )

            self.mic.setchannels(self.channels)
            self.mic.setrate(self.sample_rate)
            self.mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            self.mic.setperiodsize(self.period_size)
        except alsaaudio.ALSAAudioError as exc:
            rospy.logfatal(
                "Failed to open I2S microphone device %s: %s",
                self.device,
                exc,
            )
            raise

        rospy.loginfo(
            "I2S mic started: device=%s, topic=%s, sample_rate=%d, channels=%d, period_size=%d",
            self.device,
            self.audio_topic,
            self.sample_rate,
            self.channels,
            self.period_size,
        )

    def run(self):

        rate = rospy.Rate(100)

        while not rospy.is_shutdown():

            length, data = self.mic.read()

            if length > 0:

                msg = UInt8MultiArray()
                msg.data = bytearray(data)

                self.publisher.publish(msg)
                self._published_messages += 1
                self._published_bytes += len(data)
                rospy.loginfo_throttle(
                    5.0,
                    "Audio output alive: messages=%d, total_bytes=%d, latest_frames=%d, latest_bytes=%d",
                    self._published_messages,
                    self._published_bytes,
                    length,
                    len(data),
                )
            else:
                rospy.logwarn_throttle(
                    5.0,
                    "No audio frames read from device %s (length=%d)",
                    self.device,
                    length,
                )

            rate.sleep()


def main():
    rospy.init_node("ros_i2s_mic")

    node = I2SMicNode()
    node.run()


if __name__ == "__main__":
    main()
