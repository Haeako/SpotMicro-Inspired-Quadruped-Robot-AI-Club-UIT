#!/usr/bin/env python3

import errno

import rospy
import alsaaudio

from std_msgs.msg import MultiArrayDimension, UInt8MultiArray


SAMPLE_FORMATS = {
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, 2),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, 4),
}


class I2SMicNode:
    def __init__(self):

        self.device = rospy.get_param("~device", "hw:1,0")
        self.sample_rate = int(rospy.get_param("~sample_rate", 48000))
        self.channels = int(rospy.get_param("~channels", 2))
        self.period_size = int(rospy.get_param("~period_size", 256))
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/raw")
        self.sample_format = rospy.get_param("~sample_format", "S32_LE").upper()

        if self.sample_format not in SAMPLE_FORMATS:
            supported = ", ".join(sorted(SAMPLE_FORMATS))
            raise ValueError(
                "Unsupported sample_format '{}'. Supported values: {}".format(
                    self.sample_format,
                    supported,
                )
            )

        self.alsa_format, self.bytes_per_sample = SAMPLE_FORMATS[self.sample_format]
        self._published_messages = 0
        self._published_bytes = 0
        self._last_channels = self.channels

        self.publisher = rospy.Publisher(
            self.audio_topic,
            UInt8MultiArray,
            queue_size=2
        )

        self.mic = None
        self._open_mic()

        rospy.loginfo(
            "I2S mic started: device=%s, topic=%s, sample_rate=%d, channels=%d, sample_format=%s, period_size=%d",
            self.device,
            self.audio_topic,
            self.sample_rate,
            self.channels,
            self.sample_format,
            self.period_size,
        )

    def _open_mic(self):
        try:
            self.mic = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device=self.device
            )

            self.mic.setchannels(self.channels)
            self.mic.setrate(self.sample_rate)
            self.mic.setformat(self.alsa_format)
            self.mic.setperiodsize(self.period_size)
        except alsaaudio.ALSAAudioError as exc:
            rospy.logfatal(
                "Failed to open I2S microphone device %s: %s",
                self.device,
                exc,
            )
            raise

    def _recover_mic(self):
        try:
            if self.mic is not None:
                self.mic.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass

        self._open_mic()

    def run(self):

        while not rospy.is_shutdown():

            length, data = self.mic.read()

            if length > 0:
                actual_channels = self.channels

                if len(data) % (length * self.bytes_per_sample) == 0:
                    actual_channels = len(data) // (length * self.bytes_per_sample)

                if actual_channels != self.channels:
                    rospy.logwarn_throttle(
                        5.0,
                        "ALSA returned %d channel(s), configured channel count is %d",
                        actual_channels,
                        self.channels,
                    )

                self._last_channels = actual_channels

                msg = UInt8MultiArray()
                msg.layout.dim = [
                    MultiArrayDimension("frames", length, len(data)),
                    MultiArrayDimension("channels", actual_channels, actual_channels * self.bytes_per_sample),
                    MultiArrayDimension("sample_width_bytes", self.bytes_per_sample, self.bytes_per_sample),
                    MultiArrayDimension("sample_rate", self.sample_rate, self.sample_rate),
                ]
                msg.data = bytearray(data)

                self.publisher.publish(msg)
                self._published_messages += 1
                self._published_bytes += len(data)
                rospy.loginfo_throttle(
                    5.0,
                    "Audio output alive: messages=%d, total_bytes=%d, latest_frames=%d, latest_channels=%d, latest_bytes=%d",
                    self._published_messages,
                    self._published_bytes,
                    length,
                    actual_channels,
                    len(data),
                )
            else:
                if length == -errno.EPIPE:
                    rospy.logwarn_throttle(
                        5.0,
                        "ALSA overrun on device %s (length=%d); recovering capture stream",
                        self.device,
                        length,
                    )
                    try:
                        self._recover_mic()
                    except alsaaudio.ALSAAudioError as exc:
                        rospy.logwarn_throttle(
                            5.0,
                            "Failed to recover ALSA capture stream: %s",
                            exc,
                        )
                    continue

                rospy.logwarn_throttle(
                    5.0,
                    "No audio frames read from device %s (length=%d)",
                    self.device,
                    length,
                )


def main():
    rospy.init_node("ros_i2s_mic")

    node = I2SMicNode()
    node.run()


if __name__ == "__main__":
    main()
