#!/usr/bin/env python3

from __future__ import annotations

import errno

import alsaaudio
import rospy
from std_msgs.msg import UInt8MultiArray


SAMPLE_FORMATS = {
    "S16_LE": alsaaudio.PCM_FORMAT_S16_LE,
    "S32_LE": alsaaudio.PCM_FORMAT_S32_LE,
}


class I2SRawAudioPublisher:
    def __init__(self) -> None:
        self.device = rospy.get_param("~device", "plughw:1,0")
        self.topic = rospy.get_param("~topic", "/audio/raw")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))
        self.period_size = int(rospy.get_param("~period_size", 256))
        self.sample_format = str(rospy.get_param("~sample_format", "S16_LE")).upper()
        self.log_stats = bool(rospy.get_param("~log_stats", True))

        if self.sample_format not in SAMPLE_FORMATS:
            supported = ", ".join(sorted(SAMPLE_FORMATS))
            raise ValueError("Unsupported sample_format '{}'. Supported values: {}".format(self.sample_format, supported))

        self.publisher = rospy.Publisher(self.topic, UInt8MultiArray, queue_size=10)
        self.pcm = None
        self.messages = 0
        self.bytes_sent = 0
        self.overruns = 0
        self.errors = 0
        self._open_pcm()

        rospy.loginfo(
            "Publishing raw I2S/ALSA audio: device=%s topic=%s rate=%d channels=%d format=%s period=%d",
            self.device,
            self.topic,
            self.sample_rate,
            self.channels,
            self.sample_format,
            self.period_size,
        )

    def _open_pcm(self) -> None:
        self.pcm = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NORMAL,
            device=self.device,
        )
        self.pcm.setchannels(self.channels)
        self.pcm.setrate(self.sample_rate)
        self.pcm.setformat(SAMPLE_FORMATS[self.sample_format])
        self.pcm.setperiodsize(self.period_size)

    def _recover_pcm(self) -> None:
        try:
            if self.pcm is not None:
                self.pcm.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass
        self._open_pcm()

    def spin(self) -> None:
        while not rospy.is_shutdown():
            try:
                length, data = self.pcm.read()
            except alsaaudio.ALSAAudioError as exc:
                self.errors += 1
                rospy.logwarn_throttle(2.0, "ALSA capture read failed: %s (count=%d)", exc, self.errors)
                try:
                    self._recover_pcm()
                except alsaaudio.ALSAAudioError as recover_exc:
                    rospy.logwarn_throttle(2.0, "Failed to recover ALSA capture stream: %s", recover_exc)
                continue

            if length <= 0:
                if length == -errno.EPIPE:
                    self.overruns += 1
                    rospy.logwarn_throttle(2.0, "ALSA capture overrun (count=%d); recovering", self.overruns)
                    try:
                        self._recover_pcm()
                    except alsaaudio.ALSAAudioError as exc:
                        rospy.logwarn_throttle(2.0, "Failed to recover ALSA capture stream: %s", exc)
                continue

            if not data:
                continue

            msg = UInt8MultiArray()
            msg.data = list(data)
            self.publisher.publish(msg)

            self.messages += 1
            self.bytes_sent += len(data)
            if self.log_stats:
                rospy.loginfo_throttle(
                    5.0,
                    "Raw audio alive: messages=%d bytes=%d latest_frames=%d latest_bytes=%d",
                    self.messages,
                    self.bytes_sent,
                    length,
                    len(data),
                )

    def shutdown(self) -> None:
        try:
            if self.pcm is not None:
                self.pcm.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass


def main() -> None:
    rospy.init_node("i2s_raw_audio_publisher")
    node = I2SRawAudioPublisher()
    rospy.on_shutdown(node.shutdown)
    node.spin()


if __name__ == "__main__":
    main()
