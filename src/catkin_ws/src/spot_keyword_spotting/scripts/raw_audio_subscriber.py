#!/usr/bin/env python3

from __future__ import annotations

import wave
from pathlib import Path

import alsaaudio
import rospy
from std_msgs.msg import UInt8MultiArray


SAMPLE_FORMATS = {
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, 2),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, 4),
}


class RawAudioSubscriber:
    def __init__(self) -> None:
        self.topic = rospy.get_param("~topic", "/audio/raw")
        self.output_device = rospy.get_param("~output_device", "default")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))
        self.period_size = int(rospy.get_param("~period_size", 256))
        self.sample_format = str(rospy.get_param("~sample_format", "S16_LE")).upper()
        self.play_audio = bool(rospy.get_param("~play_audio", True))
        self.wav_path = str(rospy.get_param("~wav_path", ""))
        self.log_stats = bool(rospy.get_param("~log_stats", True))

        if self.sample_format not in SAMPLE_FORMATS:
            supported = ", ".join(sorted(SAMPLE_FORMATS))
            raise ValueError("Unsupported sample_format '{}'. Supported values: {}".format(self.sample_format, supported))

        self.alsa_format, self.sample_width = SAMPLE_FORMATS[self.sample_format]
        self.playback = None
        self.wav = None
        self.messages = 0
        self.bytes_received = 0

        if self.play_audio:
            self._open_playback()
        if self.wav_path:
            self._open_wav(Path(self.wav_path))

        self.subscriber = rospy.Subscriber(self.topic, UInt8MultiArray, self.on_audio, queue_size=50)
        rospy.loginfo(
            "Subscribed raw audio: topic=%s play_audio=%s output_device=%s wav_path=%s rate=%d channels=%d format=%s",
            self.topic,
            self.play_audio,
            self.output_device,
            self.wav_path or "<disabled>",
            self.sample_rate,
            self.channels,
            self.sample_format,
        )

    def _open_playback(self) -> None:
        self.playback = alsaaudio.PCM(
            type=alsaaudio.PCM_PLAYBACK,
            mode=alsaaudio.PCM_NORMAL,
            device=self.output_device,
        )
        self.playback.setchannels(self.channels)
        self.playback.setrate(self.sample_rate)
        self.playback.setformat(self.alsa_format)
        self.playback.setperiodsize(self.period_size)

    def _open_wav(self, path: Path) -> None:
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.wav = wave.open(str(path), "wb")
        self.wav.setnchannels(self.channels)
        self.wav.setsampwidth(self.sample_width)
        self.wav.setframerate(self.sample_rate)

    def on_audio(self, msg: UInt8MultiArray) -> None:
        data = bytes(msg.data)
        if not data:
            return

        if self.playback is not None:
            try:
                self.playback.write(data)
            except alsaaudio.ALSAAudioError as exc:
                rospy.logwarn_throttle(2.0, "ALSA playback write failed: %s", exc)

        if self.wav is not None:
            self.wav.writeframes(data)

        self.messages += 1
        self.bytes_received += len(data)
        if self.log_stats:
            rospy.loginfo_throttle(
                5.0,
                "Raw audio received: messages=%d bytes=%d latest_bytes=%d",
                self.messages,
                self.bytes_received,
                len(data),
            )

    def shutdown(self) -> None:
        try:
            if self.wav is not None:
                self.wav.close()
        finally:
            try:
                if self.playback is not None:
                    self.playback.close()
            except (AttributeError, alsaaudio.ALSAAudioError):
                pass


def main() -> None:
    rospy.init_node("raw_audio_subscriber")
    node = RawAudioSubscriber()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
