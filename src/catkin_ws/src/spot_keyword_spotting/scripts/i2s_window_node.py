#!/usr/bin/env python3
# spot_keyword_spotting/scripts/i2s_window_node.py

from __future__ import annotations

import errno
import time

import alsaaudio
import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray, MultiArrayDimension


SAMPLE_FORMATS = {
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, 2, np.dtype("<i2"), 32768.0),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, 4, np.dtype("<i4"), 2147483648.0),
}


class AveragingResampler:
    def __init__(self, input_sample_rate: int, output_sample_rate: int) -> None:
        self.input_sample_rate = int(input_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        if self.input_sample_rate <= 0 or self.output_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        if self.input_sample_rate % self.output_sample_rate != 0:
            raise ValueError("input sample rate must be an integer multiple of output sample rate")
        self.factor = self.input_sample_rate // self.output_sample_rate
        self.pending = np.empty(0, dtype=np.float32)
        self.input_count = 0
        self.output_count = 0

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)

        self.input_count += int(audio.size)
        if self.factor == 1:
            self.output_count += int(audio.size)
            return audio.astype(np.float32, copy=False)

        if self.pending.size:
            audio = np.concatenate((self.pending, audio))

        usable = audio.size - (audio.size % self.factor)
        self.pending = audio[usable:].copy()
        if usable <= 0:
            return np.empty(0, dtype=np.float32)

        out = audio[:usable].reshape(-1, self.factor).mean(axis=1).astype(np.float32)
        self.output_count += int(out.size)
        return out


class RingWindowPublisher:
    def __init__(self, window_samples: int, hop_samples: int, sample_rate: int, publisher: rospy.Publisher) -> None:
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.sample_rate = int(sample_rate)
        self.publisher = publisher
        self.ring = np.zeros(self.window_samples, dtype=np.float32)
        self.write_index = 0
        self.samples_seen = 0
        self.samples_since_publish = 0

    @property
    def ready(self) -> bool:
        return self.samples_seen >= self.window_samples

    def push(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        for sample in audio:
            self.ring[self.write_index] = sample
            self.write_index = (self.write_index + 1) % self.window_samples
            self.samples_seen += 1
            self.samples_since_publish += 1
            if self.ready and self.samples_since_publish >= self.hop_samples:
                self.publish()
                self.samples_since_publish = 0

    def current_window(self) -> np.ndarray:
        return np.concatenate((self.ring[self.write_index:], self.ring[:self.write_index])).astype(np.float32)

    def publish(self) -> None:
        window = self.current_window()
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension("sample_rate", self.sample_rate, self.sample_rate),
            MultiArrayDimension("window_samples", self.window_samples, self.window_samples),
            MultiArrayDimension("samples_seen", min(int(self.samples_seen), 2**32 - 1), 1),
        ]
        msg.data = window.tolist()
        self.publisher.publish(msg)
        rospy.loginfo_throttle(
            5.0,
            "Published Python KWS audio window: samples=%d, hop=%d, samples_seen=%d",
            window.size,
            self.hop_samples,
            self.samples_seen,
        )


class I2SWindowNode:
    def __init__(self) -> None:
        self.device = rospy.get_param("~device", "hw:1,0")
        self.input_sample_rate = int(rospy.get_param("~input_sample_rate", 48000))
        self.output_sample_rate = int(rospy.get_param("~output_sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 2))
        self.period_size = int(rospy.get_param("~period_size", 512))
        self.sample_format = str(rospy.get_param("~sample_format", "S32_LE")).upper()
        self.channel_index = int(rospy.get_param("~channel_index", 0))
        self.audio_gain = float(rospy.get_param("~audio_gain", 1.0))
        self.window_samples = int(rospy.get_param("~window_samples", 16000))
        self.hop_samples = int(rospy.get_param("~hop_samples", 3200))
        self.window_topic = rospy.get_param("~window_topic", "/voice/audio_window")

        if self.sample_format not in SAMPLE_FORMATS:
            raise ValueError("unsupported sample_format {}; supported: {}".format(self.sample_format, sorted(SAMPLE_FORMATS)))
        self.alsa_format, self.bytes_per_sample, self.dtype, self.scale = SAMPLE_FORMATS[self.sample_format]

        self.publisher = rospy.Publisher(self.window_topic, Float32MultiArray, queue_size=1)
        self.resampler = AveragingResampler(self.input_sample_rate, self.output_sample_rate)
        self.window_publisher = RingWindowPublisher(
            self.window_samples,
            self.hop_samples,
            self.output_sample_rate,
            self.publisher,
        )
        self.audio_messages = 0
        self.audio_bytes = 0
        self.alsa_overruns = 0
        self.alsa_errors = 0
        self.last_recover_time = 0.0
        self.pcm = None
        self._open_pcm()

        rospy.loginfo(
            "Python I2S window node started: device=%s, in=%d Hz, out=%d Hz, channels=%d, format=%s, period=%d, channel=%d, window=%d, hop=%d, topic=%s",
            self.device,
            self.input_sample_rate,
            self.output_sample_rate,
            self.channels,
            self.sample_format,
            self.period_size,
            self.channel_index,
            self.window_samples,
            self.hop_samples,
            self.window_topic,
        )

    def _open_pcm(self) -> None:
        self.pcm = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
        self.pcm.setchannels(self.channels)
        self.pcm.setrate(self.input_sample_rate)
        self.pcm.setformat(self.alsa_format)
        self.pcm.setperiodsize(self.period_size)

    def _recover_pcm(self) -> None:
        now = time.monotonic()
        if now - self.last_recover_time < 0.05:
            return
        self.last_recover_time = now
        try:
            if self.pcm is not None:
                self.pcm.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass
        self._open_pcm()

    def _read_mono(self) -> np.ndarray | None:
        try:
            length, data = self.pcm.read()
        except alsaaudio.ALSAAudioError as exc:
            self.alsa_errors += 1
            rospy.logwarn_throttle(2.0, "ALSA read failed in Python window node: %s (count=%d)", exc, self.alsa_errors)
            try:
                self._recover_pcm()
            except alsaaudio.ALSAAudioError as recover_exc:
                rospy.logwarn_throttle(2.0, "Failed to recover ALSA stream: %s", recover_exc)
            return None

        if length <= 0:
            if length == -errno.EPIPE:
                self.alsa_overruns += 1
                rospy.logwarn_throttle(2.0, "ALSA overrun in Python window node (count=%d); recovering", self.alsa_overruns)
                try:
                    self._recover_pcm()
                except alsaaudio.ALSAAudioError as recover_exc:
                    rospy.logwarn_throttle(2.0, "Failed to recover ALSA stream: %s", recover_exc)
                return None
            rospy.logwarn_throttle(2.0, "ALSA read returned length=%d", length)
            return None

        samples = np.frombuffer(data, dtype=self.dtype).astype(np.float32)
        expected_values = int(length) * self.channels
        if samples.size < expected_values:
            rospy.logwarn_throttle(2.0, "Short ALSA block: expected %d values, got %d", expected_values, samples.size)
            return None
        samples = samples[:expected_values].reshape(int(length), self.channels)

        if 0 <= self.channel_index < self.channels:
            selected_channel = self.channel_index
        else:
            rms_by_channel = np.sqrt(np.mean(samples * samples, axis=0))
            selected_channel = int(np.argmax(rms_by_channel))

        mono = samples[:, selected_channel] / self.scale
        mono = np.clip(mono * self.audio_gain, -1.0, 1.0).astype(np.float32)
        self.audio_messages += 1
        self.audio_bytes += len(data)
        rospy.loginfo_throttle(
            5.0,
            "Python I2S alive: messages=%d bytes=%d frames=%d selected_channel=%d peak=%.6f rms=%.6f",
            self.audio_messages,
            self.audio_bytes,
            length,
            selected_channel,
            float(np.max(np.abs(mono))) if mono.size else 0.0,
            float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0,
        )
        return mono

    def run(self) -> None:
        while not rospy.is_shutdown():
            audio_48k = self._read_mono()
            if audio_48k is None:
                continue
            audio_16k = self.resampler.process(audio_48k)
            self.window_publisher.push(audio_16k)
            rospy.loginfo_throttle(
                5.0,
                "Python window resampler: in_total=%d out_total=%d latest_in=%d latest_out=%d pending=%d",
                self.resampler.input_count,
                self.resampler.output_count,
                audio_48k.size,
                audio_16k.size,
                self.resampler.pending.size,
            )


def main() -> None:
    rospy.init_node("i2s_window_node")
    I2SWindowNode().run()


if __name__ == "__main__":
    main()