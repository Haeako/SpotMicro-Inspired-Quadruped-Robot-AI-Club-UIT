#!/usr/bin/env python3
"""ROS node that records I2S microphone audio directly to WAV chunks."""

from __future__ import annotations

import errno
import time
import wave
from pathlib import Path

import alsaaudio
import numpy as np
import rospy
import rospkg


SAMPLE_FORMATS = {
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, 2, np.dtype("<i2"), 32768.0),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, 4, np.dtype("<i4"), 2147483648.0),
}


class I2SWavRecorderNode:
    def __init__(self) -> None:
        package_root = Path(rospkg.RosPack().get_path("ros_i2s_mic"))

        self.device = rospy.get_param("~device", "hw:1,0")
        self.sample_rate = int(rospy.get_param("~sample_rate", 48000))
        self.channels = int(rospy.get_param("~channels", 2))
        self.period_size = int(rospy.get_param("~period_size", 256))
        self.sample_format = rospy.get_param("~sample_format", "S32_LE").upper()
        self.channel_index = int(rospy.get_param("~channel_index", -1))
        self.gain = float(rospy.get_param("~gain", 1.0))
        self.chunk_seconds = float(rospy.get_param("~chunk_seconds", 5.0))
        self.output_dir = Path(rospy.get_param("~output_dir", str(package_root / "recordings")))
        self.s32_shift_bits = int(rospy.get_param("~s32_shift_bits", 0))
        self.stop_after_chunks = int(rospy.get_param("~stop_after_chunks", 0))

        if self.sample_format not in SAMPLE_FORMATS:
            supported = ", ".join(sorted(SAMPLE_FORMATS))
            raise ValueError("Unsupported sample_format '{}'. Supported: {}".format(self.sample_format, supported))

        self.alsa_format, self.bytes_per_sample, self.dtype, self.scale = SAMPLE_FORMATS[self.sample_format]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_frames = max(1, int(round(self.sample_rate * self.chunk_seconds)))
        self._chunks = []
        self._frames_buffered = 0
        self._saved_chunks = 0
        self._frames_total = 0
        self._mic = None

        self._open_mic()

        rospy.loginfo(
            "I2S WAV recorder started: device=%s rate=%d channels=%d format=%s period=%d channel_index=%d gain=%.3f chunk=%.2fs output=%s s32_shift_bits=%d",
            self.device,
            self.sample_rate,
            self.channels,
            self.sample_format,
            self.period_size,
            self.channel_index,
            self.gain,
            self.chunk_seconds,
            self.output_dir,
            self.s32_shift_bits,
        )

    def _open_mic(self) -> None:
        self._mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
        self._mic.setchannels(self.channels)
        self._mic.setrate(self.sample_rate)
        self._mic.setformat(self.alsa_format)
        self._mic.setperiodsize(self.period_size)

    def _recover_mic(self) -> None:
        try:
            if self._mic is not None:
                self._mic.close()
        except (AttributeError, alsaaudio.ALSAAudioError):
            pass
        self._open_mic()

    def _decode_block(self, length: int, data: bytes) -> np.ndarray | None:
        samples = np.frombuffer(data, dtype=self.dtype)
        expected_values = length * self.channels
        if samples.size < expected_values:
            rospy.logwarn_throttle(2.0, "Expected %d values, got %d", expected_values, samples.size)
            return None

        samples = samples[:expected_values].reshape(length, self.channels)
        raw_for_stats = samples.astype(np.float32)

        if self.sample_format == "S32_LE" and self.s32_shift_bits > 0:
            samples = samples >> self.s32_shift_bits
            scale = float(1 << max(1, 31 - self.s32_shift_bits))
        else:
            scale = self.scale

        float_samples = samples.astype(np.float32) / scale

        if self.channels > 1:
            channel_rms = np.sqrt(np.mean(raw_for_stats * raw_for_stats, axis=0))
            channel_peak = np.max(np.abs(raw_for_stats), axis=0)
            if 0 <= self.channel_index < self.channels:
                channel = self.channel_index
            else:
                channel = int(np.argmax(channel_rms))
            rospy.loginfo_throttle(
                2.0,
                "Channel stats peak=%s rms=%s selected=%d",
                np.array2string(channel_peak, precision=1, separator=","),
                np.array2string(channel_rms, precision=1, separator=","),
                channel,
            )
            mono = float_samples[:, channel]
        else:
            mono = float_samples.reshape(-1)

        mono = np.clip(mono * self.gain, -1.0, 1.0).astype(np.float32, copy=False)
        return mono

    def _write_chunk(self, audio: np.ndarray) -> None:
        self._saved_chunks += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / "{}_i2s_{:04d}.wav".format(stamp, self._saved_chunks)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm.tobytes())

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        clip_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0
        rospy.loginfo("Saved %s frames=%d peak=%.6f rms=%.6f clip_ratio=%.4f", path, audio.size, peak, rms, clip_ratio)

    def _append_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        self._chunks.append(audio.copy())
        self._frames_buffered += int(audio.size)
        self._frames_total += int(audio.size)

        while self._frames_buffered >= self.chunk_frames:
            merged = np.concatenate(self._chunks)
            chunk = merged[:self.chunk_frames]
            remaining = merged[self.chunk_frames:]
            self._chunks = [remaining] if remaining.size else []
            self._frames_buffered = int(remaining.size)
            self._write_chunk(chunk)

            if self.stop_after_chunks > 0 and self._saved_chunks >= self.stop_after_chunks:
                rospy.signal_shutdown("Recorded requested chunk count")
                return

    def run(self) -> None:
        while not rospy.is_shutdown():
            length, data = self._mic.read()
            if length > 0:
                audio = self._decode_block(length, data)
                if audio is not None:
                    self._append_audio(audio)
                    rospy.loginfo_throttle(
                        5.0,
                        "Recorder alive: frames_total=%d buffered=%d saved_chunks=%d latest_frames=%d",
                        self._frames_total,
                        self._frames_buffered,
                        self._saved_chunks,
                        length,
                    )
                continue

            if length == -errno.EPIPE:
                rospy.logwarn_throttle(2.0, "ALSA overrun on %s; recovering", self.device)
                try:
                    self._recover_mic()
                except alsaaudio.ALSAAudioError as exc:
                    rospy.logwarn_throttle(2.0, "Failed to recover ALSA stream: %s", exc)
                continue

            rospy.logwarn_throttle(2.0, "No audio frames read from %s length=%d", self.device, length)


def main() -> None:
    rospy.init_node("i2s_wav_recorder")
    node = I2SWavRecorderNode()
    node.run()


if __name__ == "__main__":
    main()
