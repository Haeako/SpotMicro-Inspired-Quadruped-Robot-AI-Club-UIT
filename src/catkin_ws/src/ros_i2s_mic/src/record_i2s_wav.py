#!/usr/bin/env python3

import argparse
import wave

import alsaaudio
import numpy as np


SAMPLE_FORMATS = {
    "S16_LE": (alsaaudio.PCM_FORMAT_S16_LE, 2, np.dtype("<i2"), 32768.0),
    "S32_LE": (alsaaudio.PCM_FORMAT_S32_LE, 4, np.dtype("<i4"), 2147483648.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record an I2S microphone to a mono 16-bit WAV file."
    )
    parser.add_argument("--device", default="hw:1,0")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--sample-format", choices=sorted(SAMPLE_FORMATS), default="S32_LE")
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--period-size", type=int, default=512)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--output", default="i2s_mic_test.wav")
    parser.add_argument(
        "--dump-stats",
        action="store_true",
        help="Print raw per-channel stats before writing the WAV file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    alsa_format, bytes_per_sample, dtype, scale = SAMPLE_FORMATS[args.sample_format]

    pcm = alsaaudio.PCM(
        alsaaudio.PCM_CAPTURE,
        alsaaudio.PCM_NORMAL,
        device=args.device,
    )
    pcm.setchannels(args.channels)
    pcm.setrate(args.rate)
    pcm.setformat(alsa_format)
    pcm.setperiodsize(args.period_size)

    target_frames = int(args.rate * args.duration)
    chunks = []
    frames_read = 0

    print(
        "Recording {}s from {}: rate={}, channels={}, format={}, channel_index={}, output={}".format(
            args.duration,
            args.device,
            args.rate,
            args.channels,
            args.sample_format,
            args.channel_index,
            args.output,
        )
    )

    while frames_read < target_frames:
        length, data = pcm.read()

        if length <= 0:
            print("Warning: ALSA read returned length={}".format(length))
            continue

        samples = np.frombuffer(data, dtype=dtype).astype(np.float32)
        expected_values = length * args.channels

        if samples.size < expected_values:
            print(
                "Warning: expected at least {} values, got {}".format(
                    expected_values,
                    samples.size,
                )
            )
            continue

        samples = samples[:expected_values].reshape(length, args.channels)
        channel = args.channel_index

        if channel < 0 or channel >= args.channels:
            rms = np.sqrt(np.mean(samples * samples, axis=0))
            channel = int(np.argmax(rms))

        mono = samples[:, channel] / scale
        if args.dump_stats and not chunks:
            rms = np.sqrt(np.mean(samples * samples, axis=0))
            peak = np.max(np.abs(samples), axis=0)
            print(
                "First block: length={}, bytes={}, values={}, per_channel_peak={}, per_channel_rms={}".format(
                    length,
                    len(data),
                    samples.size,
                    np.array2string(peak, precision=1, separator=","),
                    np.array2string(rms, precision=1, separator=","),
                )
            )
        chunks.append(mono)
        frames_read += length

    audio = np.concatenate(chunks)[:target_frames]
    if args.dump_stats and audio.size:
        print(
            "Selected channel preview: float_min={:.9f}, float_max={:.9f}, raw_min={:.0f}, raw_max={:.0f}, first10_raw={}".format(
                float(np.min(audio)),
                float(np.max(audio)),
                float(np.min(audio * scale)),
                float(np.max(audio * scale)),
                np.array2string((audio[:10] * scale).astype(np.int64), separator=","),
            )
        )

    audio = np.clip(audio * args.gain, -1.0, 1.0)
    wav_pcm = (audio * 32767.0).astype("<i2")

    with wave.open(args.output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(args.rate)
        wav.writeframes(wav_pcm.tobytes())

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    print(
        "Wrote {} frames to {}. peak={:.6f}, rms={:.6f}".format(
            wav_pcm.size,
            args.output,
            peak,
            rms,
        )
    )


if __name__ == "__main__":
    main()
