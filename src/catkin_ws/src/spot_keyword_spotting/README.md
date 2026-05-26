# spot_keyword_spotting

`spot_keyword_spotting` is a ROS package for detecting a wake word or simple
voice command from an I2S microphone stream on the Spot Micro robot.

The implementation is inspired by the keyword spotting approach from
[atomic14/diy-alexa](https://github.com/atomic14/diy-alexa).

## Overview

The package captures audio directly from an ALSA PCM input device, preprocesses
the stream into a spectrogram, runs a TensorFlow Lite keyword spotting model,
and publishes a ROS message when the configured keyword is detected.

Audio capture is handled through the Linux HAL ALSA layer using `pyalsaaudio`.
This allows the node to read directly from an I2S microphone device such as
`hw:1,0` without requiring a separate ROS audio topic.

Main components:

- `scripts/i2s_voice_node.py`: ROS node that captures audio, runs inference,
  and publishes detection events.
- `keyword_spotting/weights/checkpoint.tflite`: default TensorFlow Lite model.
- `keyword_spotting/native`: C++/pybind11 preprocessing utilities used by the
  Python node.
- `config/config.yml`: example configuration for model, audio, and debug
  parameters.

## Build

The build step is required because the package compiles the native
`kws_native` Python module used for spectrogram generation.

## Dependencies

Required runtime dependencies include:

- ROS with `rospy`, `roscpp`, and `std_msgs`
- `pyalsaaudio`
- `numpy`
- `rospkg`
- `tflite_runtime` or `tensorflow`
- ALSA support enabled on the target Linux system

On the robot, make sure the I2S microphone is visible through ALSA before
starting the node:

```bash
arecord -l
```

## Usage

Start ROS core:

```bash
roscore
```

In another terminal, source the workspace and run the keyword spotting node:

```bash
cd src/catkin_ws
source devel/setup.bash
rosrun spot_keyword_spotting i2s_voice_node.py
```

By default, the node opens ALSA device `hw:1,0`, captures 48 kHz stereo audio,
selects one channel, resamples it to the model input rate, and checks for the
keyword using the bundled TFLite model.

When the confidence score passes the configured threshold, the node publishes:

```text
/voice_cmd std_msgs/Bool
```

A `True` message means the keyword was detected.

## Parameters

The node reads private ROS parameters. Common parameters are:

- `~device`: ALSA PCM device name, default `hw:1,0`.
- `~input_sample_rate`: input audio sample rate, default `48000`.
- `~channels`: number of ALSA input channels, default `2`.
- `~sample_format`: ALSA sample format, either `S32_LE` or `S16_LE`.
- `~channel_index`: channel to use; set `-1` to auto-select the loudest channel.
- `~audio_gain`: gain applied before inference.
- `~model_path`: path to the TFLite model.
- `~confidence`: detection threshold.
- `~labels`: model labels, default `["background", "marvin"]`.
- `~inference_rate`: how often inference is attempted.
- `~save_detected_chunks`: save detected audio windows as `.wav` files.

Example:

```bash
rosrun spot_keyword_spotting i2s_voice_node.py _device:=hw:1,0 _confidence:=0.95 _audio_gain:=5.0
```

## Notes

- The package currently reads audio from HAL ALSA directly instead of
  subscribing to an audio topic.
- The current node publishes detection events on `/voice_cmd`.
- If `kws_native` cannot be imported, rebuild the catkin workspace first.
- If ALSA cannot open the microphone device, check the device name with
  `arecord -l` and update `~device` accordingly.
