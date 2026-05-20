#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

class AlsaRuntime {
public:
  using snd_pcm_t = void;
  using snd_pcm_stream_t = int;
  using snd_pcm_format_t = int;
  using snd_pcm_access_t = int;
  using snd_pcm_sframes_t = long;
  using snd_pcm_uframes_t = unsigned long;

  static constexpr snd_pcm_stream_t SND_PCM_STREAM_CAPTURE = 1;
  static constexpr snd_pcm_access_t SND_PCM_ACCESS_RW_INTERLEAVED = 3;
  static constexpr snd_pcm_format_t SND_PCM_FORMAT_S16_LE = 2;
  static constexpr snd_pcm_format_t SND_PCM_FORMAT_S32_LE = 10;

  using OpenFn = int (*)(snd_pcm_t **, const char *, snd_pcm_stream_t, int);
  using SetParamsFn = int (*)(snd_pcm_t *, snd_pcm_format_t, snd_pcm_access_t, unsigned int, unsigned int, int, unsigned int);
  using ReadiFn = snd_pcm_sframes_t (*)(snd_pcm_t *, void *, snd_pcm_uframes_t);
  using PrepareFn = int (*)(snd_pcm_t *);
  using CloseFn = int (*)(snd_pcm_t *);
  using StrErrorFn = const char *(*)(int);

  AlsaRuntime() {
    handle_ = dlopen("libasound.so.2", RTLD_NOW);
    if (handle_ == nullptr) {
      throw std::runtime_error("dlopen libasound.so.2 failed: " + std::string(dlerror()));
    }

    open = load<OpenFn>("snd_pcm_open");
    set_params = load<SetParamsFn>("snd_pcm_set_params");
    readi = load<ReadiFn>("snd_pcm_readi");
    prepare = load<PrepareFn>("snd_pcm_prepare");
    close = load<CloseFn>("snd_pcm_close");
    strerror = load<StrErrorFn>("snd_strerror");
  }

  ~AlsaRuntime() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  OpenFn open = nullptr;
  SetParamsFn set_params = nullptr;
  ReadiFn readi = nullptr;
  PrepareFn prepare = nullptr;
  CloseFn close = nullptr;
  StrErrorFn strerror = nullptr;

private:
  template <typename T>
  T load(const char *name) {
    dlerror();
    void *symbol = dlsym(handle_, name);
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) {
      throw std::runtime_error("dlsym failed for " + std::string(name) + ": " + (error ? error : "null"));
    }
    return reinterpret_cast<T>(symbol);
  }

  void *handle_ = nullptr;
};

class I2SWindowNode {
public:
  I2SWindowNode()
      : private_nh_("~") {
    private_nh_.param<std::string>("device", device_, "hw:1,0");
    private_nh_.param("input_sample_rate", input_sample_rate_, 48000);
    private_nh_.param("output_sample_rate", output_sample_rate_, 16000);
    private_nh_.param("channels", channels_, 2);
    private_nh_.param("period_size", period_size_, 512);
    private_nh_.param("channel_index", channel_index_, 0);
    private_nh_.param("audio_gain", audio_gain_, 1.0);
    private_nh_.param("latency_us", latency_us_, 50000);
    private_nh_.param<std::string>("sample_format", sample_format_, "S32_LE");
    private_nh_.param<std::string>("window_topic", window_topic_, "/voice/audio_window");
    private_nh_.param("window_samples", window_samples_, 16000);
    private_nh_.param("hop_samples", hop_samples_, 3200);

    if (input_sample_rate_ <= 0 || output_sample_rate_ <= 0 || channels_ <= 0) {
      throw std::runtime_error("sample rates and channels must be positive");
    }
    if (input_sample_rate_ % output_sample_rate_ != 0) {
      throw std::runtime_error("input_sample_rate must be an integer multiple of output_sample_rate");
    }

    decimation_factor_ = input_sample_rate_ / output_sample_rate_;
    ring_.assign(static_cast<size_t>(window_samples_), 0.0f);

    window_pub_ = nh_.advertise<std_msgs::Float32MultiArray>(window_topic_, 2);
    openPcm();

    ROS_INFO("I2S window node started: device=%s, in=%d Hz, out=%d Hz, channels=%d, format=%s, period=%d, latency_us=%d, channel=%d, window=%d, hop=%d, topic=%s",
             device_.c_str(), input_sample_rate_, output_sample_rate_, channels_, sample_format_.c_str(), period_size_, latency_us_,
             channel_index_, window_samples_, hop_samples_, window_topic_.c_str());
  }

  ~I2SWindowNode() {
    if (pcm_ != nullptr) {
      alsa_.close(pcm_);
      pcm_ = nullptr;
    }
  }

  void run() {
    std::vector<uint8_t> bytes(static_cast<size_t>(period_size_ * channels_ * bytes_per_sample_));

    while (ros::ok()) {
      const auto frames = alsa_.readi(pcm_, bytes.data(), static_cast<AlsaRuntime::snd_pcm_uframes_t>(period_size_));
      if (frames < 0) {
        handleReadError(frames);
        ros::spinOnce();
        continue;
      }
      if (frames == 0) {
        ros::spinOnce();
        continue;
      }

      processBlock(bytes.data(), static_cast<size_t>(frames));
      ros::spinOnce();
    }
  }

private:
  void openPcm() {
    if (pcm_ != nullptr) {
      alsa_.close(pcm_);
      pcm_ = nullptr;
    }

    int err = alsa_.open(&pcm_, device_.c_str(), AlsaRuntime::SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
      throw std::runtime_error("snd_pcm_open failed: " + std::string(alsa_.strerror(err)));
    }

    AlsaRuntime::snd_pcm_format_t format = AlsaRuntime::SND_PCM_FORMAT_S32_LE;
    bytes_per_sample_ = 4;
    scale_ = 2147483648.0f;

    if (sample_format_ == "S16_LE") {
      format = AlsaRuntime::SND_PCM_FORMAT_S16_LE;
      bytes_per_sample_ = 2;
      scale_ = 32768.0f;
    } else if (sample_format_ != "S32_LE") {
      throw std::runtime_error("unsupported sample_format: " + sample_format_);
    }

    err = alsa_.set_params(
        pcm_,
        format,
        AlsaRuntime::SND_PCM_ACCESS_RW_INTERLEAVED,
        static_cast<unsigned int>(channels_),
        static_cast<unsigned int>(input_sample_rate_),
        1,
        static_cast<unsigned int>(latency_us_));
    if (err < 0) {
      throw std::runtime_error("snd_pcm_set_params failed: " + std::string(alsa_.strerror(err)));
    }

    alsa_.prepare(pcm_);
  }

  void handleReadError(long frames) {
    if (frames == -32) {
      ++overruns_;
      ROS_WARN_THROTTLE(2.0, "ALSA overrun in C++ capture (count=%d); preparing stream", overruns_);
      alsa_.prepare(pcm_);
      return;
    }

    ++read_errors_;
    ROS_WARN_THROTTLE(2.0, "ALSA read failed: %s (code=%ld, count=%d); reopening capture stream", alsa_.strerror(static_cast<int>(frames)), frames, read_errors_);
    try {
      openPcm();
    } catch (const std::exception &exc) {
      ROS_WARN_THROTTLE(2.0, "Failed to reopen ALSA capture stream: %s", exc.what());
    }
  }

  void processBlock(const uint8_t *data, size_t frames) {
    float latest_peak = 0.0f;
    double rms_accum = 0.0;

    for (size_t frame = 0; frame < frames; ++frame) {
      float sample = readSelectedSample(data, frame);
      sample = std::max(-1.0f, std::min(1.0f, sample * static_cast<float>(audio_gain_)));
      latest_peak = std::max(latest_peak, std::abs(sample));
      rms_accum += static_cast<double>(sample) * sample;
      pushInputSample(sample);
    }

    total_input_frames_ += frames;
    ROS_INFO_THROTTLE(5.0, "C++ I2S alive: in_frames=%llu out_samples=%llu latest_frames=%zu peak=%.6f rms=%.6f",
                      static_cast<unsigned long long>(total_input_frames_),
                      static_cast<unsigned long long>(total_output_samples_),
                      frames,
                      latest_peak,
                      std::sqrt(rms_accum / std::max<size_t>(1, frames)));
  }

  float readSelectedSample(const uint8_t *data, size_t frame) const {
    int selected = channel_index_;
    if (selected < 0 || selected >= channels_) {
      selected = 0;
    }

    const size_t offset = (frame * static_cast<size_t>(channels_) + static_cast<size_t>(selected)) * bytes_per_sample_;
    if (bytes_per_sample_ == 2) {
      int16_t raw = 0;
      std::memcpy(&raw, data + offset, sizeof(raw));
      return static_cast<float>(raw) / scale_;
    }

    int32_t raw = 0;
    std::memcpy(&raw, data + offset, sizeof(raw));
    return static_cast<float>(raw) / scale_;
  }

  void pushInputSample(float sample) {
    decimation_accum_ += sample;
    ++decimation_count_;

    if (decimation_count_ < decimation_factor_) {
      return;
    }

    const float out = decimation_accum_ / static_cast<float>(decimation_factor_);
    decimation_accum_ = 0.0f;
    decimation_count_ = 0;
    pushOutputSample(out);
  }

  void pushOutputSample(float sample) {
    ring_[ring_write_index_] = sample;
    ring_write_index_ = (ring_write_index_ + 1) % ring_.size();
    ++total_output_samples_;
    ++samples_since_publish_;
    if (!ring_full_ && total_output_samples_ >= static_cast<uint64_t>(ring_.size())) {
      ring_full_ = true;
    }

    if (ring_full_ && samples_since_publish_ >= static_cast<uint64_t>(hop_samples_)) {
      publishWindow();
      samples_since_publish_ = 0;
    }
  }

  void publishWindow() {
    std_msgs::Float32MultiArray msg;
    msg.layout.dim.resize(3);
    msg.layout.dim[0].label = "sample_rate";
    msg.layout.dim[0].size = static_cast<uint32_t>(output_sample_rate_);
    msg.layout.dim[0].stride = static_cast<uint32_t>(output_sample_rate_);
    msg.layout.dim[1].label = "window_samples";
    msg.layout.dim[1].size = static_cast<uint32_t>(window_samples_);
    msg.layout.dim[1].stride = static_cast<uint32_t>(window_samples_);
    msg.layout.dim[2].label = "samples_seen";
    msg.layout.dim[2].size = static_cast<uint32_t>(std::min<uint64_t>(total_output_samples_, UINT32_MAX));
    msg.layout.dim[2].stride = 1;
    msg.data.resize(ring_.size());

    const size_t start = ring_write_index_;
    for (size_t i = 0; i < ring_.size(); ++i) {
      msg.data[i] = ring_[(start + i) % ring_.size()];
    }

    window_pub_.publish(msg);
    ROS_INFO_THROTTLE(5.0, "Published KWS audio window: samples=%zu, hop=%d", msg.data.size(), hop_samples_);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Publisher window_pub_;
  AlsaRuntime alsa_;
  AlsaRuntime::snd_pcm_t *pcm_ = nullptr;

  std::string device_;
  std::string sample_format_;
  std::string window_topic_;
  int input_sample_rate_ = 48000;
  int output_sample_rate_ = 16000;
  int channels_ = 2;
  int period_size_ = 512;
  int latency_us_ = 50000;
  int channel_index_ = 0;
  int window_samples_ = 16000;
  int hop_samples_ = 3200;
  int decimation_factor_ = 3;
  int decimation_count_ = 0;
  int bytes_per_sample_ = 4;
  int overruns_ = 0;
  int read_errors_ = 0;
  double audio_gain_ = 1.0;
  float scale_ = 2147483648.0f;
  float decimation_accum_ = 0.0f;

  std::vector<float> ring_;
  size_t ring_write_index_ = 0;
  bool ring_full_ = false;
  uint64_t total_input_frames_ = 0;
  uint64_t total_output_samples_ = 0;
  uint64_t samples_since_publish_ = 0;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "i2s_window_node");

  try {
    I2SWindowNode node;
    node.run();
  } catch (const std::exception &exc) {
    ROS_FATAL("i2s_window_node failed: %s", exc.what());
    return 1;
  }

  return 0;
}