#pragma once

#include <vector>

namespace Spectrogram {
std::vector<float> normalize_audio(const std::vector<float>& input);
std::vector<float> get_spectrogram(const std::vector<float>& input, bool normalize = true);
}
