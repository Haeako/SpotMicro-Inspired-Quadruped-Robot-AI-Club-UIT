#include "../include/spectrogram.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr int EXPECTED_SAMPLES = 16000;
constexpr int WINDOW_SIZE = 320;
constexpr int HOP_LENGTH = 160;
constexpr int FRAME_COUNT = 99;
constexpr int FREQ_POOL_SIZE = 6;
constexpr int POOLED_BINS = 26;
constexpr float LOG_EPSILON = 1.0e-6f;

static float hann_window[WINDOW_SIZE];
static double cos_table[POOLED_BINS * FREQ_POOL_SIZE][WINDOW_SIZE];
static double sin_table[POOLED_BINS * FREQ_POOL_SIZE][WINDOW_SIZE];

static bool tables_ready = false;

static void init_tables()
{
    if (tables_ready) {
        return;
    }

    for (int n = 0; n < WINDOW_SIZE; ++n) {
        hann_window[n] =
            0.5f - 0.5f *
            std::cos((2.0 * M_PI * n) / WINDOW_SIZE);
    }

    for (int k = 0; k < POOLED_BINS * FREQ_POOL_SIZE; ++k) {
        for (int n = 0; n < WINDOW_SIZE; ++n) {

            double phase =
                (2.0 * M_PI * k * n) / WINDOW_SIZE;

            cos_table[k][n] = std::cos(phase);
            sin_table[k][n] = std::sin(phase);
        }
    }

    tables_ready = true;
}

std::vector<float>
Spectrogram::normalize_audio(
    const std::vector<float>& input)
{
    std::vector<float> out = input;

    if (out.empty()) {
        return out;
    }

    double sum = 0.0;

    for (float v : out) {
        sum += v;
    }

    float mean = sum / out.size();

    float peak = 0.0f;

    for (float& v : out) {
        v -= mean;
        peak = std::max(peak, std::abs(v));
    }

    if (peak > 0.0f) {
        for (float& v : out) {
            v /= peak;
        }
    }

    return out;
}

std::vector<float>
Spectrogram::get_spectrogram(
    const std::vector<float>& input,
    bool normalize)
{
    init_tables();

    std::vector<float> audio(
        EXPECTED_SAMPLES,
        0.0f);

    if (input.size() >= EXPECTED_SAMPLES) {

        std::memcpy(
            audio.data(),
            input.data() +
            (input.size() - EXPECTED_SAMPLES),
            EXPECTED_SAMPLES * sizeof(float));

    } else if (!input.empty()) {

        std::memcpy(
            audio.data(),
            input.data(),
            input.size() * sizeof(float));
    }

    if (normalize) {
        audio = normalize_audio(audio);
    }

    std::vector<float> spec(
        FRAME_COUNT * POOLED_BINS);

    for (int frame = 0; frame < FRAME_COUNT; ++frame) {

        int frame_start = frame * HOP_LENGTH;

        for (int pooled = 0;
             pooled < POOLED_BINS;
             ++pooled)
        {
            double pooled_power = 0.0;

            for (int offset = 0;
                 offset < FREQ_POOL_SIZE;
                 ++offset)
            {
                int k =
                    pooled * FREQ_POOL_SIZE + offset;

                double real = 0.0;
                double imag = 0.0;

                for (int n = 0;
                     n < WINDOW_SIZE;
                     ++n)
                {
                    double sample =
                        audio[frame_start + n] *
                        hann_window[n];

                    real +=
                        sample *
                        cos_table[k][n];

                    imag -=
                        sample *
                        sin_table[k][n];
                }

                pooled_power +=
                    real * real +
                    imag * imag;
            }

            float mean_power =
                pooled_power / FREQ_POOL_SIZE;

            spec[
                frame * POOLED_BINS + pooled
            ] = std::log10(
                    mean_power +
                    LOG_EPSILON);
        }
    }

    return spec;
}