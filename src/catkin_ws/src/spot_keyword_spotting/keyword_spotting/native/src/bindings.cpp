#include <algorithm>
#include <cstring>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ringbuffer.hpp"
#include "spectrogram.hpp"

namespace py = pybind11;

static py::array_t<float> vector_to_array(const std::vector<float>& values)
{
    py::array_t<float> out(values.size());
    auto buf = out.mutable_unchecked<1>();
    std::copy(values.begin(), values.end(), buf.mutable_data(0));
    return out;
}

static py::array_t<float> vector_to_spectrogram(const std::vector<float>& values)
{
    py::array_t<float> out({99, 26});
    auto buf = out.mutable_unchecked<2>();
    std::memcpy(buf.mutable_data(0, 0), values.data(), values.size() * sizeof(float));
    return out;
}

PYBIND11_MODULE(kws_native, m)
{
    m.doc() = "DSP backend for keyword spotting";

    py::class_<RingBuffer>(m, "RingBuffer")
        .def(py::init<int>(), py::arg("size") = 16000)
        .def("reset", &RingBuffer::reset)
        .def("is_full", &RingBuffer::is_full)
        .def("samples_seen", &RingBuffer::samples_seen)
        .def("size", &RingBuffer::size)
        .def(
            "push",
            [](RingBuffer& rb, py::array_t<float, py::array::c_style | py::array::forcecast> samples) {
                auto buf = samples.unchecked<1>();
                rb.push(buf.data(0), static_cast<int>(buf.shape(0)));
            },
            py::arg("samples"))
        .def(
            "get",
            [](RingBuffer& rb, bool pad_left) {
                return vector_to_array(rb.get(pad_left));
            },
            py::arg("pad_left") = true)
        .def("current_audio", [](RingBuffer& rb) {
            return vector_to_array(rb.current_audio());
        });

    m.def(
        "normalize_audio",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> audio) {
            auto buf = audio.unchecked<1>();
            std::vector<float> input(buf.data(0), buf.data(0) + buf.shape(0));
            return vector_to_array(Spectrogram::normalize_audio(input));
        },
        py::arg("audio"));

    m.def(
        "get_spectrogram",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> audio, bool normalize) {
            auto buf = audio.unchecked<1>();
            std::vector<float> input(buf.data(0), buf.data(0) + buf.shape(0));
            return vector_to_spectrogram(Spectrogram::get_spectrogram(input, normalize));
        },
        py::arg("audio"),
        py::arg("normalize") = true);

    m.def(
        "to_model_input",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> spectrogram) {
            auto buf = spectrogram.unchecked<2>();
            py::array_t<float> out({1, buf.shape(0), buf.shape(1), 1});
            std::memcpy(out.mutable_data(), buf.data(0, 0), static_cast<size_t>(buf.size()) * sizeof(float));
            return out;
        },
        py::arg("spectrogram"));

    m.attr("EXPECTED_SAMPLES") = 16000;
    m.attr("FRAME_COUNT") = 99;
    m.attr("POOLED_BINS") = 26;
    m.attr("WINDOW_SIZE") = 320;
    m.attr("HOP_LENGTH") = 160;
    m.attr("SAMPLE_RATE") = 16000;
}
