#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ringbuffer.hpp"
#include "spectrogram.hpp"

namespace py = pybind11;

PYBIND11_MODULE(kws_native, m)
{
    m.doc() =
        "DSP backend for keyword spotting";

    // RingBuffer
    py::class_<RingBuffer>(m, "RingBuffer")

        .def(
            py::init<int>(),
            py::arg("size") = 16000
        )

        .def(
            "reset",
            &RingBuffer::reset
        )

        .def(
            "is_full",
            &RingBuffer::is_full
        )

        .def(
            "samples_seen",
            &RingBuffer::samples_seen
        )

        .def(
            "size",
            &RingBuffer::size
        )

        .def(
            "push",

            [](RingBuffer& rb,

               py::array_t<
                   float,
                   py::array::c_style |
                   py::array::forcecast> samples)
        {
            auto buf = samples.unchecked<1>();

            rb.push(
                (float*)buf.data(0),
                (int)buf.shape(0)
            );
        }
        )

        .def(
            "get",

            [](RingBuffer& rb,
               bool pad_left)
        {
            auto out =
                rb.get(pad_left);

            return py::array_t<float>(
                out.size(),
                out.data()
            );
        },

        py::arg("pad_left") = true
        )

        .def(
            "current_audio",

            [](RingBuffer& rb)
        {
            auto out =
                rb.current_audio();

            return py::array_t<float>(
                out.size(),
                out.data()
            );
        }
        );

    // Spectrogram DSP
    m.def(
        "normalize_audio",

        [](py::array_t<
               float,
               py::array::c_style |
               py::array::forcecast> audio)
    {
        auto buf =
            audio.unchecked<1>();

        std::vector<float> input(
            buf.data(0),
            buf.data(0) + buf.shape(0)
        );

        auto out =
            Spectrogram::normalize_audio(
                input
            );

        return py::array_t<float>(
            out.size(),
            out.data()
        );
    },

    py::arg("audio")
    );

    m.def(
        "get_spectrogram",

        [](py::array_t<
               float,
               py::array::c_style |
               py::array::forcecast> audio,

           bool normalize)
    {
        auto buf =
            audio.unchecked<1>();

        std::vector<float> input(
            buf.data(0),
            buf.data(0) + buf.shape(0)
        );

        auto out =
            Spectrogram::get_spectrogram(
                input,
                normalize
            );

        return py::array_t<float>(
            {99, 26},
            out.data()
        );
    },

    py::arg("audio"),
    py::arg("normalize") = true
    );

    // Constants
    m.attr("EXPECTED_SAMPLES") = 16000;
    m.attr("FRAME_COUNT") = 99;
    m.attr("POOLED_BINS") = 26;
    m.attr("WINDOW_SIZE") = 320;
    m.attr("HOP_LENGTH") = 160;
}