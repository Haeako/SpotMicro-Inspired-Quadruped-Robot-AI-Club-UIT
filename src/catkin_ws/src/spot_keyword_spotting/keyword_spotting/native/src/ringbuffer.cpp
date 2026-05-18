#pragma once

#include <vector>
#include <algorithm>
#include <cstring>

class RingBuffer {
public:
    explicit RingBuffer(int size = 16000)
        : buffer_(size, 0.0f),
          size_(size),
          write_index_(0),
          samples_seen_(0)
    {
    }

    void reset()
    {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
        write_index_ = 0;
        samples_seen_ = 0;
    }

    void push(const float* samples, int count)
    {
        if (samples == nullptr || count <= 0) {
            return;
        }

        // Nếu input lớn hơn buffer
        // chỉ giữ phần cuối cùng
        if (count >= size_) {
            std::memcpy(
                buffer_.data(),
                samples + (count - size_),
                size_ * sizeof(float));

            write_index_ = 0;
            samples_seen_ += count;
            return;
        }

        int end = write_index_ + count;

        // Không wrap
        if (end <= size_) {
            std::memcpy(
                buffer_.data() + write_index_,
                samples,
                count * sizeof(float));
        }
        // Wrap around
        else {
            int first = size_ - write_index_;

            std::memcpy(
                buffer_.data() + write_index_,
                samples,
                first * sizeof(float));

            std::memcpy(
                buffer_.data(),
                samples + first,
                (count - first) * sizeof(float));
        }

        write_index_ = end % size_;
        samples_seen_ += count;
    }

    std::vector<float> get(bool pad_left = true) const
    {
        // Buffer đầy
        if (samples_seen_ >= size_) {

            std::vector<float> out(size_);

            int right_count = size_ - write_index_;

            std::memcpy(
                out.data(),
                buffer_.data() + write_index_,
                right_count * sizeof(float));

            std::memcpy(
                out.data() + right_count,
                buffer_.data(),
                write_index_ * sizeof(float));

            return out;
        }

        // Buffer chưa đầy
        int valid_count = write_index_;

        if (!pad_left) {
            return std::vector<float>(
                buffer_.begin(),
                buffer_.begin() + valid_count);
        }

        std::vector<float> out(size_, 0.0f);

        int pad_count = size_ - valid_count;

        std::memcpy(
            out.data() + pad_count,
            buffer_.data(),
            valid_count * sizeof(float));

        return out;
    }

    std::vector<float> current_audio() const
    {
        return get(true);
    }

    bool is_full() const
    {
        return samples_seen_ >= size_;
    }

    long long samples_seen() const
    {
        return samples_seen_;
    }

    int size() const
    {
        return size_;
    }

private:
    std::vector<float> buffer_;

    int size_;
    int write_index_;

    long long samples_seen_;
};