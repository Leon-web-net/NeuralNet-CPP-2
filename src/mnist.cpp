#include "mnist.hpp"
#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mnist {

    constexpr float PIXEL_MAX = 255.0f;   // 8-bit grayscale -> [0,1]

    Dataset load(const std::string& path, bool has_labels) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("mnist::load: could not open " + path);
        }

        std::string line;
        std::getline(file, line);   // skip the header row

        std::vector<float> pixel_buffer;   // sample-major: [s0_p0..s0_p783, s1_p0..]
        std::vector<int>   label_buffer;   // stays empty when has_labels is false
        std::size_t rows_read = 0;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            const std::size_t row = rows_read + 1;   // 1-based, for error messages
            const char* p = line.data();
            const char* end = p + line.size();

            // --- label (train.csv only) ---
            if (has_labels) {
                int label = 0;
                auto lr = std::from_chars(p, end, label);
                if (lr.ec != std::errc()) {
                    throw std::runtime_error(
                        "mnist::load: bad label in data row " + std::to_string(row));
                }
                if (label < 0 || label >= static_cast<int>(NUM_CLASSES)) {
                    throw std::runtime_error(
                        "mnist::load: label " + std::to_string(label) +
                        " out of range in data row " + std::to_string(row));
                }
                label_buffer.push_back(label);
                p = lr.ptr;
            }

            // --- pixels ---
            std::size_t pixels_read = 0;
            while (p < end) {
                if (*p == ',') ++p;                   // step over the delimiter
                float value = 0.0f;
                auto pr = std::from_chars(p, end, value);
                if (pr.ec != std::errc()) break;      // trailing '\r' or end of line
                pixel_buffer.push_back(value / PIXEL_MAX);
                ++pixels_read;
                p = pr.ptr;
            }

            if (pixels_read != IMAGE_SIZE) {
                throw std::runtime_error(
                    "mnist::load: expected " + std::to_string(IMAGE_SIZE) +
                    " pixels in data row " + std::to_string(row) +
                    ", got " + std::to_string(pixels_read) +
                    (has_labels ? "" : " (is this a file with labels?)"));
            }

            ++rows_read;
        }

        // --- assemble matrices: one column per sample ---
        const std::size_t N = rows_read;
        Matrix images(IMAGE_SIZE, N);
        Matrix labels(has_labels ? NUM_CLASSES : 0, N);   // zero-filled

        for (std::size_t s = 0; s < N; ++s) {
            for (std::size_t f = 0; f < IMAGE_SIZE; ++f) {
                images(f, s) = pixel_buffer[s * IMAGE_SIZE + f];
            }
            if (has_labels) {
                labels(static_cast<std::size_t>(label_buffer[s]), s) = 1.0f;   // one-hot
            }
        }

        return Dataset{ std::move(images), std::move(labels), N, has_labels };
    }

}  