#include "mnist.hpp"
#include <fstream>
#include <charconv>
#include <stdexcept>
#include <vector>

namespace mnist {

	constexpr float PIXEL_MAX = 255.0f;

	Dataset load(const std::string& path) {
		std::ifstream file(path);
		if (!file.is_open()) {
			throw std::runtime_error("Load_mnist: could not open" + path);

		}

		std::string line;
		std::getline(file, line); // Skip the header line

		std::vector<float> pixel_buffer;
		std::vector<int> label_buffer;

		while (std::getline(file, line)) {
			if (line.empty()) continue;
			
			const char* p = line.data();
			const char* end = p + line.size();

			int label = 0;
			auto lr = std::from_chars(p, end, label);
			if (lr.ec != std::errc()) {
				throw std::runtime_error("mnist::load: bad label in row");
			}
			label_buffer.push_back(label);
			p = lr.ptr;

			std::size_t pixels_read = 0;
			while (p < end) {
				if (*p == ',') ++p;
				float value = 0.0f;
				auto pr = std::from_chars(p, end, value);
				if (pr.ec != std::errc())break;
				pixel_buffer.push_back(value / PIXEL_MAX);
				++pixels_read;
				p = pr.ptr;
			}


			if (pixels_read != IMAGE_SIZE) {
				throw std::runtime_error(
					"Load_mnist: expected " + std::to_string(IMAGE_SIZE) +
					" pixels, but got " + std::to_string(pixels_read));
			}

		}

		const std::size_t N = label_buffer.size();
		Matrix images(IMAGE_SIZE, N);
		Matrix labels(NUM_CLASSES, N);

		for (std::size_t s = 0; s < N; ++s) {
			for (std::size_t f = 0; f < IMAGE_SIZE; ++f) {
				images(f, s) = pixel_buffer[s * IMAGE_SIZE + f];
			}
			const int lbl = label_buffer[s];
			if (lbl < 0 || lbl >= static_cast<int>(NUM_CLASSES)) {
				throw std::runtime_error(
					"mnist::load: label out of range: " + std::to_string(lbl)
				);
			}
			labels(static_cast<std::size_t>(lbl), s) = 1.0f; // one hot encoded
		}

		return Dataset{ std::move(images), std::move(labels),N};

	}

}