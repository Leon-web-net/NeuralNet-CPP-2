#include "mnist.hpp"
#include <fstream>
#include <sstream>
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
			std::stringstream ss(line);
			std::string cell;

			std::getline(ss, cell, ',');
			label_buffer.push_back(std::stoi(cell));

			std::size_t pixels_read = 0;
			while (std::getline(ss, cell, ',')) {
				pixel_buffer.push_back(std::stof(cell) / PIXEL_MAX);
				++pixels_read;
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