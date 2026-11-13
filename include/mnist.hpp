#pragma once
#include "matrix.hpp"
#include <string>
#include <cstddef>

namespace mnist {

	constexpr std::size_t IMAGE_WIDTH = 28;
	constexpr std::size_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH;
	constexpr std::size_t NUM_CLASSES = 10;


	struct Dataset {
		Matrix images;
		Matrix labels;
		std::size_t count;
		bool has_labels;
	};
	
	// train.csv: label column then 784 pixels  -> has_labels = true (default)
	// test.csv:  784 pixels only               -> has_labels = false
	Dataset load(const std::string& path, bool has_labels = true);
}