#pragma once
#include "matrix.hpp"
#include <string>
#include <cstddef>

namespace mnist {

	constexpr std::size_t IMAGE_SIZE = 784;
	constexpr std::size_t NUM_CLASSES = 10;
	constexpr std::size_t IMAGE_WIDTH = 28;


	struct Dataset {
		Matrix images;
		Matrix labels;
		std::size_t count;
	};

	Dataset load(const std::string& path);
}