#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Column indices into a Dataset, split into training and validation.
struct Split {
	std::vector<std::size_t> train;
	std::vector<std::size_t> val;
};

// Same seed -> same split every time, in every mode.
Split make_split(std::size_t n, float train_frac, std::uint32_t seed);

// 0, 1, ..., n-1
std::vector<std::size_t> all_indices(std::size_t n);
