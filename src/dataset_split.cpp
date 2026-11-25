#include "dataset_split.hpp"
#include <algorithm>
#include <numeric>
#include <random>

std::vector<std::size_t> all_indices(std::size_t n) {
	std::vector<std::size_t> idx(n);
	std::iota(idx.begin(), idx.end(), std::size_t{ 0 });
	return idx;
}

Split make_split(std::size_t n, float train_frac, std::uint32_t seed) {
	std::vector<std::size_t> idx = all_indices(n);
	std::mt19937 gen(seed);
	std::shuffle(idx.begin(), idx.end(), gen);

	const auto n_train = static_cast<std::size_t>(train_frac * static_cast<float>(n));
	Split s;
	s.train.assign(idx.begin(), idx.begin() + n_train);
	s.val.assign(idx.begin() + n_train, idx.end());
	return s;
}
