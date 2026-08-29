#include "network.hpp"
#include <random>
#include <cmath>

Network::Network() {
	layers_.emplace_back(mnist::IMAGE_SIZE, HIDDEN1);
	layers_.emplace_back(HIDDEN1, HIDDEN2);
	layers_.emplace_back(HIDDEN2, mnist::NUM_CLASSES);
	init_weights();
}

void Network::init_weights() {
	std::mt19937 gen(42);
	const std::size_t n_in = layer.weights.cols();
	const std::size_t n_out = layer.weights.rows();

	// Xavier standard deviation from n_in and n_out
}