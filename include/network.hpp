#pragma once
#include "matrix.hpp"
#include "mnist.hpp"
#include <vector>
#include <cstddef>

constexpr std::size_t HIDDEN1 = 128;
constexpr std::size_t HIDDEN2 =  64;

struct Layer {
	Matrix weights; // n_out x n_in
	Matrix biases; // n_out x 1
	Layer(std::size_t n_in, std::size_t n_out)
		:weights(n_out, n_in), biases(n_out,1){}
};

class Network {
public:
	Network();  // build network and init weights

	
	// TODO: forward, loss, backward, train, save, load
	void print_weights_stats() const;

	std::vector<Matrix> forward(const Matrix& input)const;
	float cross_entropy_loss(const Matrix& pred, const Matrix& labels);

private:
	std::vector<Layer>layers_;
	void init_weights();
};