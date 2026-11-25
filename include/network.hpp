#pragma once
#include "matrix.hpp"
#include "mnist.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>

constexpr std::size_t HIDDEN1 = 256;
constexpr std::size_t HIDDEN2 = 128;

// Hidden-layer activation. Values are stored in model files, so don't renumber.
enum class Activation : std::uint32_t { Sigmoid = 0, ReLU = 1 };
const char* to_string(Activation a);

struct Layer {
	Matrix weights; // n_out x n_in
	Matrix biases; // n_out x 1
	Layer(std::size_t n_in, std::size_t n_out)
		:weights(n_out, n_in), biases(n_out,1){}
};

struct LayerGrad {
	Matrix dW;
	Matrix db;
};

class Network {
public:
	explicit Network(const std::vector<std::size_t>& sizes =
		{ mnist::IMAGE_SIZE, HIDDEN1, HIDDEN2,mnist::NUM_CLASSES },
		Activation act = Activation::ReLU);

	Layer& layer(std::size_t l) {return layers_.at(l); }
	std::size_t num_layers() const { return layers_.size(); }
	Activation activation() const { return act_; }
	
	
	// TODO: forward, loss, backward, train, save, load
	void print_weights_stats() const;

	// Analytic matmul FLOP counts per image ("model FLOPs"). A (m x k)(k x n) matmul
	// costs 2mkn. Fixed by the layer shapes, so compute once, not per batch.
	double forward_flops_per_sample() const;   // W·a for every layer
	double train_flops_per_sample() const;     // forward + dW + Wᵀ·delta (skipped for layer 0)

	std::vector<Matrix> forward(const Matrix& input)const;
	static float cross_entropy_loss(const Matrix& pred, const Matrix& labels);

	std::vector<LayerGrad> backward(const std::vector<Matrix>& activations,
		const Matrix& labels) const;
	void update(const std::vector<LayerGrad>& grads, float lr);

	// const at end means will not modify any member
	void save(const std::string& path) const; // read
	void load(const std::string& path); // Write; also adopts the file's activation

private:
	std::vector<Layer>layers_;
	Activation act_;
	void init_weights();
};