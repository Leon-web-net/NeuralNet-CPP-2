#include "network.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>
#include <fstream>
#include <cstdint>
// static (internal linkage) only visible in this .cpp file

Network::Network(const std::vector <std::size_t>& sizes) {
	if (sizes.size() < 2) {
		throw std::invalid_argument("Network: need at least input and output sizes");
	}
	for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
		layers_.emplace_back(sizes[i], sizes[i + 1]);
	}
	init_weights();
}

void Network::init_weights() {
	std::mt19937 gen(42);

	for (Layer& layer : layers_) {

		// Xavier standard deviation from n_in and n_out
		const std::size_t n_in = layer.weights.cols();
		const std::size_t n_out = layer.weights.rows();

		const float fan = static_cast<float>(n_in + n_out);
		const float stddev = std::sqrt(2.0f / fan);
		std::normal_distribution<float> dist(0.0f, stddev);

		for (std::size_t r = 0; r < n_out;++r) {
			for (std::size_t c = 0; c < n_in;++c) {
				layer.weights(r, c) = dist(gen);
			}
		} // biases are initialized to zero
	}

}

void Network::print_weights_stats() const {
	for (std::size_t li = 0; li < layers_.size(); ++li) {
		const Matrix& w = layers_[li].weights;
		const std::size_t n = w.rows() * w.cols();

		float sum = 0.0f, mn = std::numeric_limits<float>::max(),
			mx = std::numeric_limits<float>::lowest();
		
		for (std::size_t r = 0; r < w.rows(); ++r) {
			for (std::size_t c = 0; c < w.cols(); ++c) {
				float v = w(r, c);
				sum += v;
				mn = (v < mn) ? v : mn;
				mx = (v > mx) ? v : mx;
			}
		}

		const float mean = sum / static_cast<float>(n);

		float sq = 0.0f;
		for (std::size_t r = 0; r < w.rows(); ++r) {
			for (std::size_t c = 0; c < w.cols(); ++c) {
				float d = w(r, c) - mean;
				sq += d * d;
			}
		}

		const float stddev = std::sqrt(sq / static_cast<float>(n));

		std::cout << "layer "<< li <<" ("<<w.rows()<<"x"<<w.cols()<<"):  "
			<<"mean= "<<mean<< ", stddev="<<stddev
			<< " min=" << mn << ", max=" << mx << "\n";
	}

}

static float sigmoid(float z) {
	return 1.0f / (1.0f + std::exp(-z));
}

// applies softmax function to every column of a matrix z
static Matrix softmax_cols(const Matrix& z) {
	Matrix result(z.rows(), z.cols());

	for (std::size_t j = 0; j < z.cols(); ++j) {
		float col_max = z(0, j);
		for (std::size_t i = 1; i < z.rows(); ++i) {
			col_max = (z(i, j) > col_max) ? z(i, j) : col_max;
		}

		float sum = 0.0f;
		for (std::size_t i = 0; i < z.rows(); ++i) {
			float e = std::exp(z(i, j) - col_max);
			result(i, j) = e;
			sum += e;
		}

		for (std::size_t i = 0; i < z.rows(); ++i) {
			result(i, j) /= sum;
		}

	}

	return result;
}

std::vector<Matrix> Network::forward(const Matrix& input)const {
	std::vector<Matrix> activations;
	activations.reserve(layers_.size() + 1);
	activations.push_back(input);

	for (std::size_t li = 0; li < layers_.size(); ++li) {
		const Layer& layer = layers_[li];
		const Matrix& x = activations.back();

		Matrix z = layer.weights.matmul(x).add_col_vector(layer.biases);

		const bool is_output = (li + 1 == layers_.size());
		if (is_output) {
			activations.push_back(softmax_cols(z));
		}
		else {
			activations.push_back(z.apply(sigmoid)); // For RELU swap
		}
		
	}

	return activations;
}

float Network::cross_entropy_loss(const Matrix& pred, const Matrix& labels) {
	if( pred.rows() != labels.rows() || pred.cols() != labels.cols() ) {
		throw std::invalid_argument(
			"cross_entropy_loss: pred (" + std::to_string(pred.rows()) + "x" +
			std::to_string(pred.cols()) + ") vs labels (" +
			std::to_string(labels.rows()) + "x" + std::to_string(labels.cols()) +
			") - dimensions don't match");
	}
	constexpr float EPS = 1e-7f; // small value to avoid log(0)
	float loss = 0.0f;

	for (std::size_t i = 0; i < pred.rows();++i) {
		for (std::size_t j = 0; j < pred.cols();++j) {
			const float y = labels(i, j);
			if(y!=0.0f){
				loss -= y * std::log(pred(i, j) + EPS);
			}
		}
	}

	return loss / static_cast<float>(pred.cols());

}



std::vector<LayerGrad> Network::backward(const std::vector<Matrix>& activations,
										const Matrix& labels) const {
	const std::size_t L = layers_.size();
	const float B = static_cast<float>(labels.cols());

	std::vector<LayerGrad> grads;
	grads.reserve(L); // makes space for three layers, no reallocation
	// Intialise gradients of weights and biases
	for (const Layer& layer : layers_) {
		grads.push_back({ Matrix(layer.weights.rows(), layer.weights.cols()),
			Matrix(layer.biases.rows(), 1) });

	}

	// Delta = (P-Y)/B. P softmax predictions. P-Y: output error
	// average over the batch B.
	Matrix delta = activations.back().subtract(labels).scale(1.0f / B);
	
	for (std::size_t i = L; i > 0;--i) {
		const std::size_t l = i - 1;
		const Matrix& a_prev = activations[l];

		grads[l].dW = delta.matmul(a_prev.transpose()); // weight gradient, delta · a_prevᵀ
		grads[l].db = delta.row_sums(); // bias gradient, sum accross batch

		if (l > 0) {
			// if layer 0 has nothing before it, sigmoid slope at each unit
			Matrix sig_deriv = a_prev.apply(
				[](float a) {return a * (1.0f - a);});
			
			delta = layers_[l].weights.transpose() // send error back through W
				.matmul(delta)						// error at previous layer's output
				.hadamard(sig_deriv);				// through that layer's sigmoid

		}
	}

	return grads;
}

void Network::update(const std::vector<LayerGrad>& grads, float lr) {
	if (grads.size() != layers_.size()) {
		throw std::invalid_argument(
			"update: got " + std::to_string(grads.size()) +" gradients for " +
			std::to_string(layers_.size()) + " layers");
	}

	for (std::size_t l = 0; l < layers_.size(); ++l) {
		// W = W -lr·dW
		layers_[l].weights = layers_[l].weights.subtract(grads[l].dW.scale(lr));
		// b  = b − lr·db
		layers_[l].biases = layers_[l].biases.subtract(grads[l].db.scale(lr));
	}

}

// File format (all little-endian on x86):
//   uint32 magic, uint64 layer count,
//   then per layer: [rows, cols, floats...] for weights, then biases.
constexpr std::uint32_t MODEL_MAGIC = 0x4E4E4331;   // "NNC1": identifies our files

static void write_u64(std::ofstream& out, std::uint64_t v) {
	out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static std::uint16_t read_u64(std::ifstream& in) {
	std::uint64_t v = 0;
	in.read(reinterpret_cast<char*>(&v), sizeof(v));
	if (!in) throw std::runtime_error("load: file truncated");
	return v;
}

static void write_matrix(std::ofstream& out, const Matrix& m) {
	write_u64(out, m.rows());
	write_u64(out, m.cols());
	out.write(reinterpret_cast<const char*>(m.data()),
		static_cast<std::streamsize>(m.rows() * m.cols() * sizeof(float)));
}

static void read_matrix_into(std::ifstream& in, Matrix& m) {
	const std::uint64_t rows = read_u64(in);
	const std::uint64_t cols = read_u64(in);
	if (rows != m.rows() || cols != m.cols()) {
		throw std::runtime_error(
			"load: file has a (" + std::to_string(rows) + "x" + std::to_string(cols) +
			") matrix, network expects (" + std::to_string(m.rows()) + "x" +
			std::to_string(m.cols()) + ")");
	}

	in.read(reinterpret_cast<char*>(m.data()),
		static_cast<std::streamsize>(rows * cols * sizeof(float)));
	if (!in) throw std::runtime_error("load: file truncated");
}

void Network::save(const std::string& path) const {
	std::ofstream out(path, std::ios::binary);
	if (!out) throw std::runtime_error("save: could not open " + path);

	out.write(reinterpret_cast<const char*>(&MODEL_MAGIC), sizeof(MODEL_MAGIC));
	write_u64(out, layers_.size());

	for (const Layer& layer : layers_) {
		write_matrix(out, layer.weights);
		write_matrix(out, layer.biases);
	}
}

void Network::load(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) throw std::runtime_error("load: could not open " + path);

	std::uint32_t magic = 0;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!in || magic != MODEL_MAGIC) {
		throw std::runtime_error("load: " + path + " is not a model file");
	}

	const std::uint64_t n = read_u64(in);
	if (n != layers_.size()) {
		throw std::runtime_error("load: file has "+ std::to_string(n)+
			" layers, network has " + std::to_string(layers_.size()));
	}

	for (Layer& layer : layers_) {
		read_matrix_into(in, layer.weights);
		read_matrix_into(in, layer.biases);
	}
}