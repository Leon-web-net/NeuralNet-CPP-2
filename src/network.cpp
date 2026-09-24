#include "network.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <limits>

// static (internal linkage) only visible in this .cpp file

Network::Network() {
	layers_.emplace_back(mnist::IMAGE_SIZE, HIDDEN1);
	layers_.emplace_back(HIDDEN1, HIDDEN2);
	layers_.emplace_back(HIDDEN2, mnist::NUM_CLASSES);
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

