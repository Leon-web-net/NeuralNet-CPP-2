#include "matrix.hpp"
#include "mnist.hpp"
#include <iostream>
#include <chrono>
#include "network.hpp"

static void dataset_load_speed() {
	auto t0 = std::chrono::steady_clock::now();
	mnist::Dataset data = mnist::load(std::string(DATA_DIR) + "/train.csv");
	auto t1 = std::chrono::steady_clock::now();
	std::cout << "load took "
		<< std::chrono::duration<double, std::milli>(t1 - t0).count()
		<< " ms\n";

	std::cout << "images: " << data.images.rows() << " X " << data.images.cols() << "\n";
	std::cout << "labels: " << data.labels.rows() << " X " << data.images.cols() << "\n";
	std::cout << "samples: " << data.count << "\n";

	// one hot encode
	float col0_sum = 0.0f;
	for (std::size_t k = 0; k < data.labels.rows();++k) {
		col0_sum += data.labels(k, 0);
	}

	std::cout << "label[0] one hot sum = " << col0_sum << "\n";

	// Normalise pixels [0 1]
	float pmin = data.images(0, 0), pmax = pmin;
	for (std::size_t f = 0; f < data.images.rows(); ++f) {
		float v = data.images(f, 0);
		if (v < pmin) pmin = v;
		if (v > pmax) pmax = v;
	}
	std::cout << "image[0] pixel range: [" << pmin << ", " << pmax << "]\n";
}


// Which digit is column s? Find the row holding the 1 in the one-hot labels.
std::size_t true_label(const Matrix& labels, std::size_t s) {
	for (std::size_t k = 0; k < labels.rows(); ++k)
		if (labels(k, s) == 1.0f) return k;
	return labels.rows();   // shouldn't happen: means no 1 was found
}


int main() {
	
	//dataset_load_speed();
	mnist::Dataset data = mnist::load(std::string(DATA_DIR) + "/train.csv");
	Network net;
	net.print_weights_stats();

	const std::size_t B = 5;
	Matrix batch(mnist::IMAGE_SIZE, B);
	for (std::size_t f = 0; f < mnist::IMAGE_SIZE; ++f)
		for (std::size_t s = 0; s < B; ++s)
			batch(f, s) = data.images(f, s);

	Matrix batch_labels(mnist::NUM_CLASSES, B);
	for (std::size_t k = 0; k < mnist::NUM_CLASSES; ++k) {
		for (std::size_t s = 0; s < B; ++s) {
			batch_labels(k, s) = data.labels(k, s);
		}
	}

	std::vector<Matrix> acts = net.forward(batch);
	const Matrix& out = acts.back();

	
	std::cout << "loss= " << net.cross_entropy_loss(out, batch_labels) << "\n";

	return 0;
}