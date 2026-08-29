#include "matrix.hpp"
#include "mnist.hpp"
#include <iostream>
#include <chrono>

int main() {
	
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

	return 0;
}