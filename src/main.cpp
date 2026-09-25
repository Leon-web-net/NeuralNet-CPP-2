#include "network.hpp"
#include "mnist.hpp"
#include "matrix.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

constexpr std::size_t EPOCHS = 25;
constexpr std::size_t BATCH = 64;
constexpr float LR_START = 0.5f;
constexpr float LR_DECAY = 0.9f;
constexpr std::size_t DECAY_EVERY = 5; // lr*= 0.9 at epochs 5, 10, 15 ...
constexpr float TRAIN_FRAC = 0.8f;


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

float accuracy(const Matrix& pred, const Matrix& labels) {
	const Matrix p = pred.argmax_cols();
	const Matrix t = labels.argmax_cols();
	std::size_t correct = 0;
	for (std::size_t j = 0; j < p.cols(); ++j) {
		if (p(0, j) == t(0, j)) ++correct;
	}

	return static_cast<float>(correct) / static_cast<float>(p.cols());
}

int main() {

	//dataset_load_speed();
	try {
		mnist::Dataset data = mnist::load(std::string(DATA_DIR) + "/train.csv");
		const std::size_t N = data.count;

		// shuffle all sample indices once (fixed seed), and 80/20 split
		std::vector<std::size_t> idx(N);
		std::iota(idx.begin(), idx.end(), std::size_t{ 0 });
		std::mt19937 gen(123);
		std::shuffle(idx.begin(), idx.end(), gen);

		const std::size_t n_train = static_cast<std::size_t>(TRAIN_FRAC * static_cast<float>(N));
		std::vector<std::size_t> train_idx(idx.begin(), idx.begin() + n_train);
		std::vector<std::size_t> val_idx(idx.begin() + n_train, idx.end());

		const Matrix X_val = data.images.gather_cols(val_idx);
		const Matrix Y_val = data.labels.gather_cols(val_idx);
		std::cout << "train: " << n_train << " val: " << val_idx.size() << "\n";

		Network net;
		float lr = LR_START;

		for (std::size_t epoch = 1; epoch <= EPOCHS; ++epoch) {
			if (epoch % DECAY_EVERY == 0) lr *= LR_DECAY;

			std::shuffle(train_idx.begin(), train_idx.end(), gen);
			const auto t0 = std::chrono::steady_clock::now();

			float loss_sum = 0.0f;
			std::size_t n_batches = 0;

			for (std::size_t start = 0; start < n_train; start += BATCH) {
				const std::size_t end = std::min(start + BATCH, n_train);
				const std::vector<std::size_t> b_idx(train_idx.begin() + start,
					train_idx.begin() + end);

				const Matrix Xb = data.images.gather_cols(b_idx);
				const Matrix Yb = data.labels.gather_cols(b_idx);

				std::vector<Matrix> acts = net.forward(Xb);
				loss_sum += Network::cross_entropy_loss(acts.back(), Yb);
				net.update(net.backward(acts, Yb), lr);
				++n_batches;
			}

			const auto t1 = std::chrono::steady_clock::now();
			const double secs = std::chrono::duration<double>(t1 - t0).count();
			const float val_acc = accuracy(net.forward(X_val).back(), Y_val);

			std::cout << "epoch " << epoch
				<< "  lr=" << lr
				<< "  loss=" << loss_sum / static_cast<float>(n_batches)
				<< "  val_acc=" << val_acc * 100.0f << "%"
				<< "  (" << secs << " s)" << std::endl;
		}
	}
	catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << "\n";
		return 1;
	}

	return 0;
}