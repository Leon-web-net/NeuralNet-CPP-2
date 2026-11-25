#include "trainer.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <string>

void train(Network& net, const mnist::Dataset& data, std::vector<std::size_t> train_idx,
	const std::vector<std::size_t>& val_idx, const TrainConfig& cfg, RunLogger* logger) {
	const bool has_val = !val_idx.empty();
	const Matrix X_val = data.images.gather_cols(val_idx);
	const Matrix Y_val = data.labels.gather_cols(val_idx);

	const std::size_t n_train = train_idx.size();
	const double peak_gflops = cfg.cpu_peak.gflops();
	constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

	// Work per epoch is fixed by the network shape and dataset size: count it once.
	const double epoch_flops = net.train_flops_per_sample() * static_cast<double>(n_train);

	std::cout << "training on " << n_train << " images, "
		<< (has_val ? "validating on " + std::to_string(val_idx.size())
			: std::string("no validation set")) << "\n"
		<< "work: " << net.train_flops_per_sample() / 1e3 << " kFLOP/image, "
		<< epoch_flops / 1e9 << " GFLOP/epoch\n"
		<< "peak: " << peak_gflops << " GFLOP/s chip ("
		<< cfg.cpu_peak.gflops_per_core() << " per core)\n";

	std::mt19937 gen(cfg.shuffle_seed);
	float lr = cfg.lr_start;

	for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
		if (epoch % cfg.decay_every == 0) lr *= cfg.lr_decay;

		std::shuffle(train_idx.begin(), train_idx.end(), gen);
		const auto t0 = std::chrono::steady_clock::now();

		float loss_sum = 0.0f;
		std::size_t n_batches = 0;

		for (std::size_t start = 0; start < n_train; start += cfg.batch) {
			const std::size_t end = std::min(start + cfg.batch, n_train);
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
		const float train_loss = loss_sum / static_cast<float>(n_batches);
		const double images_per_sec = static_cast<double>(n_train) / secs;
		const double gflops = epoch_flops / secs / 1e9;
		const double mfu_pct = 100.0 * gflops / peak_gflops;

		float val_loss = NaN, val_acc = NaN;
		if (has_val) {
			const Matrix val_pred = net.forward(X_val).back();
			val_acc = accuracy(val_pred, Y_val);
			val_loss = Network::cross_entropy_loss(val_pred, Y_val);
		}

		if (logger) logger->log_epoch({ epoch, lr, train_loss, val_loss, val_acc, secs,
										images_per_sec, gflops, mfu_pct });

		std::cout << "epoch " << epoch << "  lr=" << lr << "  train_loss=" << train_loss;
		if (has_val) {
			std::cout << "  val_loss=" << val_loss << "  val_acc=" << val_acc * 100.0f << "%";
		}
		std::cout << "  (" << secs << " s, " << static_cast<long long>(images_per_sec)
			<< " img/s, " << gflops << " GFLOP/s, MFU " << mfu_pct << "%)" << std::endl;
	}
}
