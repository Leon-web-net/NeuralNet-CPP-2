#include "commands.hpp"
#include "metrics.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
	std::string model_path(const std::string& name) {
		return std::string(MODELS_DIR) + "/" + name + ".bin";
	}
}

void run_train(const mnist::Dataset& data, std::vector<std::size_t> train_idx,
	const std::vector<std::size_t>& val_idx, const Options& opt) {
	const TrainConfig& cfg = opt.cfg;

	std::string run_label = cfg.backend;
	if (val_idx.empty()) run_label += "_full";

	std::optional<RunLogger> logger;
	if (opt.logging) {
		const std::string run_id = make_run_id(run_label);
		logger.emplace(LOGS_DIR, run_id, cfg.cpu_peak.gflops());
		std::cout << "logging run: " << run_id << " (" << BUILD_CONFIG << ")\n";
	}
	else {
		std::cout << "logging disabled\n";
	}

	Network net(cfg.layer_sizes, cfg.activation);
	std::cout << "model: " << opt.model_name << "  lr=" << cfg.lr_start
		<< "  batch=" << cfg.batch << "  epochs=" << cfg.epochs << "\n";
	train(net, data, std::move(train_idx), val_idx, cfg, logger ? &*logger : nullptr);

	if (logger) logger->write_summary(run_label, opt.model_name, cfg.batch, cfg.lr_start);

	if (opt.save) {
		std::filesystem::create_directories(MODELS_DIR);
		const std::string path = model_path(opt.model_name);
		net.save(path);
		std::cout << "saved model: " << path << "\n";
	}
}

void run_eval(const mnist::Dataset& data, const std::vector<std::size_t>& val_idx,
	const Options& opt) {
	Network net(opt.cfg.layer_sizes);
	net.load(model_path(opt.model_name));

	const Matrix X = data.images.gather_cols(val_idx);
	const Matrix Y = data.labels.gather_cols(val_idx);
	const Matrix pred = net.forward(X).back();

	std::cout << "model " << opt.model_name << " on " << val_idx.size()
		<< " validation images:  acc=" << accuracy(pred, Y) * 100.0f
		<< "%  loss=" << Network::cross_entropy_loss(pred, Y) << "\n";
}

void run_predict(const Options& opt) {
	const mnist::Dataset test =
		mnist::load(std::string(DATA_DIR) + "/test.csv", /*has_labels=*/false);

	Network net(opt.cfg.layer_sizes);
	net.load(model_path(opt.model_name));
	const Matrix digits = net.forward(test.images).back().argmax_cols();   // 1 x N

	std::filesystem::create_directories(SUBMISSIONS_DIR);
	const std::string path =
		std::string(SUBMISSIONS_DIR) + "/" + opt.model_name + "_submission.csv";
	std::ofstream out(path);
	if (!out) throw std::runtime_error("predict: could not open " + path);

	out << "ImageId,Label\n";
	for (std::size_t j = 0; j < digits.cols(); ++j) {
		out << (j + 1) << ',' << static_cast<int>(digits(0, j)) << '\n';   // Kaggle ids start at 1
	}
	std::cout << "wrote " << digits.cols() << " predictions to " << path << "\n";
}
