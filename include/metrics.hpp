#pragma once
#include "matrix.hpp"
#include <string>
#include <string_view>
#include <fstream>
#include <cstddef>

struct EpochMetrics {
	std::size_t epoch;
	float lr;
	float train_loss;
	float val_loss;
	float val_acc;
	double train_secs;
	double images_per_sec;
	double gflops;      // achieved GFLOP/s during training (analytic FLOPs / measured time)
	double mfu_pct;     // gflops as % of hardware peak (model FLOPs utilisation)
};

// Theoretical peak = cores x clock x FLOPs per cycle per core.
struct HardwarePeak {
	double cores;
	double ghz;
	double flops_per_cycle;
	double gflops() const { return cores * ghz * flops_per_cycle; }
	double gflops_per_core() const { return ghz * flops_per_cycle; }
};

// Fraction of columns where argmax(pred) == argmax(labels).
float accuracy(const Matrix& pred, const Matrix& labels);

std::string make_run_id(std::string_view backend);

class RunLogger {
public:
	RunLogger(const std::string& log_dir, const std::string& run_id, double peak_gflops);

	void log_epoch(const EpochMetrics& m);

	void write_summary(std::string_view backend, std::string_view model,
		std::size_t batch, float lr_start) const;

private:
	std::string log_dir_;
	std::string run_id_;
	std::ofstream epoch_file_;
	double peak_gflops_;

	std::size_t n_epochs_ = 0;
	double total_secs_ = 0.0;
	double total_gflop_ = 0.0;    // work done, so run-level rates are total / total
	double total_images_ = 0.0;
	float last_acc_ = 0.0f;
	float best_acc_ = 0.0f;

};
