#pragma once
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

};

std::string make_run_id(std::string_view bakcend);

class RunLogger {
public:
	RunLogger(const std::string& log_dir, const std::string& run_id);

	void log_epoch(const EpochMetrics& m);

	void write_summary(std::string_view backend, std::size_t batch, float lr_start) const;

private:
	std::string log_dir_;
	std::string run_id_;
	std::ofstream epoch_file_;

	std::size_t n_epochs_ = 0;
	double total_secs_ = 0.0;
	float last_acc_ = 0.0f;
	float best_acc_ = 0.0f;

};