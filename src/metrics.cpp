#include "metrics.hpp"
#include <chrono>
#include <format>
#include <filesystem>
#include <stdexcept>

namespace {
	constexpr std::string_view EPOCH_HEADER =
		"epoch,lr,train_loss,val_loss,val_acc,train_secs,images_per_sec,gflops,mfu_pct";

	constexpr std::string_view RUNS_HEADER =
		"run_id,build,backend,model,epochs,batch,lr_start,"
		"final_val_acc,best_val_acc,mean_epoch_secs,total_train_secs,"
		"mean_images_per_sec,mean_gflops,peak_gflops,mfu_pct";

	// If runs.csv was written with a different column set, move it aside so one
	// file never mixes row shapes.
	void rotate_if_stale(const std::string& path, std::string_view run_id) {
		std::string first_line;
		{
			std::ifstream in(path);
			if (!in) return;
			std::getline(in, first_line);
		}
		if (!first_line.empty() && first_line.back() == '\r') first_line.pop_back();
		if (first_line == RUNS_HEADER) return;

		const auto dot = path.rfind('.');
		const std::string legacy = path.substr(0, dot) + "_legacy_" + std::string(run_id) + ".csv";
		std::filesystem::rename(path, legacy);
	}
}

float accuracy(const Matrix& pred, const Matrix& labels) {
	if (pred.rows() != labels.rows() || pred.cols() != labels.cols()) {
		throw std::invalid_argument(
			"accuracy: pred (" + std::to_string(pred.rows()) + "x" +
			std::to_string(pred.cols()) + ") vs labels (" +
			std::to_string(labels.rows()) + "x" + std::to_string(labels.cols()) +
			") - dimensions don't match");
	}
	const Matrix p = pred.argmax_cols();
	const Matrix t = labels.argmax_cols();
	std::size_t correct = 0;
	for (std::size_t j = 0; j < p.cols(); ++j) {
		if (p(0, j) == t(0, j)) ++correct;
	}
	return static_cast<float>(correct) / static_cast<float>(p.cols());
}

std::string make_run_id(std::string_view backend) {
	const auto now = std::chrono::floor<std::chrono::seconds>(
		std::chrono::system_clock::now());
	return std::format("{:%Y%m%d-%H%M%S}_{}", now, backend);
}

RunLogger::RunLogger(const std::string& log_dir, const std::string& run_id, double peak_gflops)
	:log_dir_(log_dir), run_id_(run_id), peak_gflops_(peak_gflops) {
	std::filesystem::create_directories(log_dir_);

	const std::string path = log_dir_ + "/" + run_id_ + "_epochs.csv";
	epoch_file_.open(path);
	if(!epoch_file_) throw std::runtime_error("RunLogger: could not open " + path);

	epoch_file_ << EPOCH_HEADER << '\n';
}

void RunLogger::log_epoch(const EpochMetrics& m) {
	epoch_file_ << m.epoch << ',' << m.lr << ',' << m.train_loss << ','
		<< m.val_loss << ',' << m.val_acc << ',' << m.train_secs << ','
		<< m.images_per_sec << ',' << m.gflops << ',' << m.mfu_pct << '\n';
	epoch_file_.flush();

	++n_epochs_;
	total_secs_ += m.train_secs;
	total_gflop_ += m.gflops * m.train_secs;
	total_images_ += m.images_per_sec * m.train_secs;
	last_acc_ = m.val_acc;
	if (m.val_acc > best_acc_) best_acc_ = m.val_acc;
}

void RunLogger::write_summary(std::string_view backend, std::string_view model, std::size_t batch,
	float lr_start)const {
	const std::string path = log_dir_ + "/runs.csv";
	rotate_if_stale(path, run_id_);
	const bool is_new = !std::filesystem::exists(path);

	std::ofstream out(path, std::ios::app);
	if (!out) throw std::runtime_error("RunLogger: could not open " + path);

	if (is_new) out << RUNS_HEADER << '\n';

	// Rates are total work / total time, not the mean of per-epoch rates
	// (averaging rates over-weights fast epochs).
	const double secs = total_secs_ > 0.0 ? total_secs_ : 1.0;
	const double mean_epoch_secs = n_epochs_ ? total_secs_ / static_cast<double>(n_epochs_) : 0.0;
	const double mean_gflops = total_gflop_ / secs;
	const double mean_ips = total_images_ / secs;
	const double mfu_pct = peak_gflops_ > 0.0 ? 100.0 * mean_gflops / peak_gflops_ : 0.0;

	out << run_id_ << ',' << BUILD_CONFIG << ',' << backend << ',' << model << ','
		<< n_epochs_ << ',' << batch << ',' << lr_start << ','
		<< last_acc_ << ',' << best_acc_ << ',' << mean_epoch_secs << ','
		<< total_secs_ << ',' << mean_ips << ',' << mean_gflops << ','
		<< peak_gflops_ << ',' << mfu_pct << '\n';
}
