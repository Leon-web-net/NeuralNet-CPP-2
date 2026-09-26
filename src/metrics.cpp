#include "metrics.hpp"
#include <chrono>
#include <format>
#include <filesystem>
#include <stdexcept>

std::string make_run_id(std::string_view backend) {
	const auto now = std::chrono::floor<std::chrono::seconds>(
		std::chrono::system_clock::now());
	return std::format("{:%Y%m%d-%H%M%S}_{}", now, backend);
}

RunLogger::RunLogger(const std::string& log_dir, const std::string& run_id)
	:log_dir_(log_dir), run_id_(run_id) {
	std::filesystem::create_directories(log_dir_);

	const std::string path = log_dir_ + "/" + run_id_ + "_epochs.csv";
	epoch_file_.open(path);
	if(!epoch_file_) throw std::runtime_error("RunLogger: could not open " + path);
	
	epoch_file_ << "epoch,lr,train_loss,val_loss,val_acc,train_secs\n";
}

void RunLogger::log_epoch(const EpochMetrics& m) {
	epoch_file_ << m.epoch << "," << m.lr << ',' << m.train_loss << ","
		<< m.val_loss << "," << m.val_acc<<"," << m.train_secs << "\n";
	epoch_file_.flush();

	++n_epochs_;
	total_secs_ += m.train_secs;
	last_acc_ = m.val_acc;
	if (m.val_acc > best_acc_) best_acc_ = m.val_acc;
}

void RunLogger::write_summary(std::string_view backend, std::size_t batch,
	float lr_start)const {
	const std::string path = log_dir_ + "/runs.csv";
	const bool is_new = !std::filesystem::exists(path);

	std::ofstream out(path, std::ios::app);
	if (!out) throw std::runtime_error("RunLogger: could not open" + path);

	if (is_new) {
		out << "run_id,build,backend,epochs,batch,lr_start,"
			"final_val_acc,best_val_acc,mean_epoch_secs,total_train_secs\n";
	}

	const double mean_secs = n_epochs_ ? total_secs_ / static_cast<double>(n_epochs_) : 0.0;
	out << run_id_ << ',' << BUILD_CONFIG << ',' << backend << ','
		<< n_epochs_ << ',' << batch << ',' << lr_start << ','
		<< last_acc_ << ',' << best_acc_ << ',' << mean_secs << ','
		<< total_secs_ << '\n';

}