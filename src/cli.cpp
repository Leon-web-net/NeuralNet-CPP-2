#include "cli.hpp"
#include <stdexcept>
#include <string_view>

namespace {
	constexpr std::string_view USAGE =
		"usage: NeuralNet [train | train-full | eval | predict] "
		"[--no-log] [--no-save] [--model NAME]\n"
		"                 [--epochs N] [--batch N] [--lr X] [--activation relu|sigmoid]";

	[[noreturn]] void bad_arg(std::string_view what) {
		throw std::invalid_argument(std::string(what) + "\n" + std::string(USAGE));
	}

	// Throws on junk ("abc", "12x") instead of silently reading a prefix.
	std::size_t parse_count(std::string_view flag, const std::string& s) {
		std::size_t pos = 0;
		unsigned long v = 0;
		try { v = std::stoul(s, &pos); } catch (const std::exception&) { pos = 0; }
		if (pos != s.size() || v == 0) bad_arg(std::string(flag) + " needs a positive integer, got '" + s + "'");
		return v;
	}

	float parse_positive_float(std::string_view flag, const std::string& s) {
		std::size_t pos = 0;
		float v = 0.0f;
		try { v = std::stof(s, &pos); } catch (const std::exception&) { pos = 0; }
		if (pos != s.size() || !(v > 0.0f)) bad_arg(std::string(flag) + " needs a positive number, got '" + s + "'");
		return v;
	}
}

Options parse_args(int argc, char* argv[]) {
	Options opt;
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		const bool has_value = i + 1 < argc;

		if (arg == "train")            opt.mode = Mode::Train;
		else if (arg == "train-full")  opt.mode = Mode::TrainFull;
		else if (arg == "eval")        opt.mode = Mode::Eval;
		else if (arg == "predict")     opt.mode = Mode::Predict;
		else if (arg == "--no-log")    opt.logging = false;
		else if (arg == "--no-save")   opt.save = false;
		else if (arg == "--model" && has_value)  opt.model_name = argv[++i];
		else if (arg == "--epochs" && has_value) opt.cfg.epochs = parse_count(arg, argv[++i]);
		else if (arg == "--batch" && has_value)  opt.cfg.batch = parse_count(arg, argv[++i]);
		else if (arg == "--lr" && has_value)     opt.cfg.lr_start = parse_positive_float(arg, argv[++i]);
		else if (arg == "--activation" && has_value) {
			const std::string_view a = argv[++i];
			if (a == "relu")         opt.cfg.activation = Activation::ReLU;
			else if (a == "sigmoid") opt.cfg.activation = Activation::Sigmoid;
			else bad_arg("--activation must be relu or sigmoid, got '" + std::string(a) + "'");
		}
		else bad_arg("unknown argument '" + std::string(arg) + "'");
	}
	if (opt.model_name.empty()) {
		opt.model_name = opt.cfg.default_model_name();
		if (opt.mode == Mode::TrainFull) opt.model_name += "_full";
	}
	return opt;
}
