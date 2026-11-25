#include "cli.hpp"
#include <stdexcept>
#include <string_view>

namespace {
	constexpr std::string_view USAGE =
		"usage: NeuralNet [train | train-full | eval | predict] "
		"[--no-log] [--no-save] [--model NAME]";
}

Options parse_args(int argc, char* argv[]) {
	Options opt;
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "train")            opt.mode = Mode::Train;
		else if (arg == "train-full")  opt.mode = Mode::TrainFull;
		else if (arg == "eval")        opt.mode = Mode::Eval;
		else if (arg == "predict")     opt.mode = Mode::Predict;
		else if (arg == "--no-log")    opt.logging = false;
		else if (arg == "--no-save")   opt.save = false;
		else if (arg == "--model" && i + 1 < argc) opt.model_name = argv[++i];
		else throw std::invalid_argument(
			"unknown argument '" + std::string(arg) + "'\n" + std::string(USAGE));
	}
	if (opt.model_name.empty()) {
		opt.model_name = opt.cfg.model_name;
		if (opt.mode == Mode::TrainFull) opt.model_name += "_full";
	}
	return opt;
}
