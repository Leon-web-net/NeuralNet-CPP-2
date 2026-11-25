#pragma once
#include "config.hpp"
#include <string>

enum class Mode { Train, TrainFull, Eval, Predict };

struct Options {
	Mode mode = Mode::Train;
	bool logging = true;
	bool save = true;
	std::string model_name;   // filled with a default if not given
	TrainConfig cfg;
};

// Throws std::invalid_argument (with usage text) on an unknown argument.
Options parse_args(int argc, char* argv[]);
