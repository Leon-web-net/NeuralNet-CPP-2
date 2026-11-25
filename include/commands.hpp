#pragma once
#include "cli.hpp"
#include "mnist.hpp"
#include <cstddef>
#include <vector>

// One function per CLI mode. Each builds what it needs, runs, and writes its outputs.

// val_idx empty = train-full (no validation, "_full" run label).
void run_train(const mnist::Dataset& data, std::vector<std::size_t> train_idx,
	const std::vector<std::size_t>& val_idx, const Options& opt);

void run_eval(const mnist::Dataset& data, const std::vector<std::size_t>& val_idx,
	const Options& opt);

// Reads test.csv itself, writes a Kaggle submission CSV.
void run_predict(const Options& opt);
