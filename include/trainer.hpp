#pragma once
#include "config.hpp"
#include "metrics.hpp"
#include "mnist.hpp"
#include "network.hpp"
#include <cstddef>
#include <vector>

// Runs the epoch loop on an already-built network: shuffle, mini-batch SGD,
// validation, timing. Doesn't create, save or summarise; the caller owns that.
// val_idx may be empty (train-full): validation metrics are then NaN.
// logger may be null (logging disabled).
void train(Network& net, const mnist::Dataset& data, std::vector<std::size_t> train_idx,
	const std::vector<std::size_t>& val_idx, const TrainConfig& cfg, RunLogger* logger);
