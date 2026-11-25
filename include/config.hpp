#pragma once
#include "metrics.hpp"   // HardwarePeak
#include "mnist.hpp"
#include "network.hpp"   // HIDDEN1, HIDDEN2
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Everything a run depends on, as one value. Defaults reproduce the baseline;
// command-line flags can override fields without recompiling.
struct TrainConfig {
	// Hyperparameters
	std::size_t epochs = 15;
	std::size_t batch = 64;
	float       lr_start = 0.4f;
	float       lr_decay = 0.9f;
	std::size_t decay_every = 5;
	float       train_frac = 0.8f;

	// Separate seeds: changing the shuffle never changes which images are held out.
	std::uint32_t split_seed = 123;
	std::uint32_t shuffle_seed = 456;

	// Architecture: input, hidden..., output
	std::vector<std::size_t> layer_sizes =
		{ mnist::IMAGE_SIZE, HIDDEN1, HIDDEN2, mnist::NUM_CLASSES };
	Activation activation = Activation::ReLU;

	// Labels
	std::string backend = "cpu_naive";                       // for timing logs
	std::string model_version = "v1";   // default model name = mlp_<sizes>_<activation>_<version>

	std::string default_model_name() const {
		std::string name = "mlp_";
		for (std::size_t i = 0; i < layer_sizes.size(); ++i) {
			if (i) name += '-';
			name += std::to_string(layer_sizes[i]);
		}
		return name + '_' + to_string(activation) + '_' + model_version;
	}

	// Hardware peak for MFU. AMD Ryzen 7 5800X (Zen 3): 8 cores, 3.8 GHz base
	// (4.7 boost), 2 FMA units x 8 floats (AVX2) x 2 ops = 32 FP32 FLOPs/cycle/core.
	HardwarePeak cpu_peak{ 8.0, 3.8, 32.0 };
};
