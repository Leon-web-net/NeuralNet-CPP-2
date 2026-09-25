#include "network.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>


int main() {
	const std::size_t IN = 4, OUT = 2, B = 5;
	Network net({ IN,3,OUT });

	// random inputs in [0 1]; random one-hot labels.
	std::mt19937 gen(7);
	std::uniform_real_distribution<float> pix(0.0f, 1.0f);
	std::uniform_int_distribution<std::size_t> cls(0, OUT - 1);
	Matrix X(IN, B), Y(OUT, B);
	for (std::size_t s = 0; s < B; ++s) {
		for (std::size_t f = 0; f < IN; ++f) {
			X(f, s) = pix(gen);
		}
		Y(cls(gen), s) = 1.0f;
	}

	std::vector<LayerGrad> grads = net.backward(net.forward(X), Y);

	const float EPS = 1e-3f;
	float worst = 0.0f;

	for (std::size_t l = 0; l < net.num_layers(); ++l) {
		Matrix& W = net.layer(l).weights;
		for (std::size_t r = 0; r < W.rows(); ++r) {
			for (std::size_t c = 0; c < W.cols(); ++c) {
				const float original = W(r, c);

				W(r, c) = original - EPS;
				float loss_minus = net.cross_entropy_loss(net.forward(X).back(), Y);
				W(r, c) = original + EPS;
				float loss_plus = net.cross_entropy_loss(net.forward(X).back(), Y);
				W(r, c) = original;

				float numerical = (loss_plus - loss_minus) / (2.0f * EPS);
				float analytic = grads[l].dW(r, c);
				float rel = std::abs(analytic - numerical) /
					std::max(std::abs(analytic) + std::abs(numerical), 1e-8f);
				
				worst = std::max(worst, rel);
			}
		}
	}

	std::cout << "worst relative error: " << worst << "\n";
	std::cout << (worst < 1e-2f ? "Pass\n" : "Fail\n");
	
	return worst < 1e-2f ? 0 : 1;

}

