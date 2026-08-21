#include "matrix.hpp"
#include <iostream>
#include <iomanip>
#include <random>


void print(const Matrix& m, const std::string& name, int width = 10, int precision = 4) {
	std::cout << name << " (" << m.rows() << "x" << m.cols() << "):\n";

	// Set floating-point format to fixed decimal places
	std::cout << std::fixed << std::setprecision(precision);

	for (std::size_t i = 0; i < m.rows(); ++i) {
		std::cout << "  "; // Small left indent
		for (std::size_t j = 0; j < m.cols(); ++j) {
			// std::setw must be called before every item; it resets after one output
			std::cout << std::setw(width) << m(i, j) << " ";
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

void rand_matrix(Matrix& m, float lo = -1.0f, float hi = 1.0f) {
	static std::mt19937 gen(42);
	std::uniform_real_distribution<float> dist(lo, hi);

	for(std::size_t i =0; i<m.rows();++i){
		for (std::size_t j = 0; j < m.cols();++j) {
			m(i, j) = dist(gen);
		}
	}
}

int main() {
	
	std::vector<float> vec_A = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	Matrix A(2, 3, vec_A);
	print(A, "Matrix A");

	std::vector<float> vec_B = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
	Matrix B(3, 2, vec_B);
	print(B, "Matrix B");

	Matrix T_A = A.transpose();
	print(T_A, "Matrix A^T");

	try {
		Matrix bad_add = A.add(B);
		std::cout << "FAIL: mismatched add should have thrown\n";
	}
	catch (const std::invalid_argument& e) {
		std::cout << "PASS: threw as expected ->" << e.what() << '\n';
	}
	Matrix C = A.add(A);
	print(C, "Matrix C = A + A");

	Matrix D = A.matmul(B);
	print(D, "Matrix D = A * B");

	Matrix rand_mat = Matrix(4, 5);
	rand_matrix(rand_mat);

	print(rand_mat, "Random Matrix 4x5");

	Matrix max_col_idx = rand_mat.argmax_cols();
	print(max_col_idx, "Argmax of Random Matrix (column indices)");


	
	return 0;
}