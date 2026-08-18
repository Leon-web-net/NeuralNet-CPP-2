#include "matrix.hpp"
#include <iostream>

void print(const Matrix& m, const std::string& name) {
	std::cout << name << " (" << m.rows() << "x" << m.cols() << "):\n";
	for (std::size_t i = 0; i < m.rows(); ++i) {
		for (std::size_t j = 0; j < m.cols(); ++j) {
			std::cout << m(i, j) << "\t";
		}
		std::cout <<'\n';
	}
	std::cout << '\n';
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


	
	return 0;
}