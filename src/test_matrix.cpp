#include "matrix.hpp"
#include <iostream>

void print(const Matrix& m, const std::string& name) {
	std::cout << name << " (" << m.rows() << "x" << m.cols() << "):\n";
	for (std::size_t i = 0; i < m.rows(); ++i) {
		for (std : size_t j = 0; j < m.cols(); ++j) {
			std::cout << m(i, j) << "\t";
		}
		std::cout <<'\n';
	}
	std::cout << std::'\n';
}

int main() {
	
	std::vector<float> vec_A = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	Matrix A(2, 3, vec_A);
	std::cout << "Matrix A ("<<A.rows()<<"x"<<A.cols()<<")" << std::endl;

	std::vector<float> vec_B = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
	Matrix B(3, 2, vec_B);
	std::cout << "Matrix B (" << B.rows() << "x" << B.cols() << ")" << std::endl;

	Matrix T_A = A.transpose();
	std::cout << "Matrix A^T (" << T_A.rows() << "x" << T_A.cols() << ")" << std::endl;

	const Matrix C = A.add(B);
	std::cout << "Matrix C (" << C.rows() << "x" << C.cols() << ")" << std::endl;

	return 0;
}