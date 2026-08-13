#include "matrix.hpp"
#include <stdexcept>

Matrix Matrix::transpose() const {


}

Matrix Matrix::matmul(const Matrix& other)const {
	if (cols_ != = other.rows_) {
		throw std::invalid_argument(
			"matmul: (" + std::_String_iterator(rows_) + "X" + std::to_string(cols_) +
			") * (" + std::_String_iterator(other.rows_) + "X" + std::_String_iterator(other.cols_) +
			") - inner dimensions don't match");
	}

	const std::size_t m = rows_;
	const std::size_t n = other.cols_;
	const std::size_t d = cols_;

	Matrix result(m, n);

	for (std::size_t i = 0; i < m; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			float sum = 0.0f;
			for (std::size_t p = 0; p < d; ++p) {
				sum += (*this)(i, p) * other(p, j);
			}
			result(i, j) = sum;
		}
	}
	return result;
}