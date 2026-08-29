#include "matrix.hpp"
#include <stdexcept>
#include <string>
 
Matrix Matrix::matmul(const Matrix& other)const {
	if (cols_ != other.rows_) {
		throw std::invalid_argument(
			"matmul: (" + std::to_string(rows_) + "X" + std::to_string(cols_) +
			") * (" + std::to_string(other.rows_) + "X" + std::to_string(other.cols_) +
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

Matrix Matrix::transpose()const {
	Matrix result(cols_, rows_);

	for (std::size_t i=0; i < rows_;++i) {
		for (std::size_t j=0; j < cols_;++j) {
			result(j,i) = (*this)(i,j);
		}
	}
	return result;
}

Matrix::Matrix(std::size_t rows, std::size_t cols)
	:rows_(rows), cols_(cols), data_(rows* cols, 0.0f)
{

}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<float> values) 
	:rows_(rows), cols_(cols), data_(std::move(values))
{
	if (data_.size() != rows * cols){
		throw std::invalid_argument(
			"Matrix constructor: values size (" + std::to_string(data_.size()) +
			") does not match rows * cols (" + std::to_string(rows * cols) + ")"
		);
	}
}

float& Matrix::at(std::size_t r, std::size_t c) {
	if (r >= rows_ || c >= cols_) {
		throw std::out_of_range(
			"at(" + std::to_string(r) + "," + std::to_string(c) +
			") out of range for " + std::to_string(rows_) + "x" +
			std::to_string(cols_));
	}
		return data_[r * cols_ + c];
}

const float& Matrix::at(std::size_t r, std::size_t c) const {
	if (r >= rows_ || c >= cols_) {
		throw std::out_of_range(
			"at(" + std::to_string(r) + "," + std::to_string(c) +
			") out of range for " + std::to_string(rows_) + "x" +
			std::to_string(cols_));
	}

	return data_[r * cols_ + c];
}

Matrix Matrix::add(const Matrix& other)const {
	if (cols_ != other.cols_ || rows_ != other.rows_){
		throw std::invalid_argument(
			"add: (" + std::to_string(rows_) + "X" + std::to_string(cols_) +
			") + (" + std::to_string(other.rows_) + "X" + std::to_string(other.cols_) +
			") - dimensions don't match");
	}

	Matrix result(rows_, cols_);
	
	for(std::size_t i=0; i<rows_;++i){
		for (std::size_t j = 0;j < cols_;++j) {
			result(i, j) = (*this)(i, j) + other(i, j);
		}
	}
	return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument(
			"subtract: (" + std::to_string(rows_) + "x" + std::to_string(cols_) +
			") - (" + std::to_string(other.rows_) + "x" + std::to_string(other.cols_) +
			") - dimensions don't match");
	}
	Matrix result(rows_, cols_);

	for (std::size_t i = 0; i < rows_; ++i) {
		for (std::size_t j = 0;j < cols_;++j) {
			result(i, j) = (*this)(i, j) - other(i, j);
		}
	}

	return result;
}

Matrix Matrix::scale(float scalar)const {
	Matrix result(rows_, cols_);
	for (std::size_t i = 0; i < rows_; ++i) {
		for (std::size_t j = 0;j < cols_;++j) {
			result(i, j) = (*this)(i, j) * scalar;
		}
	}
	return result;
}

Matrix Matrix::hadamard(const Matrix& other)const {
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument(
			"hadamard: (" + std::to_string(rows_) + "x" + std::to_string(cols_) +
			") hadamard (" + std::to_string(other.rows_) + "x" + std::to_string(other.cols_) +
			") - dimensions don't match");
	}
	Matrix result(rows_, cols_);
	for (std::size_t i = 0; i < rows_; ++i) {
		for (std::size_t j = 0;j < cols_;++j) {
			result(i, j) = (*this)(i, j) * other(i, j);
		}
	}
	return result;
}

Matrix Matrix::add_col_vector(const Matrix& col_vector)const {
	if (col_vector.cols_ != 1 || col_vector.rows_ != rows_) {
		throw std::invalid_argument(
			"add_col_vector: matrix (" + std::to_string(rows_) + "x" + std::to_string(cols_) +
			") needs a column vector (" + std::to_string(rows_) + "x1), got (" +
			std::to_string(col_vector.rows_) + "x" + std::to_string(col_vector.cols_) + ")");
	}

	Matrix result(rows_, cols_);
	for (std::size_t i = 0; i < rows_;++i) {
		for (std::size_t j = 0; j < cols_;++j) {
			result(i, j) = (*this)(i, j) + col_vector(i, 0);
		}
	}
	return result;
}

float Matrix::sum() const {
	float total = 0.0f;
	
	for (std::size_t i = 0; i < rows_;++i) {
		for (std::size_t j = 0; j < cols_;++j) {
			total += (*this)(i, j);
		}
	}
	return total;
};

Matrix Matrix::argmax_cols() const{
	Matrix result(1, cols_);

	for (std::size_t j = 0; j < cols_; ++j) {
		float best_value = (*this)(0, j);
		std::size_t best_row = 0;

		for (std::size_t i = 1; i < rows_; ++i) {
			if ((*this)(i, j) > best_value) {
				best_value = (*this)(i, j);
				best_row = i;
			}
		}

		result(0, j) = static_cast<float>(best_row);
	}

	return result;
}

Matrix Matrix::apply(std::function<float(float)> func) const {
	Matrix result(rows_, cols_);
	for (std::size_t i = 0; i < rows_; ++i) {
		for (std::size_t j = 0; j < cols_; ++j) {
			result(i, j) = func((*this)(i, j));
		}
	}
	return result;
}
