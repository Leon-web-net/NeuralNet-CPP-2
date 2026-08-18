#pragma once 

#include <vector>
#include <cstddef>
#include <functional>

class Matrix {
public:

	// Constructor declarations
	Matrix(std::size_t rows, std::size_t cols);
	Matrix(std::size_t rows, std::size_t cols, std::vector<float> values);

	std::size_t rows() const { return rows_; }
	std::size_t cols() const { return cols_; }

	// Bounds checked access to matrix elements
	// write and read access
	float& at(std::size_t r, std::size_t c);
	const float& at(std::size_t r, std::size_t c) const;

	//unchecked access
	// write and read access
	float& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
	const float& operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

	Matrix matmul(const Matrix& other) const;
	Matrix transpose() const;

	// TODO: add, subtract, hadamard, scale, add_col_vector, apply, sum, argmax_cols
	Matrix add(const Matrix& other) const;
	Matrix subtract(const Matrix& other)const;
	Matrix hadamard(const Matrix& other)const;
	Matrix scale(float scalar)const;
	Matrix add_col_vector(const Matrix& col_vector)const;

private:
	std::size_t rows_;
	std::size_t cols_;
	std::vector<float>data_;

};