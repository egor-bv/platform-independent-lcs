#pragma once
#include <algorithm>
#include <cmath>

class PermutationMatrix
{

	int *row_to_col;
	int *col_to_row;

public:
	const int size;

	PermutationMatrix(int _size)
		: size(_size)
	{
		row_to_col = new int[size] {};
		col_to_row = new int[size] {};
	}

	~PermutationMatrix()
	{
		delete[] row_to_col;
		delete[] col_to_row;
	}

	void set_point(int row, int col)
	{
		row_to_col[row] = col;
		col_to_row[col] = row;
	}

	int get_row_by_col(int col) const
	{
		return col_to_row[col];
	}

	int get_col_by_row(int row) const
	{
		return row_to_col[row];
	}

	static PermutationMatrix FromStrands(int *h_strands, int m, int *v_strands, int n)
	{
		PermutationMatrix p(m + n);
		for (int l = 0; l < m; l++) p.set_point(h_strands[l], n + l);
		for (int r = m; r < m + n; r++) p.set_point(v_strands[r - m], r - m);

		if (p.size < 100)
		{
			std::cout << "DEBUG INFO:" << std::endl;
			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_row_by_col(row_i) << " ";
			}
			std::cout << std::endl;

			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_col_by_row(row_i) << " ";
			}
			std::cout << std::endl;
		}

		return p;
	}
};

const long long R = 4294967279;
const long long M = 4294967291;

long long hash(PermutationMatrix &arr, int size)
{
	long long hash = 0;
	for (int i = 0; i < size; i++) {
		hash = (R * hash + arr.get_row_by_col(i)) % M;
	}
	return hash;
}


namespace std
{

template<>
struct hash<PermutationMatrix>
{

	std::size_t operator()(const PermutationMatrix &p) const
	{
		std::size_t sum = 0;
		int bits_per_symbol = int(std::ceil(log2(p.size)));

		for (int i = 0; i < p.size; ++i)
		{
			sum = (sum << bits_per_symbol) + p.get_col_by_row(i);
		}
		return sum;
	}
};

}