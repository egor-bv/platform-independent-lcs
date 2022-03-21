#pragma once
#include <inttypes.h>

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
#if 0
		if (m + n < 100)
		{
			std::cout << "<.......>\n";
			for (int i = 0; i < m; ++i)
			{
				std::cout << h_strands[i] << " ";
			}
			std::cout << "\n";
			for (int j = 0; j < n; ++j)
			{
				std::cout << v_strands[j] << " ";
			}
			std::cout << "\n";
		}
#endif
		for (int l = 0; l < m; l++) p.set_point(h_strands[l], n + l);
		for (int r = m; r < m + n; r++) p.set_point(v_strands[r - m], r - m);
		return p;
	}

	int64_t hash()
	{
		const int64_t R = 4294967279;
		const int64_t M = 4294967291;

		int64_t result = 0;
		for (int i = 0; i < size; i++) {
			result = (R * result + get_row_by_col(i)) % M;
		}
		return result;
	}


};


