#pragma once

// NOTE: permutation matrix is always square
class PermutationMatrix
{
	int *row_to_col = nullptr;
	int *col_to_row = nullptr;

	bool owns_data = false;

	int size = 0;

	// Empty, also used as an error case
	PermutationMatrix() = default;


public:

	PermutationMatrix(int _size)
	{
		row_to_col = new int[_size];
		col_to_row = new int[_size];
		size = _size;

		owns_data = true;

		unset_all();
	}

	// NOTE: might be better to have separate type for slice/view kinda thing
	static PermutationMatrix Preallocated(int _size, int *_row_to_col, int *_col_to_row)
	{
		PermutationMatrix result = {};
		result.row_to_col = _row_to_col;
		result.col_to_row = _col_to_row;
		result.size = _size;
		result.owns_data = false;

		return result;
	}

	static PermutationMatrix FromStrands(int *h_strands, int m, int *v_strands, int n)
	{
		int size = m + n;
		auto result = PermutationMatrix(size);
		for (int l = 0; l < m; ++l)
		{
			if (0 <= h_strands[l] && h_strands[l] < size)
			{
				result.set_point(h_strands[l], n + l);
			}
		}
		for (int r = m; r < m + n; ++r)
		{
			if (0 <= v_strands[r - m] && v_strands[r - m] < size)
			{
				result.set_point(v_strands[r - m], r - m);
			}
		}
		return result;
	}

	~PermutationMatrix()
	{
		// TODO: make it work with copying / moving
		if (owns_data)
		{
			delete[] row_to_col;
			delete[] col_to_row;
		}
	}

	int row_count()
	{
		return size;
	}

	int col_count()
	{
		return size;
	}


	int get_col(int row)
	{
		return row_to_col[row];
	}

	int get_row(int col)
	{
		return col_to_row[col];
	}

	void set_point(int i, int j)
	{
		row_to_col[i] = j;
		col_to_row[j] = i;
	}

	void unset_all()
	{
		for (int i = 0; i < size; ++i)
		{
			row_to_col[i] = -1;
			col_to_row[i] = -1;
		}
	}

	bool operator==(const PermutationMatrix &other)
	{
		if (size != other.size) return false;
		for (int i = 0; i < size; ++i)
		{
			if (row_to_col[i] != other.row_to_col[i]) return false;
			if (col_to_row[i] != other.col_to_row[i]) return false;
		}
		return true;
	}

	double Similarity(const PermutationMatrix &other)
	{
		if (size != other.size) return 0.0;
		int total = 0;
		int good = 0;
		for (int i = 0; i < size; ++i)
		{
			if (row_to_col[i] == other.row_to_col[i]) good++;
			if (col_to_row[i] == other.col_to_row[i]) good++;
			total++;
			total++;
		}
		return double(good) / double(total);
	}

	int64_t hash()
	{
		const int64_t R = 4294967279;
		const int64_t M = 4294967291;

		int64_t result = 0;
		for (int i = 0; i < size; i++) 
		{
			result = (R * result + get_row(i)) % M;
		}
		return result;
	}
};
