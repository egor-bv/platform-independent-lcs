#pragma once

#include <CL/sycl.hpp>
#include "lcs_common.hpp"
#include "lcs_grid.hpp"



template<typename Symbol, typename Strand>
class GridEmbeddingMulti
{
public:
	Symbol *a_data;
	Symbol *b_data;
	Strand *h_strands;
	Strand *v_strands;

	int num_subproblems_m;
	int num_subproblems_n;

	int m;
	int n;

	int sub_m;
	int sub_n;
	int clipped_sub_m;
	int clipped_sub_n;

	int stride_h;
	int stride_v;

	int h_len_alloc;
	int v_len_alloc;

	int h_size_combined(int i_sub_first, int i_sub_count)
	{
		Assert(i_sub_first + i_sub_count < num_subproblems_m);
		int end = i_sub_first + i_sub_count;
		if (end == num_subproblems_m)
		{
			return sub_m * (i_sub_count - 1) + clipped_sub_m;
		}
		else
		{
			return sub_m * (i_sub_count);
		}
	}

	int v_size_combined(int j_sub_first, int j_sub_count)
	{
		Assert(j_sub_first + j_sub_count < num_subproblems_n);
		int end = j_sub_first + j_sub_count;
		if (end == num_subproblems_n)
		{
			return sub_n * (j_sub_count - 1) + clipped_sub_n;
		}
		else
		{
			return sub_n * (j_sub_count);
		}
	}

	int h_offset(int i_sub, int j_sub) const
	{
		return i_sub * stride_h + j_sub * num_subproblems_m * stride_h;
	}

	int v_offset(int i_sub, int j_sub) const
	{
		return i_sub * stride_v + j_sub * num_subproblems_m * stride_v;
	}

	GridEmbeddingMulti(const Symbol *a_raw, int a_len,
						 const Symbol *b_raw, int b_len,
						 int count_m, int count_n)
		: num_subproblems_m(count_m)
		, num_subproblems_n(count_n)
	{
		m = a_len;
		n = b_len;

		sub_m = CeilDiv(m, num_subproblems_m);
		sub_n = CeilDiv(n, num_subproblems_n);

		clipped_sub_m = m - sub_m * (num_subproblems_m - 1);
		clipped_sub_n = n - sub_n * (num_subproblems_n - 1);

		// Conservative estimate
		constexpr int cacheline_elements = 64;
		stride_h = AlignedToMultiple(sub_m, cacheline_elements);
		stride_v = AlignedToMultiple(sub_n, cacheline_elements);

		int replication = num_subproblems_m * num_subproblems_n;
		h_len_alloc = stride_h * replication;
		v_len_alloc = stride_v * replication;

		a_data = new Symbol[m];
		b_data = new Symbol[n];

		for (int i = 0; i < m; ++i)
		{
			a_data[i] = a_raw[m - i - 1];
		}
		for (int j = 0; j < n; ++j)
		{
			b_data[j] = b_raw[j];
		}

		

		h_strands = new Strand[h_len_alloc];
		v_strands = new Strand[v_len_alloc];

		for (int i_sub = 0; i_sub < num_subproblems_m; ++i_sub)
		{
			for (int j_sub = 0; j_sub < num_subproblems_n; ++j_sub)
			{
				int actual_sub_m = i_sub == num_subproblems_m - 1 ? clipped_sub_m : sub_m;
				int actual_sub_n = j_sub == num_subproblems_n - 1 ? clipped_sub_n : sub_n;

				for (int i = 0; i < actual_sub_m; ++i)
				{
					h_strands[h_offset(i_sub, j_sub) + i] = i;
				}
				for (int j = 0; j < actual_sub_n; ++j)
				{
					v_strands[v_offset(i_sub, j_sub) + j] = actual_sub_m + j;
				}
			}
		}
	}


	~GridEmbeddingMulti()
	{
		delete[] a_data;
		delete[] b_data;
		delete[] h_strands;
		delete[] v_strands;
	}
};

GridEmbeddingMulti<int, int> make_embedding_multi(const LcsInput &input, int count_m, int count_n)
{
	return GridEmbeddingMulti<int, int>(input.a_data, input.a_size, input.b_data, input.b_size,
							  count_m, count_n);
}



GridBuffers<int, int>
make_buffers(GridEmbeddingMulti<int, int> &grid)
{
	return GridBuffers<int, int>(grid.a_data, grid.m,
								 grid.b_data, grid.n,
								 grid.h_strands, grid.h_len_alloc,
								 grid.v_strands, grid.v_len_alloc);
}
