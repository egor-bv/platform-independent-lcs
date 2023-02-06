#pragma once

#include "lcs_types.hpp"
#include "lcs_grid_multi.hpp"
#include "lcs_common.hpp"

#include "permutation.hpp"
#include "braid_multiplication.hpp"

PermutationMatrix
PermutationMatrixRecursivelyCombine(int i_sub_first, int i_sub_count,
									int j_sub_first, int j_sub_count,
									GridEmbeddingMulti<int, int> &grid)
{
	if (i_sub_count == 1 && j_sub_count == 1)
	{
		int h_size = grid.h_size_combined(i_sub_first, 1);
		int v_size = grid.v_size_combined(j_sub_first, 1);

		return PermutationMatrix::FromStrands(grid.h_strands + grid.h_offset(i_sub_first, j_sub_first), h_size,
											  grid.v_strands + grid.v_offset(i_sub_first, j_sub_first), v_size);
	}

	if (i_sub_count > j_sub_count)
	{
		int overlap = grid.v_size_combined(j_sub_first, j_sub_count);
		auto p = PermutationMatrixRecursivelyCombine(i_sub_first, i_sub_count / 2,
													 j_sub_first, j_sub_count,
													 grid);
		auto q = PermutationMatrixRecursivelyCombine(i_sub_first + i_sub_count / 2, i_sub_count / 2,
													 j_sub_first, j_sub_count,
													 grid);

		return staggered_multiply<true>(q, p, overlap);
	}
	else
	{
		int overlap = grid.h_size_combined(i_sub_first, i_sub_count);
		auto p = PermutationMatrixRecursivelyCombine(i_sub_first, i_sub_count,
													 j_sub_first, j_sub_count / 2,
													 grid);
		auto q = PermutationMatrixRecursivelyCombine(i_sub_first, i_sub_count,
													 j_sub_first + j_sub_count / 2, j_sub_count / 2,
													 grid);
		return staggered_multiply<false>(p, q, overlap);
	}
}

PermutationMatrix
CombineMultiGrid(GridEmbeddingMulti<int, int> &grid)
{
	return PermutationMatrixRecursivelyCombine(0, grid.num_subproblems_m,
											   0, grid.num_subproblems_n,
											   grid);
}

template <int SG_SIZE, int DEPTH>
void
Lcs_Semi_Antidiagonal_Hybrid_MT(const LcsInput &input, LcsContext &ctx)
{
	int count_m = 1;
	int count_n = 1;
	int m_remaining = input.a_size;
	int n_remaining = input.b_size;

	for (int depth = 0; depth < DEPTH; ++depth)
	{
		if (m_remaining <= 64 && n_remaining <= 64) break;
		if (m_remaining >= n_remaining)
		{
			m_remaining /= 2;
			count_m *= 2;
		}
		else
		{
			n_remaining /= 2;
			count_n *= 2;
		}
	}
	// printf("Given size: %dx%d, Depth: %d, Subproblems: %dx%d\n", input.a_size, input.b_size, DEPTH, count_m, count_n);

	auto grid = make_embedding_multi(input, count_m, count_n);

	int m = grid.m;
	int n = grid.n;

	int stride_h = grid.stride_h;
	int stride_v = grid.stride_v;

	int sub_m = grid.sub_m;
	int sub_n = grid.sub_n;
	int clipped_sub_m = grid.clipped_sub_m;
	int clipped_sub_n = grid.clipped_sub_n;

	// Use single thread per subproblem, all done in a single dispatch
	{
		int num_groups = grid.num_subproblems_m * grid.num_subproblems_n;
		int local_size = SG_SIZE;
		int global_size = local_size * num_groups;


		auto buf = make_buffers(grid);
		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);
			cgh.parallel_for(
				sycl::nd_range<1>(global_size, local_size),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				auto sg_id = sg.get_local_linear_id();

				auto group_id = item.get_group_linear_id();
				int i_sub = group_id % count_m;
				int j_sub = group_id / count_m;

				int actual_m = i_sub == count_m - 1 ? clipped_sub_m : sub_m;
				int actual_n = j_sub == count_n - 1 ? clipped_sub_n : sub_n;

				auto update_cell = [&](int i, int j)
				{
					int i_strands = i + stride_h * group_id;
					int j_strands = j + stride_v * group_id;

					int i_symbols = i + i_sub * sub_m;
					int j_symbols = j + j_sub * sub_n;
					update_cell_semilocal_separate_indexing(acc.a, acc.b, acc.h_strands, acc.v_strands,
															i_symbols, j_symbols,
															i_strands, j_strands);
				};

				int diag_count = actual_m + actual_n - 1;
				for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
				{
					auto d = antidiag_at(diag_idx, actual_m, actual_n);
					int sg_step_count = d.diag_len / SG_SIZE;

					// complete steps
					#pragma unroll 4
					for (int sg_step = 0; sg_step < sg_step_count; ++sg_step)
					{
						int step = sg_step * SG_SIZE + sg_id;
						int i = d.i_first + step;
						int j = d.j_first + step;
						update_cell(i, j);
					}

					// last incomplete step when diag_len not multiple of SG_SIZE
					int last_step = sg_step_count * SG_SIZE + sg_id;
					if (last_step < d.diag_len)
					{
						int i = d.i_first + last_step;
						int j = d.j_first + last_step;
						update_cell(i, j);
					}

					sg.barrier();
				}

			}); // end parallel_for
		}); // end submit
	} // end buffers lifetime

	ctx.matrix = CombineMultiGrid(grid);
}