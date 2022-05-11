#pragma once

#include "permutation.hpp"

void get_vert_slice(PermutationMatrix &p, int col_start, int col_end,
	int *cur_to_prev_row_mapping, PermutationMatrix &p_next)
{
	int idx = 0;
	for (int row = 0; row < p.row_count(); ++row)
	{
		int col = p.get_col(row);
		if (col >= col_start && col < col_end)
		{
			cur_to_prev_row_mapping[idx] = row;
			p_next.set_point(idx, col - col_start);
			idx++;
		}
	}
}


void get_horz_slice(PermutationMatrix &q, int row_start, int row_end,
	int *cur_to_prev_col_mapping, PermutationMatrix &q_next)
{
	int idx = 0;
	for (int col = 0; col < q.col_count(); ++col)
	{
		int row = q.get_row(col);
		if (row >= row_start && row < row_end)
		{
			cur_to_prev_col_mapping[idx] = col;
			q_next.set_point(row - row_start, idx);
			++idx;
		}
	}
}


namespace dom_sum_count
{
namespace bottom_right
{
int left_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (col == 0) return sum;
	int row_cap = p.get_row(col - 1);
	if (row_cap != -1 && row_cap < row) sum--;
	return sum;
}

int right_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (col >= p.col_count()) return 0;
	int row_cap = p.get_row(col);
	if (row_cap != -1 && row_cap < row) sum++;
	return sum;
}


int up_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (row == 0) return sum;
	int col_cap = p.get_col(row - 1);
	if (col_cap != -1 && col_cap < col) sum--;
	return sum;
}


int down_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (row >= p.row_count()) return 0;
	int col_cap = p.get_col(row);
	if (col_cap != -1 && col_cap < col) sum++;
	return sum;
}



};

namespace top_left
{

int left_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (col == 0) return sum;
	int row_cap = p.get_row(col - 1);
	if (row_cap >= row && row_cap != -1) sum++;
	return sum;
}

int right_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (col >= p.col_count()) return 0;
	int row_cap = p.get_row(col);
	if (row_cap >= row && row_cap != -1) sum--;
	return sum;
}


int up_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (row == 0) return sum;
	int col_cap = p.get_col(row - 1);
	if (col_cap >= col && col_cap != -1) sum++;
	return sum;
}


int down_move(int row, int col, int sum, PermutationMatrix &p)
{
	if (row >= p.row_count()) return 0;
	int col_cap = p.get_col(row);
	if (col_cap >= col && col_cap != -1) sum--;
	return sum;
}

};
};

#include <vector>

void ant_passage(PermutationMatrix &r_lo, PermutationMatrix &r_hi, int n,
	std::vector<int> &good_row_pos, std::vector<int> &good_col_pos)
{

	int end_row = -1;
	int end_col = r_lo.col_count();
	int cur_row = r_hi.row_count();
	int cur_col = -1;

	int rhi = 0;
	int rlo = 0;

	bool went_right = false;
	for (;;)
	{
		if (end_col == cur_col && end_row == cur_row) break;
		if (cur_row == 0) break;
		if (cur_col == n) break;

		int dominance_row = cur_row - 1;
		int dominance_col = cur_col + 1;

		if (went_right)
		{
			rhi = dom_sum_count::bottom_right::right_move(dominance_row, dominance_col - 1, rhi, r_hi);
			rlo = dom_sum_count::top_left::right_move(dominance_row, dominance_col - 1, rlo, r_lo);
		}
		else
		{
			rhi = dom_sum_count::bottom_right::up_move(dominance_row + 1, dominance_col, rhi, r_hi);
			rlo = dom_sum_count::top_left::up_move(dominance_row + 1, dominance_col, rlo, r_lo);
		}

		if (rhi - rlo < 0)
		{
			went_right = true;
			cur_col++;
		}
		else if (rhi - rlo == 0)
		{
			went_right = false;
			cur_row--;
		}
		else
		{
			printf("Unreachable!\n");
		}

		if (dominance_col > 0)
		{
			int delta_top_left =
				dom_sum_count::bottom_right::left_move(dominance_row, dominance_col, rhi, r_hi) -
				dom_sum_count::top_left::left_move(dominance_row, dominance_col, rlo, r_lo);

			int delta_bottom_right =
				dom_sum_count::bottom_right::down_move(dominance_row, dominance_col, rhi, r_hi) -
				dom_sum_count::top_left::down_move(dominance_row, dominance_col, rlo, r_lo);

			if (delta_top_left < 0 && delta_bottom_right > 0)
			{
				good_row_pos.push_back(dominance_row);
				good_col_pos.push_back(dominance_col - 1);
			}
		}

	}
}

void inverse_mapping(PermutationMatrix &shrinked, int *row_mapper, int *col_mapper, PermutationMatrix &flattened)
{
	flattened.unset_all();
	for (int cur_col = 0; cur_col < shrinked.col_count(); ++cur_col)
	{
		int old_col = col_mapper[cur_col];
		int cur_row = shrinked.get_row(cur_col);
		int old_row = row_mapper[cur_row];

		flattened.set_point(old_row, old_col);
	}
}

void steady_ant_seq(PermutationMatrix &p, PermutationMatrix &q,
	int *current_mem,
	int *free_space_mem,
	int *indices_mem)
{
	int n = p.row_count();

	if (n <= 1) // When have precalculated value for small sizes
	{
		return;
	}

	int splitter = n / 2;
	int size_lo = splitter;
	int size_hi = n - splitter;

	int *p_lo_row_mapper = indices_mem;
	auto p_lo = PermutationMatrix::Preallocated(size_lo, free_space_mem, free_space_mem + size_lo);
	get_vert_slice(p, 0, splitter, p_lo_row_mapper, p_lo);

	int *q_lo_col_mapper = indices_mem + size_lo;
	auto q_lo = PermutationMatrix::Preallocated(size_lo, free_space_mem + 2 * size_lo, free_space_mem + 3 * size_lo);
	get_horz_slice(q, 0, splitter, q_lo_col_mapper, q_lo);

	int *p_hi_row_mapper = indices_mem + size_lo * 2;
	auto p_hi = PermutationMatrix::Preallocated(size_hi, free_space_mem + 4 * size_lo, free_space_mem + 4 * size_lo + size_hi);
	get_vert_slice(p, splitter, n, p_hi_row_mapper, p_hi);


	int *q_hi_col_mapper = indices_mem + size_lo * 2 + size_hi;
	auto q_hi = PermutationMatrix::Preallocated(size_hi,
		free_space_mem + 4 * size_lo + 2 * size_hi,
		free_space_mem + 4 * size_lo + 3 * size_hi);
	get_horz_slice(q, splitter, q.row_count(), q_hi_col_mapper, q_hi);


	steady_ant_seq(p_lo, q_lo, free_space_mem, current_mem, indices_mem + 2 * n);
	steady_ant_seq(p_hi, q_hi, free_space_mem + 4 * size_lo, current_mem + 4 * size_lo, indices_mem + 2 * n);

	// HACK: need to allocate these separately?
	auto r_lo = p;
	auto r_hi = q;

	inverse_mapping(p_lo, p_lo_row_mapper, q_lo_col_mapper, r_lo);
	inverse_mapping(p_hi, p_hi_row_mapper, q_hi_col_mapper, r_hi);

	std::vector<int> good_row_pos;
	std::vector<int> good_col_pos;
	ant_passage(r_lo, r_hi, n, good_row_pos, good_col_pos);

	for (int i = 0; i < n; ++i)
	{
		int col = r_hi.get_col(i);
		if (col != -1)
		{
			r_lo.set_point(i, col);
		}
	}

	for (int i = 0; i < good_col_pos.size(); ++i)
	{
		int row = good_row_pos[i];
		int col = good_col_pos[i];
		r_lo.set_point(row, col);
	}

}


template<bool RowGlue>
PermutationMatrix staggered_multiply(PermutationMatrix &p, PermutationMatrix &q, int k)
{
	// constexpr bool RowGlue = true;
	constexpr int NO_POINT = -1;

	auto product = PermutationMatrix(p.row_count() + q.row_count() - k);

	// no change
	if (k == 0)
	{
		for (int i = 0; i < p.row_count(); ++i)
		{
			int col = p.get_col(i);

			if (RowGlue)
			{
				if (col != NO_POINT)
				{
					product.set_point(i + q.row_count(), col + q.col_count());
				}
			}
			else
			{
				if (col != NO_POINT)
				{
					product.set_point(i, col);
				}
			}
		}

		for (int i = 0; i < q.row_count(); ++i)
		{
			int col = q.get_col(i);

			if (RowGlue)
			{
				if (col != NO_POINT)
				{
					product.set_point(i, col);
				}
			}
			else
			{
				if (col != NO_POINT)
				{
					product.set_point(i + p.row_count(), col + p.col_count());
				}
			}
		}

	}
	else
	{
		int *memory_block = new int[k * 8 * 2];

		int *mapping_row = new int[k];
		int *mapping_col = new int[k];

		auto p_red = PermutationMatrix::Preallocated(k, memory_block, memory_block + k);
		auto q_red = PermutationMatrix::Preallocated(k, memory_block + 2 * k, memory_block + 3 * k);

		int *free_block_1 = memory_block;
		int *free_block_2 = memory_block + 4 * k;
		int *free_indices_block = memory_block + 8 * k;

		if (RowGlue)
		{
			get_vert_slice(p, 0, k, mapping_row, p_red);
			get_horz_slice(q, q.row_count() - k, q.row_count(), mapping_col, q_red);
		}
		else
		{
			get_vert_slice(p, p.row_count() - k, p.row_count(), mapping_row, p_red);
			get_horz_slice(q, 0, k, mapping_col, q_red);
		}

		// Sequential impl

		steady_ant_seq(p_red, q_red, free_block_1, free_block_2, free_indices_block);

		// result is in p_red
		for (int i = 0; i < p_red.row_count(); ++i)
		{
			int old_col = p_red.get_col(i);
			int cur_col = mapping_col[old_col];
			int cur_row = mapping_row[i];
			if (RowGlue)
			{
				product.set_point(q.row_count() - k + cur_row, cur_col);
			}
			else
			{
				product.set_point(cur_row, cur_col + p.col_count() - k);
			}
		}

		if (RowGlue)
		{
			for (int j = k; j < p.col_count(); ++j)
			{
				int row = p.get_row(j);
				if (row != -1) product.set_point(row + q.row_count() - k, j + q.col_count() - k);
			}
		}
		else
		{
			for (int j = 0; j < p.col_count() - k; ++j)
			{
				int row = p.get_row(j);
				if (row != -1) product.set_point(row, j);
			}
		}

		if (RowGlue)
		{
			for (int i = 0; i < q.row_count() - k; ++i)
			{
				int col = q.get_col(i);
				if (col != -1) product.set_point(i, col);
			}
		}
		else
		{
			for (int i = k; i < q.row_count(); ++i)
			{
				int col = q.get_col(i);
				if (col != -1) product.set_point(i - k + p.row_count(), col - k + p.col_count());
			}
		}

		printf("Done with multiply...\n");
		// Cleanup intermediate values...
	}

	return product;
}