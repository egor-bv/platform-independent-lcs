#pragma once
#include "permutation.hpp"
#include "utility.hpp"
#include "lcs.hpp"

#include <CL/sycl.hpp>

struct StrandMap
{
	const int m;
	const int n;
	const int pad;

	// NOTE(Egor): h_strands are padded at the beginning, v_strands on both sides
	int *padded_h_strands;
	int *padded_v_strands;

	StrandMap(int m_, int n_, int pad_)
		:m(m_), n(n_), pad(pad_)
	{
		padded_h_strands = new int[m + pad];
		padded_v_strands = new int[n + pad + pad];

		for (int i = 0; i < m; ++i) h_strands()[i] = i;
		for (int j = 0; j < n; ++j) v_strands()[j] = j + m;
	}

	~StrandMap()
	{
		delete[] padded_h_strands;
		delete[] padded_v_strands;
	}

	int padded_m() const
	{
		return m + pad;
	}

	int padded_n() const
	{
		return n + pad + pad;
	}


	int *h_strands()
	{
		return padded_h_strands + pad;
	}

	int *v_strands()
	{
		return padded_v_strands + pad;
	}
};


struct StairsBlock
{
	int i_first;
	int j_first;

	void get_ij(int row, int &i, int &j)
	{
		i = i_first + row;
		j = j_first - row;
	}
};

struct StairsBlockDiagonal
{
	int block_count;
	int block_m;
	int block_n;

	int i_first;
	int j_first;

	StairsBlock get_block(int idx)
	{
		StairsBlock result = {};
		result.i_first = i_first - block_m * idx;
		result.j_first = j_first + (block_m + block_n - 1) * idx;
		return result;
	}
};


struct MatrixIterationScheme
{
	const int m;
	const int n;
	const int block_m;
	const int block_n;

	int big_m;
	int big_n;
	int block_diagonal_count;


	MatrixIterationScheme(const int m_, const int n_, const int block_m_, const int block_n_)
		:m(m_), n(n_), block_m(block_m_), block_n(block_n_)
	{
		big_m = SmallestMultipleToFit(block_m, m);
		int overall_leftmost = -m + big_m;
		big_n = SmallestMultipleToFit(block_n, n - overall_leftmost);
		block_diagonal_count = big_m + big_n - 1;
	}

	StairsBlockDiagonal get_block_diagonal(int idx)
	{
		bool starts_on_left = idx < big_m;
		StairsBlockDiagonal d = {};
		int block_i_first = starts_on_left ? idx : big_m - 1;
		int block_j_first = starts_on_left ? 0 : idx - big_m + 1;
		d.block_m = block_m;
		d.block_n = block_n;
		d.block_count = Min(block_i_first + 1, big_n - block_j_first);

		// clip diagonal against rectangle, only leaving blocks that touch it
		{
			int horz_coverage = block_m + block_n - 1;
			int horz_stride = block_m - 1;

			int i_end = block_i_first - d.block_count;
			int j_end = block_j_first + d.block_count;

			int leftmost = -horz_stride * (block_i_first + 1) + block_j_first * block_n;
			int rightmost = -horz_stride * (i_end + 1) + j_end * block_n;

			int extra_left = Max(0, -leftmost);
			int extra_right = Max(0, rightmost - n);
			int clip_first = extra_left / horz_coverage;
			int clip_last = extra_right / horz_coverage;

			block_i_first -= clip_first;
			block_j_first += clip_first;
			d.block_count -= clip_first + clip_last;
		}

		// from block to real coordinates
		d.i_first = block_i_first * block_m;
		d.j_first = block_j_first * block_n - block_i_first * (block_m - 1);

		return d;
	}
};

struct CombingContext
{
	StrandMap &smap;
	const InputSequencePair &spair;

	using IntBuffer = sycl::buffer<int, 1>;
	using LocalAccessor = sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local>;
	using ROAccessor = sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	using RWAccessor = sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;

	IntBuffer buf_a;
	IntBuffer buf_b;
	IntBuffer buf_h_strands;
	IntBuffer buf_v_strands;

	CombingContext(const InputSequencePair &input_pair, StrandMap &strand_map)
		: smap(strand_map)
		, spair(input_pair)
		, buf_a(input_pair.a, input_pair.length_a)
		, buf_b(input_pair.b, input_pair.length_b)
		, buf_h_strands(strand_map.padded_h_strands, strand_map.padded_m())
		, buf_v_strands(strand_map.padded_v_strands, strand_map.padded_n())
	{

	}

};

void CombingPass(sycl::queue q, CombingContext &ctx, StairsBlockDiagonal &diag)
{
	for (int block_idx = 0; block_idx < diag.block_count; ++block_idx)
	{
		auto block = diag.get_block(block_idx);
		for (int col_idx = 0; col_idx < diag.block_n; ++col_idx)
		{
			for (int row_idx = 0; row_idx < diag.block_m; ++row_idx)
			{
				int i, j;
				block.get_ij(row_idx, i, j);

				auto a_sym = ctx.spair.a[i];
				auto b_sym = ctx.spair.b[j];
			}
		}
	}
}



// same as below but with loop order swapped
long long
StickyBraidStairs(const InputSequencePair &p)
{
	int m = p.length_a;
	int n = p.length_b;
	const int block_m = 256;
	const int block_n = 32;

	StrandMap strand_map(m, n, block_m);

	// this variation does not use SYCL
	// CombingContext ctx(p, strand_map);

	MatrixIterationScheme scheme(m, n, block_m, block_n);

	auto *h_strands = strand_map.padded_h_strands;
	auto *v_strands = strand_map.padded_v_strands;

	size_t total_inner_loops = 0;
	for (int diag_idx = 0; diag_idx < scheme.block_diagonal_count; ++diag_idx)
	{
		auto diag = scheme.get_block_diagonal(diag_idx);
		for (int block_idx = 0; block_idx < diag.block_count; ++block_idx)
		{
			auto block = diag.get_block(block_idx);

			// clip whole antidiagonals within block
			int clip_left = Max(-block.j_first, 0);
			int clip_right = Max(block.j_first + block_n - block_m - n, 0);
			int clip_bottom = Max(block.i_first + block_m - m, 0);
			int horz_steps = block_n - clip_left - clip_right;
			block.j_first += clip_left;

			int i_first, j_first;
			block.get_ij(0, i_first, j_first);

			for (int step = 0; step < horz_steps; ++step)
			{
				for (int local_id = 0; local_id < block_m; ++local_id)
				{
					++total_inner_loops;
					int i = i_first + local_id;
					int j = j_first - local_id + step;
					int clamped_i = Clamp(i, 0, m - 1);
					int clamped_j = Clamp(j, 0, n - 1);

					bool clip = j < 0 || j >= n || i >= m;

					int h_index = m - 1 - i + strand_map.pad;
					int v_index = j + strand_map.pad;

					int h_strand = h_strands[h_index];
					int v_strand = v_strands[v_index];

					bool need_swap = (p.a[clamped_i] == p.b[clamped_j] || h_strand > v_strand) && !clip;
					{
						h_strands[h_index] = need_swap ? v_strand : h_strand;
						v_strands[v_index] = need_swap ? h_strand : v_strand;
					}

				}
			}

			//for (int local_id = 0; local_id < block_m - clip_bottom; ++local_id)
			//{
			//	int i, j;
			//	block.get_ij(local_id, i, j);
			//	int clamped_i = Clamp(i, 0, m - 1);


			//	int j_left_border = strand_map.pad;
			//	int j_right_border = n + strand_map.pad;
			//

			//	int h_index = m - 1 - i + strand_map.pad;
			//	int h_strand = h_strands[h_index];
			//	
			//	for (int step = 0; step < horz_steps; ++step)
			//	{
			//		bool clip = j < 0 || j >= m;

			//		int clamped_j = Clamp(j, 0, n - 1);
			//		
			//		int v_index = j + strand_map.pad;

			//		int h_strand = h_strands[h_index];
			//		int v_strand = v_strands[v_index];

			//		bool need_swap = (p.a[clamped_i] == p.b[clamped_j] || h_strand > v_strand) && !clip;
			//		{
			//			h_strands[h_index] = need_swap ? v_strand : h_strand;
			//			v_strands[v_index] = need_swap ? h_strand : v_strand;
			//		}
			//		++j;
			//	}
			//}
		}
	}

	std::cout << "\nTotal inner loops: " << total_inner_loops << "\n";
	auto perm = PermutationMatrix::FromStrands(strand_map.h_strands(), strand_map.m, strand_map.v_strands(), strand_map.n);

	return hash(perm, perm.size);
}



long long
StickyBraidStairsLinear(const InputSequencePair &p)
{
	int m = p.length_a;
	int n = p.length_b;
	const int block_m = 256;
	const int block_n = 128;

	StrandMap strand_map(m, n, block_m);

	// this variation does not use SYCL
	// CombingContext ctx(p, strand_map);

	MatrixIterationScheme scheme(m, n, block_m, block_n);

	auto *h_strands = strand_map.padded_h_strands;
	auto *v_strands = strand_map.padded_v_strands;

	for (int diag_idx = 0; diag_idx < scheme.block_diagonal_count; ++diag_idx)
	{
		auto diag = scheme.get_block_diagonal(diag_idx);
		for (int block_idx = 0; block_idx < diag.block_count; ++block_idx)
		{
			auto block = diag.get_block(block_idx);

			// clip whole antidiagonals within block
			int clip_left = Max(-block.j_first, 0);
			int clip_right = Max(block.j_first + block_n - block_m - n, 0);
			int clip_bottom = Max(block.i_first + block_m - m, 0);
			int horz_steps = block_n - clip_left - clip_right;
			block.j_first += clip_left;

			for (int local_id = 0; local_id < block_m - clip_bottom; ++local_id)
			{
				int i, j;
				block.get_ij(local_id, i, j);
				int clamped_i = Clamp(i, 0, m - 1);


				int j_left_border = strand_map.pad;
				int j_right_border = n + strand_map.pad;


				int h_index = m - 1 - i + strand_map.pad;
				int h_strand = h_strands[h_index];

				for (int step = 0; step < horz_steps; ++step)
				{
					bool clip = j < 0 || j >= m;

					int clamped_j = Clamp(j, 0, n - 1);

					int v_index = j + strand_map.pad;

					int h_strand = h_strands[h_index];
					int v_strand = v_strands[v_index];

					bool need_swap = (p.a[clamped_i] == p.b[clamped_j] || h_strand > v_strand) && !clip;
					{
						h_strands[h_index] = need_swap ? v_strand : h_strand;
						v_strands[v_index] = need_swap ? h_strand : v_strand;
					}
					++j;
				}
			}
		}
	}

	auto perm = PermutationMatrix::FromStrands(strand_map.h_strands(), strand_map.m, strand_map.v_strands(), strand_map.n);

	return hash(perm, perm.size);
}

