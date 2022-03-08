#pragma once

#include "utility.hpp"
#include "permutation.hpp"
#include "semi_lcs_types.hpp"

#include <CL/sycl.hpp>


template<int SG_SIZE, int TILE_M, int TILE_N, int TARGET_THREADS>
PermutationMatrix SemiLcs_Tiled_Universal(sycl::queue &q, const InputSequencePair &given)
{
	// NOTE: to support smaller size needs different logic
	static_assert(SG_SIZE == 16);
	constexpr int CACHELINE_ELEMENTS = 64 / sizeof(int);

	int m = given.length_a;
	int n = given.length_b;

	// NOTE:
	// tile   = tile of original table processed in order
	// region = part of deinterleaved table, containg element with same modulo

	int region_m = SmallestMultipleToFit(TILE_M, m);
	int region_n = SmallestMultipleToFit(TILE_N, n);

	// ensure everything is aligned
	int region_m_padded = SmallestMultipleToFit(CACHELINE_ELEMENTS, region_m) * CACHELINE_ELEMENTS;
	int region_n_padded = SmallestMultipleToFit(CACHELINE_ELEMENTS, region_n) * CACHELINE_ELEMENTS;

	int alloc_m = region_m_padded * TILE_M;
	int alloc_n = region_n_padded * TILE_N;

	// deinterleave inputs
	int *a_rev = new int[alloc_m];
	int *b = new int[alloc_n];

	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_deinterleaved = mod * region_m_padded + idx;
		a_rev[i_deinterleaved] = given.a[m - i - 1];
	}

	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_deinterleaved = mod * region_n_padded + idx;
		b[j_deinterleaved] = given.b[j];
	}

	// deinterleave strands
	int *h_strands = new int[alloc_m] {};
	int *v_strands = new int[alloc_n] {};

	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_deinterleaved = mod * region_m_padded + idx;
		h_strands[i_deinterleaved] = i;
	}
	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_deinterleaved = mod * region_n_padded + idx;
		v_strands[j_deinterleaved] = j + m;
	}

	int i_step = region_m_padded;
	int j_step = region_n_padded;

	// separate scope to destroy buffers at the end
	{
		sycl::buffer<int, 1> buf_a_rev(a_rev, alloc_m);
		sycl::buffer<int, 1> buf_b(b, alloc_n);
		sycl::buffer<int, 1> buf_h_strands(h_strands, alloc_m);
		sycl::buffer<int, 1> buf_v_strands(v_strands, alloc_n);

		int BLOCK_M = SG_SIZE;
		int block_even_width = region_n_padded / TARGET_THREADS;
		int BLOCK_N = SmallestMultipleToFit(CACHELINE_ELEMENTS, block_even_width) * CACHELINE_ELEMENTS;
		
		// NOTE: for now irregular remainder is skipped
		int num_blocks_m = region_m_padded / BLOCK_M;
		int num_blocks_n = region_n_padded / BLOCK_N;

		int block_diag_count = num_blocks_m + num_blocks_n;

		for (int block_diag_idx = 0; block_diag_idx < block_diag_count; ++block_diag_idx)
		{
			int i_block_first = block_diag_idx < num_blocks_m ? block_diag_idx : num_blocks_m - 1;
			int j_block_first = block_diag_idx < num_blocks_m ? 0 : block_diag_idx - num_blocks_m + 1;

			// along antidiagonal, i goes down, j goes up
			int block_diag_len = Min(i_block_first + 1, num_blocks_n - j_block_first);
			int i_block_last = num_blocks_m - 1 - i_block_first;

			int local_size = SG_SIZE;
			int global_size = block_diag_len * SG_SIZE;

			q.submit([&](auto &h)
				{
					auto ia_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
					auto ib = buf_b.get_access<sycl::access::mode::read>(h);
					auto ih_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
					auto iv_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);


					h.parallel_for(
						sycl::nd_range<1>(global_size, local_size),
						[=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
						{
							auto sg = item.get_sub_group();
							auto sg_id = sg.get_local_linear_id();

							auto process_tile = [=](int i_low, int j_low)
							{
								int h[TILE_M];
								int a_sym[TILE_M];
								for (int vstep = 0; vstep < TILE_M; ++vstep)
								{
									h[vstep] = ih_strands[i_low + i_step * vstep];
									a_sym[vstep] = ia_rev[i_low + i_step * vstep];
								}

								for (int hstep = 0; hstep < TILE_N; ++hstep)
								{
									int v = iv_strands[j_low + j_step * hstep];
									int b_sym = ib[j_low + j_step * hstep];
									for (int vstep = TILE_M - 1; vstep >= 0; --vstep)
									{
										int sym_equal = a_sym[vstep] == b_sym;
										int has_crossing = h[vstep] > v;
										int need_swap = sym_equal || has_crossing;

										int new_h = need_swap ? v : h[vstep];
										int new_v = need_swap ? h[vstep] : v;

										h[vstep] = new_h;
										v = new_v;
									}

									iv_strands[j_low + j_step * hstep] = v;
								}

								for (int vstep = 0; vstep < TILE_M; ++vstep)
								{
									ih_strands[i_low + i_step * vstep] = h[vstep];
								}
							};

							int row = i_block_last + item.get_group_linear_id();
							int col = j_block_first + item.get_group_linear_id();

							{
								int i = row * SG_SIZE + sg_id;
								int left_border = col * BLOCK_N;
								int right_border = left_border + BLOCK_N;

								// first columns
								for (int horz_step = left_border + 1 - SG_SIZE; horz_step < left_border; ++horz_step)
								{
									int j = horz_step + sg_id;
									if (j >= left_border)
									{
										process_tile(i, j);
									}
									sg.barrier();
								}

								// middle columns
								for (int horz_step = left_border; horz_step < right_border - SG_SIZE; ++horz_step)
								{
									int j = horz_step + sg_id;
									process_tile(i, j);
									sg.barrier();
								}

								// last columns
								for (int horz_step = right_border - SG_SIZE; horz_step < right_border; ++horz_step)
								{
									int j = horz_step + sg_id;
									if (j < right_border)
									{
										process_tile(i, j);
									}
									sg.barrier();
								}

								sg.barrier();
							}

						}
					);
				}
			);
		}
	}

	// get back to interleaved strands
	int *h_strands_interleaved = new int[m];
	int *v_strands_interleaved = new int[n];

	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_deinterleaved = mod * region_m_padded + idx;
		h_strands_interleaved[i] = h_strands[i_deinterleaved];
	}
	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_deinterleaved = mod * region_n_padded + idx;
		v_strands_interleaved[j] = v_strands[j_deinterleaved];
	}

	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands_interleaved, m, v_strands_interleaved, n);

	delete[] a_rev;
	delete[] b;
	delete[] h_strands;
	delete[] v_strands;
	delete[] h_strands_interleaved;
	delete[] v_strands_interleaved;

	return result;
}