#pragma once

#include "utility.hpp"
#include "permutation.hpp"
#include "semi_lcs_types.hpp"

#include <CL/sycl.hpp>


// for now assume 
// m is a multiple of TILE_M * SG_SIZE
// n is a multiple of TILE_N

PermutationMatrix SemiLcs_Tiled(sycl::queue &q, const InputSequencePair &given)
{
	const int m = given.length_a;
	const int n = given.length_b;

	constexpr int SG_POW2 = 3;
	constexpr int SG_SIZE = 1 << SG_POW2;
	constexpr int TILE_M = 4;
	constexpr int TILE_N = 4;

	int num_tiles_h = SmallestMultipleToFit(TILE_M, m);
	int num_tiles_v = SmallestMultipleToFit(TILE_N, n);

	int full_m = num_tiles_h * TILE_M;
	int full_n = num_tiles_v * TILE_N;


	int *a_rev = new int[full_m];
	int *b = new int[full_n];
	// initialize inputs
	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_interleaved = mod * num_tiles_h + idx;
		a_rev[i_interleaved] = given.a[m - i - 1];
	}

	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_interleaved = mod * num_tiles_v + idx;
		b[j_interleaved] = given.b[j];
	}

	int *h_strands = new int[full_m];
	int *v_strands = new int[full_n];
	// initialize strands
	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_interleaved = mod * num_tiles_h + idx;
		h_strands[i_interleaved] = i;
	}
	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_interleaved = mod * num_tiles_v + idx;
		v_strands[j_interleaved] = j + m;
	}

	{
		// create buffers in a separate scope
		sycl::buffer<int, 1> buf_a_rev(a_rev, full_m);
		sycl::buffer<int, 1> buf_b(b, full_n);
		sycl::buffer<int, 1> buf_h_strands(h_strands, full_m);
		sycl::buffer<int, 1> buf_v_strands(v_strands, full_n);

		int fat_row_size = TILE_N * SG_SIZE;
		int num_fat_rows = SmallestMultipleToFit(fat_row_size, m);
		int num_rows = num_tiles_h / SG_SIZE;

		int i_step = num_tiles_h;
		int j_step = num_tiles_v;

		// perform main combing work!
		q.submit([&](auto &h)
			{
				auto ia_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
				auto ib = buf_b.get_access<sycl::access::mode::read>(h);
				auto ih_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
				auto iv_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

				h.parallel_for(
					sycl::nd_range<1>(SG_SIZE, SG_SIZE),
					[=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
					{
						auto sg = item.get_sub_group();
						int sg_id = sg.get_local_linear_id();

						auto process_block = [=](int i_low, int j_low)
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

						//auto process_block_bounded = [=](int i_low, int j_low)
						//{

						//};


						for (int row = 0; row < num_rows; ++row)
						{
							int i = (num_rows - row - 1) * SG_SIZE + sg_id;
											
							int tile_n = num_tiles_v;
							
							// first columns
							for (int horz_step = 1 - SG_SIZE; horz_step < 0; ++horz_step)
							{
								int j = horz_step + sg_id;
								if (j >= 0)
								{
									process_block(i, j);
								}
								sg.barrier();
							}

							// middle columns

							for (int horz_step = 0; horz_step < tile_n - SG_SIZE; ++horz_step)
							{
								int j = horz_step + sg_id;
								process_block(i, j);
								sg.barrier();
							}

							// last columns
							for (int horz_step = tile_n - SG_SIZE; horz_step < tile_n; ++horz_step)
							{
								int j = horz_step + sg_id;
								if (j < tile_n)
								{
									process_block(i, j);
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

	int *h_strands_deinterleaved = new int[m];
	int *v_strands_deinterleaved = new int[n];

	// deinterleave strands
	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_interleaved = mod * num_tiles_h + idx;
		h_strands_deinterleaved[i] = h_strands[i_interleaved];
	}
	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_interleaved = mod * num_tiles_v + idx;
		v_strands_deinterleaved[j] = v_strands[j_interleaved];
	}

	// write resulting permutation matrix
	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands_deinterleaved, m, v_strands_deinterleaved, n);

	delete[] h_strands_deinterleaved;
	delete[] v_strands_deinterleaved;
	delete[] h_strands;
	delete[] v_strands;
	delete[] a_rev;
	delete[] b;

	return result;
}

PermutationMatrix SemiLcs_Tiled_MT(sycl::queue &q, const InputSequencePair &given)
{
	const int m = given.length_a;
	const int n = given.length_b;


	// NOTE: want row height to be multiple of cacheline size (i.e 16 ints)
	constexpr int SG_POW2 = 4;
	constexpr int SG_SIZE = 1 << SG_POW2;
	constexpr int TILE_M = 8;
	constexpr int TILE_N = 8;

	int num_tiles_h = SmallestMultipleToFit(TILE_M, m);
	int num_tiles_v = SmallestMultipleToFit(TILE_N, n);

	int full_m = num_tiles_h * TILE_M;
	int full_n = num_tiles_v * TILE_N;


	int *a_rev = new int[full_m];
	int *b = new int[full_n];
	// initialize inputs
	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_interleaved = mod * num_tiles_h + idx;
		a_rev[i_interleaved] = given.a[m - i - 1];
	}

	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_interleaved = mod * num_tiles_v + idx;
		b[j_interleaved] = given.b[j];
	}

	int *h_strands = new int[full_m];
	int *v_strands = new int[full_n];
	// initialize strands
	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_interleaved = mod * num_tiles_h + idx;
		h_strands[i_interleaved] = i;
	}
	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_interleaved = mod * num_tiles_v + idx;
		v_strands[j_interleaved] = j + m;
	}


	{
		// create buffers in a separate scope
		sycl::buffer<int, 1> buf_a_rev(a_rev, full_m);
		sycl::buffer<int, 1> buf_b(b, full_n);
		sycl::buffer<int, 1> buf_h_strands(h_strands, full_m);
		sycl::buffer<int, 1> buf_v_strands(v_strands, full_n);


		int i_step = num_tiles_h;
		int j_step = num_tiles_v;

		constexpr int BLOCK_M = SG_SIZE;
		constexpr int BLOCK_N = SG_SIZE * 16;

		int num_blocks_m = num_tiles_h / BLOCK_M;
		int num_blocks_n = num_tiles_v / BLOCK_N;

		int block_diag_count = num_blocks_m + num_blocks_n - 1;

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

	int *h_strands_deinterleaved = new int[m];
	int *v_strands_deinterleaved = new int[n];

	// deinterleave strands
	for (int i = 0; i < m; ++i)
	{
		int mod = i % TILE_M;
		int idx = i / TILE_M;
		int i_interleaved = mod * num_tiles_h + idx;
		h_strands_deinterleaved[i] = h_strands[i_interleaved];
	}
	for (int j = 0; j < n; ++j)
	{
		int mod = j % TILE_N;
		int idx = j / TILE_N;
		int j_interleaved = mod * num_tiles_v + idx;
		v_strands_deinterleaved[j] = v_strands[j_interleaved];
	}

	// write resulting permutation matrix
	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands_deinterleaved, m, v_strands_deinterleaved, n);

	delete[] h_strands_deinterleaved;
	delete[] v_strands_deinterleaved;
	delete[] h_strands;
	delete[] v_strands;
	delete[] a_rev;
	delete[] b;

	return result;
}

