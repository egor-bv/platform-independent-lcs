#pragma once

#include "lcs_common.hpp"

void Lcs_Hybrid(LcsInput &input, LcsContext &ctx)
{
	int m = input.size_a;
	int n = input.size_b;

	constexpr int SG_SIZE = 16;
	constexpr int NUM_THREADS = 16;
	int block_count_m = SG_SIZE;
	int block_count_n = NUM_THREADS;

	Assert(m % block_count_m == 0);
	Assert(n % block_count_n == 0);


	// make every block size be a multiple of cachline (assumed sizeof(Strand) == 4 here)
	int block_size_m = m / block_count_m; // SmallestMultipleToFit(m / block_count_m, 16) * 16;
	int block_size_n = n / block_count_n; // SmallestMultipleToFit(n / block_count_n, 16) * 16;

	int a_alloc_len = block_size_m * block_count_m;
	int b_alloc_len = block_size_n * block_count_n;
	
	int h_alloc_len = block_size_m * block_count_m * block_count_n;
	int v_alloc_len = block_size_n * block_count_m * block_count_n;

	int *as_data = new int[a_alloc_len];
	int *bs_data = new int[b_alloc_len];

	int *hs_data = new int[h_alloc_len];
	int *vs_data = new int[v_alloc_len];

	for (int i = 0; i < m; ++i)
	{
		as_data[i] = input.seq_a[m - i - 1];
	}

	for (int j = 0; j < n; ++j)
	{
		bs_data[j] = input.seq_b[j];
	}

	for (int block_j = 0; block_j < block_count_n; ++block_j)
	{
		for (int tile_i = 0; tile_i < block_size_m; ++tile_i)
		{
			for (int step = 0; step < block_count_m; ++step)
			{
				int idx = 0;
				idx += block_j * block_count_m * block_size_m;
				idx += tile_i * block_count_m;
				idx += step;
				hs_data[idx] = block_size_m - tile_i - 1;
			}
		}
	}

	for (int j = 0; j < n; ++j)
	{
		for (int step = 0; step < block_count_m; ++step)
		{
			int idx = 0;
			idx += j * block_count_m;
			idx += step;
			vs_data[idx] = block_size_m + j % block_size_n;
		}
	}


	{
		sycl::buffer<int, 1> as_buf(as_data, a_alloc_len);
		sycl::buffer<int, 1> bs_buf(bs_data, b_alloc_len);
		sycl::buffer<int, 1> hs_buf(hs_data, h_alloc_len);
		sycl::buffer<int, 1> vs_buf(vs_data, v_alloc_len);

		ctx.queue->submit([&](auto &cgh)
		{
			auto ias = as_buf.get_access<sycl::access::mode::read>(cgh);
			auto ibs = bs_buf.get_access<sycl::access::mode::read>(cgh);
			auto ihs = hs_buf.get_access<sycl::access::mode::read_write>(cgh);
			auto ivs = vs_buf.get_access<sycl::access::mode::read_write>(cgh);

			int local_size = SG_SIZE;
			int global_size = SG_SIZE * NUM_THREADS;

			cgh.parallel_for(
				sycl::nd_range<1>(global_size, local_size),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				auto sg_id = sg.get_local_linear_id();

				// for (int block_j = 0; block_j < block_count_n; ++block_j)
				{
					int block_j = item.get_group_linear_id();
					// row-major order
					for (int row = block_size_m - 1; row >= 0; --row)
					{
						int h_idx = (row * block_count_m + sg_id) + block_j * block_count_m * block_size_m;
						int a_idx = (row * block_count_m + sg_id);

						int h = ihs[h_idx];
						int a = ias[a_idx];

						#pragma unroll 8
						for (int col = 0; col < block_size_n; ++col)
						{
							int v_idx = (block_j * block_size_n + col) *block_count_m + sg_id;
							int b_idx = (block_j * block_size_n + col);

							int v = ivs[v_idx];
							int b = ibs[b_idx];

							// swap if ...
							bool need_swap = (a == b) || h > v;

							int new_v = need_swap ? h : v;
							int new_h = need_swap ? v : h;

							h = new_h;
							v = new_v;

							ivs[v_idx] = v;
						}
						ihs[h_idx] = h;
					}
				}
			});

		});
	}

	// TODO: cleanup
}