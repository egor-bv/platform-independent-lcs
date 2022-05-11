#pragma once

#include "lcs_types.hpp"
#include "lcs_common.hpp"
#include "lcs_grid.hpp"



template<typename Symbols, typename Strands>
void semi_process_cell(Symbols a, Symbols b, Strands h_strands, Strands v_strands, int i, int j)
{
	auto h_strand = h_strands[i];
	auto v_strand = v_strands[j];

	bool has_match = a[i] == b[j];
	bool has_crossing = h_strand > v_strand;
	bool need_swap = has_match || has_crossing;

	h_strands[i] = need_swap ? v_strand : h_strand;
	v_strands[j] = need_swap ? h_strand : v_strand;
}



void Lcs_Semi_Stripes(LcsInput &input, LcsContext &ctx)
{


	auto grid = GridEmbeddingSimple<int, int>(input.seq_a, input.size_a, input.seq_b, input.size_b);
	int m = grid.m;
	int n = grid.n;
	{
		auto buf = GridBuffers<int, int>::FromSimple(grid);


		ctx.queue->submit([&](sycl::handler &cgh)
		{
			constexpr int SG_SIZE = 16;
			auto acc = GridAccessors<int, int>(buf, cgh);

			int full_stripe_count = m / SG_SIZE;
			bool extra_stripe = full_stripe_count * SG_SIZE != m;

			int local_size = SG_SIZE;
			int global_size = SG_SIZE;

			auto process_cell = [=](int i, int j)
			{
				semi_process_cell(acc.a, acc.b, acc.h_strands, acc.v_strands, i, j);
			};


			cgh.parallel_for(
				sycl::nd_range<1>(local_size, global_size),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				int sg_id = sg.get_local_linear_id();

				int left_border = 0;
				int right_border = n;

				// top stripe is incomplete
				int top = full_stripe_count * SG_SIZE;
				if (extra_stripe)
				{
					int i = top + sg_id;
					bool inside = i < m;

					for (int j_first = left_border + 1 - SG_SIZE; j_first < left_border; ++j_first)
					{
						int j = j_first + sg_id;
						if (j >= left_border && j < right_border && inside)
						{
							process_cell(i, j);
						}
						sg.barrier();
					}

					int right_cap_first = Max(left_border, right_border - SG_SIZE);

					for (int j_first = left_border; j_first < right_cap_first; ++j_first)
					{

						int j = j_first + sg_id;
						if (inside)
						{
							process_cell(i, j);
						}
						sg.barrier();
					}

					for (int j_first = right_cap_first; j_first < right_border; ++j_first)
					{
						int j = j_first + sg_id;
						if (j < right_border && inside)
						{
							process_cell(i, j);
						}
						sg.barrier();
					}
					sg.barrier();
				}

				// full stripes don't need extra check
				for (int stripe = full_stripe_count - 1; stripe >= 0; --stripe)
				{
					int i = stripe * SG_SIZE + sg_id;

					#pragma unroll 16
					for (int j_first = left_border + 1 - SG_SIZE; j_first < left_border; ++j_first)
					{
						int j = j_first + sg_id;
						if (j >= left_border && j < right_border)
						{
							process_cell(i, j);
						}
						sg.barrier();
					}

					int right_cap_first = Max(left_border, right_border - SG_SIZE);

					#pragma unroll 16
					for (int j_first = left_border; j_first < right_cap_first; ++j_first)
					{

						int j = j_first + sg_id;
						process_cell(i, j);
						sg.barrier();
					}

					#pragma unroll 16
					for (int j_first = right_cap_first; j_first < right_border; ++j_first)
					{
						int j = j_first + sg_id;
						if (j < right_border)
						{
							process_cell(i, j);
						}
						sg.barrier();
					}
					sg.barrier();
				}
			});
		});
	}

	// after we're done, copy strands into context
	// this is for while new interface is not yet implemented...

	ctx.m = grid.m;
	ctx.n = grid.n;
	ctx.h_strands = new LcsContext::Index[ctx.m];
	ctx.v_strands = new LcsContext::Index[ctx.n];

	for (int i = 0; i < ctx.m; ++i)
	{
		ctx.h_strands[i] = grid.h_strands[i];
	}

	for (int j = 0; j < ctx.n; ++j)
	{
		ctx.v_strands[j] = grid.v_strands[j];
	}
}