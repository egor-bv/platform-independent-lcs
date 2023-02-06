#pragma once

#include "lcs_types.hpp"
#include "lcs_grid.hpp"
#include "lcs_common.hpp"


// Process cells in horizontal stripes with height SG_SIZE left to right, top to bottom
// NOTE: this would be greatly improved by avoiding writing to v_strands on every iteration
// for now it ends up slower than Antidiagonal variant because the distance
// between write and read on the same memory is too small
template<int SG_SIZE>
void
Lcs_Semi_Stripes_ST(const LcsInput &input, LcsContext &ctx)
{
	int m = input.a_size;
	int n = input.b_size;

	auto grid = make_embedding(input);
	{
		auto buf = make_buffers(grid);
		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);

			int stripe_size = SG_SIZE;
			int any_stripe_count = SmallestMultipleToFit(m, stripe_size);
			int complete_stripe_count = m / stripe_size;

			int left_border = 0;
			int right_border = n;
			int top_border = m;

			cgh.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{

				auto update_cell = [&](int i, int j)
				{
					update_cell_semilocal(acc.a, acc.b, acc.h_strands, acc.v_strands, i, j);
				};


				auto sg = item.get_sub_group();
				int sg_id = sg.get_local_linear_id();

				auto vertically_within_bounds = [&](int i, int j)
				{
					return i < top_border;
				};

				// Top stripe may be incomplete if it's unaligned with stripe_size,
				// run code path that handles vertically incomplete stripe
				if (complete_stripe_count < any_stripe_count)
				{
					int i0 = complete_stripe_count * stripe_size;
					stripe_iterate_with_predicate<SG_SIZE>(update_cell, vertically_within_bounds, sg, i0, left_border, right_border);
				}

				// Complete stripes can be processed with less checks
				int top_complete_stripe = complete_stripe_count;
				for (int stripe_idx = top_complete_stripe - 1; stripe_idx >= 0; --stripe_idx)
				{
					int i0 = stripe_idx * stripe_size;
					stripe_iterate<SG_SIZE>(update_cell, update_cell, sg, i0, left_border, right_border);
				}

			}); // end parallel_for
		}); // end submit
	} // end buffers lifetime
	copy_strands(ctx, grid);
}