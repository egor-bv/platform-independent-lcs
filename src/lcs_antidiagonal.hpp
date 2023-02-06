#pragma once

#include "lcs_types.hpp"
#include "lcs_grid.hpp"
#include "lcs_common.hpp"

template<int SG_SIZE>
void 
Lcs_Semi_Antidiagonal_ST(const LcsInput &input, LcsContext &ctx)
{
	int m = input.a_size;
	int n = input.b_size;

	int diag_count = m + n - 1;
	auto grid = make_embedding(input);
	{
		auto buf = make_buffers(grid);
		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);
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

				for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
				{
					auto d = antidiag_at(diag_idx, m, n);
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
			});
		});
	}
	copy_strands(ctx, grid);
}
