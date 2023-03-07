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


void Lcs_Semi_FpgaLike(const LcsInput &input, LcsContext &ctx)
{
    constexpr int STRIPE_M = 32;
    int stripe_count = input.a_size / STRIPE_M;
    int right_border = input.b_size;

    auto grid = make_embedding(input);
    {
        auto buf = make_buffers(grid);
        for (int stripe_idx = 0; stripe_idx < stripe_count; ++stripe_idx)
        {
            int i_base = (stripe_count - stripe_idx - 1) * STRIPE_M;
            ctx.queue->submit([&](auto &cgh)
            {
                auto acc = make_accessors(buf, cgh);
                cgh.single_task([=]()
                {
                    int as[STRIPE_M];
                    int hs[STRIPE_M];

                    #pragma unroll
                    //[[intel::ivdep]]
                    for (int ii = 0; ii < STRIPE_M; ++ii)
                    {
                        as[ii] = acc.a[i_base + ii];
                    }

                    #pragma unroll
                    //[[intel::ivdep]]
                    for (int ii = 0; ii < STRIPE_M; ++ii)
                    {
                        hs[ii] = acc.h_strands[i_base + ii];
                    }

                    //[[intel::ivdep]]
                    for (int j = 0; j < right_border; ++j)
                    {
                        int b_symbol = acc.b[j];
                        int v_strand = acc.v_strands[j];

                        #pragma unroll
                        //[[intel::ivdep]]
                        for (int ii = STRIPE_M - 1; ii >= 0; --ii)
                        {
                            int a_symbol = as[ii];
                            int h_strand = hs[ii];

                            bool has_match = a_symbol == b_symbol;
                            bool has_crossing = h_strand > v_strand;
                            bool need_swap = has_match || has_crossing;
                            int h_strand_new = need_swap ? v_strand : h_strand;
                            int v_strand_new = need_swap ? h_strand : v_strand;

                            hs[ii] = h_strand_new;
                            v_strand = v_strand_new;
                        }

                        acc.v_strands[j] = v_strand;
                    }

                    #pragma unroll
                    //[[intel::ivdep]]
                    for (int ii = 0; ii < STRIPE_M; ++ii)
                    {
                        acc.h_strands[i_base + ii] = hs[ii];
                    }
                }); // end single_task
            }); // end submit
            printf("i_base: %d\n", i_base);
        } // end for 
    } // end bufers lifetime
    copy_strands(ctx, grid);
}