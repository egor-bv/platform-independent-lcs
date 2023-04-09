#pragma once
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <utility>

#include "lcs_interface_internal.hpp"
#include "lcs_grid.hpp"


#define FPGA_IMPL_IDX 101

#define FPGA_SIZE 256



#if FPGA_IMPL_IDX == 101

void Lcs_Semi_Fpga(LcsProblemContext &ctx)
{
    constexpr int STRIPE_M = FPGA_SIZE;
    
    auto grid = make_grid_embedding_simple(ctx, STRIPE_M, 1);
    
    int right_border = grid.shape.n_aligned;
    int top_border = grid.shape.m_aligned;
    int stripe_count = grid.shape.m_aligned / STRIPE_M;


    {
        auto buf = make_buffers(grid);
        for (int stripe_idx = 0; stripe_idx < stripe_count; ++stripe_idx)
        {
            int i_base = top_border - (stripe_idx + 1) * STRIPE_M;
            ctx.queue->submit([&](auto &cgh)
            {
                auto acc = make_accessors(buf, cgh);
                cgh.single_task([=]()
                {
                    Word as[STRIPE_M];
                    Word hs[STRIPE_M];

                    // #pragma unroll
                    [[intel::ivdep]]
                    for (int ii = 0; ii < STRIPE_M; ++ii)
                    {
                        as[ii] = acc.a[i_base + ii];
                    }

                    // #pragma unroll
                    [[intel::ivdep]]
                    for (int ii = 0; ii < STRIPE_M; ++ii)
                    {
                        hs[ii] = acc.h_strands[i_base + ii];
                    }

                    [[intel::ivdep]]
                    for (int j = 0; j < right_border; ++j)
                    {
                        Word b_symbol = acc.b[j];
                        Word v_strand = acc.v_strands[j];

                        #pragma unroll
                        [[intel::ivdep]]
                        for (int ii = STRIPE_M - 1; ii >= 0; --ii)
                        {
                            Word a_symbol = as[ii];
                            Word h_strand = hs[ii];

                            bool has_match = a_symbol == b_symbol;
                            bool has_crossing = h_strand > v_strand;
                            bool need_swap = has_match || has_crossing;
                            Word h_strand_new = need_swap ? v_strand : h_strand;
                            Word v_strand_new = need_swap ? h_strand : v_strand;

                            hs[ii] = h_strand_new;
                            v_strand = v_strand_new;
                        }

                        acc.v_strands[j] = v_strand;
                    }

                    // #pragma unroll
                    [[intel::ivdep]]
                    for (int ii = 0; ii < STRIPE_M; ++ii)
                    {
                        acc.h_strands[i_base + ii] = hs[ii];
                    }
                }); // end single_task
            }); // end submit
        } // end for 
    } // end bufers lifetime
    copy_strands_and_fixup(ctx, grid);
}


#endif