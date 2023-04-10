#pragma once
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <utility>

#include "lcs_interface_internal.hpp"
#include "lcs_grid.hpp"
#include "lcs_buffers.hpp"

#include "sycl_utility.hpp"

#define FPGA_IMPL_IDX 102

#define FPGA_SIZE 256



#if FPGA_IMPL_IDX == 102

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

                    // K_PRINTF("No pipe: stripe_idx: %d, i_base: %d, hs[0]: %d\n", stripe_idx, i_base, hs[0].value);

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


                    // K_PRINTF("No pipe: stripe_idx: %d, i_base: %d, hs[0]: %d\n", stripe_idx, i_base, hs[0].value);
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

#define FPGA_PIPE_DEPTH 8


// Metaprogramming utility for defining many pipes 
template<int ...indices, typename F>
constexpr void
UnrolledLoop(std::integer_sequence<int, indices...>, F &&f)
{
    (f(std::integral_constant<int, indices>{}), ...);
}

template<int n, typename F>
constexpr void
UnrolledLoop(F &&f)
{
    UnrolledLoop(std::make_integer_sequence<int, n>{}, std::forward<F>(f));
}


template<int I>
class PipeId;

template<int I>
using Pipe = sycl::ext::intel::pipe<PipeId<I>, Word, FPGA_PIPE_DEPTH>;



template<typename ProducingPipe>
sycl::event
PipeLoad(sycl::queue *q, GridBuffersSeparate &buf)
{
    auto result = q->submit([&](sycl::handler &cgh) {
        
        int right_border = buf.v_strands_in.get_count();
        auto v_strands_in = AccessorRead(buf.v_strands_in, cgh);
        cgh.single_task([=]() {
            // K_PRINTF("Enter load\n", 0);
            for (int j = 0; j < right_border; ++j)
            {
                // K_PRINTF("LD j=%d\n", j);
                ProducingPipe::write(v_strands_in[j]);
            }
        }); // end single_task
    }); // end submit
    return result;
}


template<typename ConsumingPipe>
sycl::event
PipeStore(sycl::queue *q, GridBuffersSeparate &buf)
{
    auto result = q->submit([&](sycl::handler &cgh) {

        int right_border = buf.v_strands_in.get_count();
        auto v_strands_out = AccessorWrite(buf.v_strands_out, cgh);
        cgh.single_task([=]() {
            // K_PRINTF("Enter store\n", 0);
            for (int j = 0; j < right_border; ++j)
            {
                // K_PRINTF("ST j=%d\n", j);
                auto v = ConsumingPipe::read();
                v_strands_out[j] = v;
            }
        }); // end single_task
    }); // end submit
    return result;
}


template<int STRIPE_M, int IDX, typename InPipe, typename OutPipe>
sycl::event
PipeWork(int stripe_idx, int i_base, sycl::queue *q, GridBuffersSeparate &buf)
{
    auto result = q->submit([&](sycl::handler &cgh) {

        int right_border = buf.v_strands_in.get_count();

        auto acc_a = AccessorRead(buf.a, cgh);
        auto acc_b = AccessorRead(buf.b, cgh);

        // printf("IDX: %d, stripe_idx: %d\n", IDX, stripe_idx);
        auto acc_h_strands = AccessorReadWrite(buf.h_strands_inout[stripe_idx], cgh);

        cgh.single_task([=]() [[intel::kernel_args_restrict]] {
            
            Word as[STRIPE_M];
            Word hs[STRIPE_M];

            [[intel::ivdep]]
            for (int ii = 0; ii < STRIPE_M; ++ii)
            {
                as[ii] = acc_a[i_base + ii];
            }
            
            [[intel::ivdep]]
            for (int ii = 0; ii < STRIPE_M; ++ii)
            {
                hs[ii] = acc_h_strands[ii];
            }
            
            // K_PRINTF("Working: stripe_idx: %d, i_base: %d, hs[0]: %d\n", stripe_idx, i_base, hs[0].value);
            [[intel::ivdep]]
            for (int j = 0; j < right_border; ++j)
            {
                // K_PRINTF("work read j=%d\n", j);
                Word b_symbol = acc_b[j];
                Word v_strand = InPipe::read();
                // if (stripe_idx == 5) K_PRINTF("%d ", v_strand.value);
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

                // K_PRINTF("work write j=%d\n", j);
                OutPipe::write(v_strand);
            }
            // K_PRINTF("----\n", 0);
            // K_PRINTF("Working: stripe_idx: %d, i_base: %d, hs[0]: %d\n", stripe_idx, i_base, hs[0].value);
            [[intel::ivdep]]
            for (int ii = 0; ii < STRIPE_M; ++ii)
            {
                acc_h_strands[ii] = hs[ii];
            }
        }); // end single_task
    }); // end submit
    return result;
}







#if FPGA_IMPL_IDX == 102


template<int NUM_UNITS>
void Lcs_Semi_Fpga_Pipes(LcsProblemContext &ctx)
{
    constexpr int STRIPE_M = FPGA_SIZE;
    constexpr int MULTI_STRIPE_M = STRIPE_M * NUM_UNITS;
    auto grid = make_grid_embedding_simple(ctx, MULTI_STRIPE_M, 1);

    int right_border = grid.shape.n_aligned;
    int top_border = grid.shape.m_aligned;
    int multi_stripe_count = grid.shape.m_aligned / MULTI_STRIPE_M;

    {
        auto buf = make_buffers_separate(grid, multi_stripe_count * NUM_UNITS);

        for (int multi_stripe_idx = 0; multi_stripe_idx < multi_stripe_count; ++multi_stripe_idx)
        {
            // single stripe is processed by multiple kernels at once

            PipeLoad<Pipe<0>>(ctx.queue, buf);
            // PipeStore<Pipe<0>>(ctx.queue, buf);

            UnrolledLoop<NUM_UNITS>([&](auto IDX) {
                int stripe_idx = multi_stripe_count * NUM_UNITS - (1 + multi_stripe_idx * NUM_UNITS + IDX);
                int i_base = stripe_idx * STRIPE_M;
                PipeWork<STRIPE_M, IDX, Pipe<IDX>, Pipe<IDX + 1>>(stripe_idx, i_base, ctx.queue, buf);
            });


            PipeStore<Pipe<NUM_UNITS>>(ctx.queue, buf);


            buf.Swap();
            ctx.queue->wait();

        } // end for
    } // end bufers lifetime
    if (multi_stripe_count % 2 == 0)
    {
        Swap(grid.v_strands, grid.v_strands_out);
    }
    copy_strands_and_fixup(ctx, grid);
}

#endif