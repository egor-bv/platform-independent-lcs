#pragma once

#include "lcs_common.hpp"

void Lcs_Semi_Antidiagonal(LcsInput &input, LcsContext &ctx)
{
	InitInputs(input, ctx);
	InitStrands(ctx);

	Assert(ctx.queue);

	auto m = ctx.m;
	auto n = ctx.n;

	auto *a_data = ctx.a;
	auto *b_data = ctx.b;

	auto *h_strands_data = ctx.h_strands;
	auto *v_strands_data = ctx.v_strands;


	using Symbol = LcsContext::Symbol;
	using Index = LcsContext::Index;

	Index diag_count = m + n - 1;
	constexpr int SG_SIZE = 16;
	{
		sycl::buffer<Symbol, 1> a_buf(a_data, m);
		sycl::buffer<Symbol, 1> b_buf(b_data, n);
		sycl::buffer<Index, 1> h_strands_buf(h_strands_data, m);
		sycl::buffer<Index, 1> v_strands_buf(v_strands_data, n);

		ctx.queue->submit([&](auto &cgh)
		{
			auto a = a_buf.get_access<sycl::access::mode::read>(cgh);
			auto b = b_buf.get_access<sycl::access::mode::read>(cgh);
			auto h_strands = h_strands_buf.get_access<sycl::access::mode::read_write>(cgh);
			auto v_strands = v_strands_buf.get_access<sycl::access::mode::read_write>(cgh);


			cgh.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				auto sg_id = sg.get_local_linear_id();

				auto update_cell = [&](Index i, Index j)
				{
					auto a_sym = a[i];
					auto b_sym = b[j];
					auto h_strand = h_strands[i];
					auto v_strand = v_strands[j];

					auto has_match = a_sym == b_sym;
					auto has_crossing = h_strand > v_strand;
					auto need_swap = has_match || has_crossing;

					h_strands[i] = need_swap ? v_strand : h_strand;
					v_strands[j] = need_swap ? h_strand : v_strand;
				};


				for (Index diag_idx = 0; diag_idx < diag_count; ++diag_idx)
				{
					Index i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
					Index j_first = diag_idx < m ? 0 : (diag_idx - m + 1);
					Index diag_len = Min(m - i_first, n - j_first);

					Index qstep_count = diag_len / SG_SIZE;

					for (Index qstep = 0; qstep < qstep_count; ++qstep)
					{
						Index step = qstep * SG_SIZE + sg_id;
						Index i = i_first + step;
						Index j = j_first + step;

						update_cell(i, j);
					}

					Index last_step = qstep_count * SG_SIZE + sg_id;
					if (last_step < diag_len)
					{
						Index i = i_first + last_step;
						Index j = j_first + last_step;

						update_cell(i, j);
					}

					sg.barrier();
				}
			});
		});
	}
}



void Lcs_Semi_Antidiagonal_old(LcsInput &input, LcsContext &ctx)
{

	InitInputs(input, ctx);
	InitStrands(ctx);

	Assert(ctx.queue);

	auto m = ctx.m;
	auto n = ctx.n;

	auto *_a_rev = ctx.a;
	auto *_b = ctx.b;

	auto *_h_strands = ctx.h_strands;
	auto *_v_strands = ctx.v_strands;
	using Index = uint32_t;
	{
		sycl::buffer<uint32_t, 1> buf_a_rev(_a_rev, m);
		sycl::buffer<uint32_t, 1> buf_b(_b, n);
		sycl::buffer<uint32_t, 1> buf_h_strands(_h_strands, m);
		sycl::buffer<uint32_t, 1> buf_v_strands(_v_strands, n);

		constexpr size_t SG_SIZE = 16;

		const size_t diag_count = m + n - 1;

		ctx.queue->submit([&](auto &h)
		{
			auto a_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			h.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{

				auto sg = item.get_sub_group();
				Index sg_id = sg.get_local_id()[0];
				Index sg_range = sg.get_local_range()[0];

				for (Index diag_idx = 0; diag_idx < diag_count; ++diag_idx)
				{
					Index i_first = diag_idx < m ? diag_idx : m - 1;
					Index j_first = diag_idx < m ? 0 : diag_idx - m + 1;
					Index diag_len = Min(i_first + 1, n - j_first);
					Index i_last = m - 1 - i_first;

					Index step_count = diag_len / SG_SIZE;

					for (Index qstep = 0; qstep < step_count; ++qstep)
					{
						Index step = qstep * SG_SIZE + sg_id;
						Index i = i_last + step;
						Index j = j_first + step;
						int a_sym = a_rev[i];
						int b_sym = b[j];
						int h_strand = h_strands[i];
						int v_strand = v_strands[j];
						int sym_equal = a_sym == b_sym;
						int has_crossing = h_strand > v_strand;
						int need_swap = sym_equal || has_crossing;
						h_strands[i] = need_swap ? v_strand : h_strand;
						v_strands[j] = need_swap ? h_strand : v_strand;
					}

					// remainder
					Index step = step_count * SG_SIZE + sg_id;
					if (step < diag_len)
					{

						Index i = i_last + step;
						Index j = j_first + step;
						int a_sym = a_rev[i];
						int b_sym = b[j];
						int h_strand = h_strands[i];
						int v_strand = v_strands[j];
						int sym_equal = a_sym == b_sym;
						int has_crossing = h_strand > v_strand;
						int need_swap = sym_equal || has_crossing;
						h_strands[i] = need_swap ? v_strand : h_strand;
						v_strands[j] = need_swap ? h_strand : v_strand;
					}

					sg.barrier();
				}
			});

		});
	}
}