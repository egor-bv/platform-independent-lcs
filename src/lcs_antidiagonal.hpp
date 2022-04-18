#pragma once

#include "lcs_common.hpp"

void Lcs_Semi_Antidiagonal(LcsInput &input, LcsContext &ctx)
{
	InitInputs(input, ctx);
	InitStrands(ctx);

	Assert(ctx.queue);


	using Symbol = int;// LcsContext::Symbol;
	using Index = int; // LcsContext::Index;

	auto m = int(ctx.m);
	auto n = int(ctx.n);

	auto *a_data = (int *)ctx.a;
	auto *b_data = (int *)ctx.b;

	auto *h_strands_data = (int *)ctx.h_strands;
	auto *v_strands_data = (int *)ctx.v_strands;


	
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

				auto update_cell = [&](int i, int j)
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


				for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
				{
					int i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
					int j_first = diag_idx < m ? 0 : (diag_idx - m + 1);
					int diag_len = Min(m - i_first, n - j_first);

					int qstep_count = diag_len / SG_SIZE;

					#pragma unroll 2
					for (int qstep = 0; qstep < qstep_count; ++qstep)
					{
						int step = qstep * SG_SIZE + sg_id;
						int i = i_first + step;
						int j = j_first + step;

						update_cell(i, j);
					}

					int last_step = qstep_count * SG_SIZE + sg_id;
					if (last_step < diag_len)
					{
						int i = i_first + last_step;
						int j = j_first + last_step;

						update_cell(i, j);
					}

					sg.barrier();
				}
			});
		});
	}
}


