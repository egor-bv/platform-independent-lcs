#pragma once

#include "lcs_common.hpp"

void Lcs_Semi_Subproblems(LcsInput &input, LcsContext &ctx)
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

	constexpr int SG_SIZE = 16;
	int diag_count = m + n - 1;
	int stripe_count = SmallestMultipleToFit(m, SG_SIZE);

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
				int sg_id = sg.get_local_linear_id();

				// first stripe -- incomplete, need masking
				{
					for (int col = -...; col < ...; ++col)
					{
					}
				}

				// all remaining stripes...
				for (int stripe = 0; stripe < stripe_count; ++stripe)
				{
					// left part

					// middle part

					// right part
				}

			});
		});
	}
}