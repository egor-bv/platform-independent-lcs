#pragma once

#include "utility.hpp"
#include <CL/sycl.hpp>


/// <summary>
/// Comb sticky braids in antidiagonal pattern
/// * Template parameter SG_POW2 must be 2 or 3
/// </summary>
/// <param name="q">command queue</param>
/// <param name="_a_rev">first sequence, reversed</param>
/// <param name="_b">second sequence, in original order</param>
/// <param name="m">length of first sequence</param>
/// <param name="n">length of second sequence</param>
/// <param name="_h_strands">horizontal strands, initialized in reverse order</param>
/// <param name="_v_strands">vertical strands, initialized in original order</param>
template <int SG_POW2>
void StickyBraidComb_Antidiagonal(sycl::queue q, const int *_a_rev, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a_rev(_a_rev, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	constexpr size_t SG_SIZE = 1 << SG_POW2;

	const size_t diag_count = m + n - 1;

	q.submit([&](auto &h)
		{
			auto a_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			h.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					using Index = int;
					auto sg = item.get_sub_group();
					Index sg_id = sg.get_local_id()[0];
					Index sg_range = sg.get_local_range()[0];

					for (Index diag_idx = 0; diag_idx < diag_count; ++diag_idx)
					{
						Index i_first = diag_idx < m ? diag_idx : m - 1;
						Index j_first = diag_idx < m ? 0 : diag_idx - m + 1;
						Index diag_len = Min(i_first + 1, n - j_first);
						Index i_last = m - 1 - i_first;

						Index step_count = diag_len >> SG_POW2;

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
				}
			);

		}
	);
}