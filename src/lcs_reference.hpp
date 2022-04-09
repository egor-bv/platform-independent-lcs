#pragma once

#include "lcs_common.hpp"

void Lcs_Semi_Reference(LcsInput &input, LcsContext &ctx)
{
	InitInputs(input, ctx);
	InitStrands(ctx);

	// antidiagonal iteration
	auto m = ctx.m;
	auto n = ctx.n;

	auto *h_strands = ctx.h_strands;
	auto *v_strands = ctx.v_strands;

	auto *a = ctx.a;
	auto *b = ctx.b;

	auto diag_count = m + n - 1;

	for (auto diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		auto i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
		auto j_first = diag_idx < m ? 0 : (diag_idx - m + 1);

		auto diag_len = Min(m - i_first, n - j_first);

		for (auto step = 0; step < diag_len; ++step)
		{
			auto i = i_first + step;
			auto j = j_first + step;

			auto h = h_strands[i];
			auto v = v_strands[j];

			bool has_match = a[i] == b[j];
			bool has_crossing = h > v;

			bool need_swap = has_match || has_crossing;

			h_strands[i] = need_swap ? v : h;
			v_strands[j] = need_swap ? h : v;
		}
	}
}

void Lcs_Prefix_Reference(LcsInput &input, LcsContext &ctx)
{
	InitInputsPrefix(input, ctx);
	InitDiagsPrefix(ctx);

	auto m = ctx.m;
	auto n = ctx.n;

	auto *diag0 = ctx.diag0;
	auto *diag1 = ctx.diag1;
	auto *diag2 = ctx.diag2;

	auto *a = ctx.a;
	auto *b = ctx.b;

	auto diag_count = m + n - 1;

	using Index = LcsContext::Index;
	using Symbol = LcsContext::Symbol;

	for (Index diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		Index i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
		Index j_first = diag_idx < m ? 0 : (diag_idx - m + 1);

		Index diag_len = Min(m - i_first, n - j_first);

		for (Index step = 0; step < diag_len; ++step)
		{
			Index i = i_first + step;
			Index j = j_first + step;

			Index di = i;
			Index d_w = diag1[di];
			Index d_n = diag1[di + 1];
			Index d_nw = diag0[di + 1] + Index(a[i] == b[j]);

			Index d = Max(Max(d_w, d_n), d_nw);
			diag2[di] = d;
		}

		// cycle diagonals
		auto diag_old0 = diag0;
		diag0 = diag1;
		diag1 = diag2;
		diag2 = diag_old0;
	}

	// final value is in diag1[0]

	Index llcs = diag1[0];
	ctx.llcs = llcs;

}


void Lcs_Prefix_Binary_Reference(LcsInput &input, LcsContext &ctx)
{
	InitInputsPrefixBinary(input, ctx);
	InitStrandsPrefixBinary(ctx);

	auto m = ctx.m;
	auto n = ctx.n;

	auto *h_strands = ctx.h_strands;
	auto *v_strands = ctx.v_strands;

	auto *a = ctx.a;
	auto *b = ctx.b;

	auto diag_count = m + n - 1;

	using Index = uint32_t;
	using Symbol = uint32_t;

	for (uint32_t diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		uint32_t i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
		uint32_t j_first = diag_idx < m ? 0 : (diag_idx - m + 1);

		uint32_t diag_len = Min(m - i_first, n - j_first);

		for (uint32_t step = 0; step < diag_len; ++step)
		{
			uint32_t i = i_first + step;
			uint32_t j = j_first + step;

			uint32_t l_strand = h_strands[i];
			uint32_t t_strand = v_strands[j];
			uint32_t symbol_a = a[i];
			uint32_t symbol_b = b[j];

			{
				uint32_t mask = 1;
				#pragma unroll
				for (uint32_t shift = 31; shift > 0; shift--)
				{
					uint32_t l_strand_cap = l_strand >> shift;
					uint32_t t_strand_cap = t_strand << shift;
					uint32_t cond = ~((symbol_a >> shift) ^ symbol_b);

					t_strand = (l_strand_cap | (~mask)) & (t_strand | (cond & mask));
					l_strand = t_strand_cap ^ (t_strand << shift) ^ l_strand;

					mask = (mask << 1) | 1u;
				}

				{
					uint32_t cond = ~(symbol_a ^ symbol_b);
					uint32_t l_strand_cap = l_strand;
					uint32_t t_strand_cap = t_strand;

					t_strand = (l_strand_cap | (~mask)) & (t_strand | (cond & mask));
					l_strand = t_strand_cap ^ (t_strand) ^ l_strand;
				}


				mask = ~0;
				#pragma unroll
				for (uint32_t shift = 1; shift < 32; shift++)
				{
					mask <<= 1;

					uint32_t l_strand_cap = l_strand << shift;
					uint32_t t_strand_cap = t_strand >> shift;
					uint32_t cond = ~(((symbol_a << (shift)) ^ symbol_b));
					t_strand = (l_strand_cap | (~mask)) & (t_strand | (cond & mask));
					l_strand = t_strand_cap ^ (t_strand >> shift) ^ l_strand;
				}

			}

			h_strands[i] = l_strand;
			v_strands[j] = t_strand;
		}
	}

	uint32_t bit_count = 0;
	for (int i = 0; i < m; ++i)
	{
		uint32_t h_strand = h_strands[i];
		bit_count += sycl::popcount(h_strand);
	}

	ctx.llcs = 32*m - bit_count;
}