#pragma once

#include "lcs_types.hpp"

void Lcs_Semi_Reference(const LcsInput &input, LcsContext &ctx)
{
	int m = input.a_size;
	int n = input.b_size;

	// Copy symbols
	int *a = new int[m];
	int *b = new int[n];
	for (int i = 0; i < m; ++i)
	{
		a[i] = input.a_data[m - i - 1];
	}
	for (int j = 0; j < n; ++j)
	{
		b[j] = input.b_data[j];
	}

	// Init strands
	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i)
	{
		h_strands[i] = i;
	}
	for (int j = 0; j < n; ++j)
	{
		v_strands[j] = m + j;
	}

	int diag_count = m + n - 1;
	for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		int i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
		int j_first = diag_idx < m ? 0 : (diag_idx - m + 1);

		int diag_len = Min(m - i_first, n - j_first);

		for (int step = 0; step < diag_len; ++step)
		{
			int i = i_first + step;
			int j = j_first + step;

			int h = h_strands[i];
			int v = v_strands[j];

			bool has_match = a[i] == b[j];
			bool has_crossing = h > v;

			bool need_swap = has_match || has_crossing;

			h_strands[i] = need_swap ? v : h;
			v_strands[j] = need_swap ? h : v;
		}
	}

	// Store strands in the context
	ctx.h_strands = h_strands;
	ctx.h_strands_size = m;
	ctx.v_strands = v_strands;
	ctx.v_strands_size = n;

	// Cleanup
	delete[] a;
	delete[] b;
}

void Lcs_Prefix_Reference(const LcsInput &input, LcsContext &ctx)
{
	int m = input.a_size;
	int n = input.b_size;

	// Copy symbols
	int *a = new int[m];
	int *b = new int[n];
	for (int i = 0; i < m; ++i)
	{
		a[i] = input.a_data[m - i - 1];
	}
	for (int j = 0; j < m; ++j)
	{
		b[j] = input.b_data[j];
	}

	int diag_count = m + n - 1;
	int diag_max_size = Min(n, m) + 1;

	// NOTE: initialized to zero
	int *diag0 = new int[diag_max_size] {};
	int *diag1 = new int[diag_max_size] {};
	int *diag2 = new int[diag_max_size] {};

	for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		int i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
		int j_first = diag_idx < m ? 0 : (diag_idx - m + 1);

		int diag_len = Min(m - i_first, n - j_first);

		for (int step = 0; step < diag_len; ++step)
		{
			int i = i_first + step;
			int j = j_first + step;

			int di = i;
			int d_w = diag1[di];
			int d_n = diag1[di + 1];
			int d_nw = diag0[di + 1] + int(a[i] == b[j]);

			int d = Max(Max(d_w, d_n), d_nw);
			diag2[di] = d;
		}

		// cycle diagonals
		int *diag_old0 = diag0;
		diag0 = diag1;
		diag1 = diag2;
		diag2 = diag_old0;
	}

	// final value is in diag1[0]
	ctx.llcs = diag1[0];

	delete[] a;
	delete[] b;
	delete[] diag0;
	delete[] diag1;
	delete[] diag2;
}


#if 0
void Lcs_Prefix_Binary_Reference(LcsInput &input, LcsContext &ctx)
{
	// Init symbols by converting to binary...

	for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		int i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
		int j_first = diag_idx < m ? 0 : (diag_idx - m + 1);

		int diag_len = Min(m - i_first, n - j_first);

		for (int step = 0; step < diag_len; ++step)
		{
			int i = i_first + step;
			int j = j_first + step;

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
#endif
