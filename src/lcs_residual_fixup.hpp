#pragma once
#include "lcs_interface_interlal.hpp"

void Lcs_Semi_Reference2(LcsProblemContext &ctx)
{
	int m = ctx.a_length;
	int n = ctx.b_length;

	// Copy input sequences
	Word *a = new Word[m];
	Word *b = new Word[n];
	for (int i = 0; i < m; ++i)
	{
		a[i] = ctx.a_given[m - i - 1];
	}
	for (int j = 0; j < n; ++j)
	{
		b[j] = ctx.b_given[j];
	}

	// Init strands
	Word *h_strands = new Word[m];
	Word *v_strands = new Word[n];
	for (int i = 0; i < m; ++i)
	{
		h_strands[i] = i;
	}
	for (int j = 0; j < n; ++j)
	{
		v_strands[j] = m + j;
	}


	// Loop over antidiagonals, updating strand indices
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

			Word h = h_strands[i];
			Word v = v_strands[j];

			bool has_match = a[i] == b[j];
			bool has_crossing = h > v;

			bool need_swap = has_match || has_crossing;
			h_strands[i] = need_swap ? v : h;
			v_strands[j] = need_swap ? h : v;

		}
	}

	// Store strands into context
	ctx.h_strands = h_strands;
	ctx.h_strands_length = m;
	ctx.v_strands = v_strands;
	ctx.v_strands_length = n;

	// Cleanup
	delete[] a;
	delete[] b;

}

// Assuming symbols and strands inside ctx are initialized and valid at whatever state,
// process subrectangle defined by bottom-left corner {i0, j0} and size {isize, bsize}
void Lcs_Semi_Fixup(LcsProblemContext &ctx, 
					int i0, int isize,
					int j0, int jsize)
{
	// printf("i0 = %d, isize = %d, j0 = %d, jsize = %d\n", i0, isize, j0, jsize);
	if (isize == 0 || jsize == 0)
	{
		return;
	}



	int m = isize;
	int n = jsize;

	Assert(i0 + isize <= ctx.h_strands_length);
	Assert(j0 + jsize <= ctx.v_strands_length);

	Word *h_strands = ctx.h_strands + i0;
	Word *v_strands = ctx.v_strands + j0;
	
	const Word *a = ctx.a_prepared + i0;
	const Word *b = ctx.b_prepared + j0;

	int diag_count = m + n - 1;
	for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		// All relative to subrectangle
		int i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
		int j_first = diag_idx < m ? 0 : (diag_idx - m + 1);
		int diag_len = Min(m - i_first, n - j_first);

		for (int step = 0; step < diag_len; ++step)
		{
			int i = i_first + step;
			int j = j_first + step;

			Word h = h_strands[i];
			Word v = v_strands[j];

			bool has_match = a[i] == b[j];
			bool has_crossing = h > v;

			bool need_swap = has_match || has_crossing;
			h_strands[i] = need_swap ? v : h;
			v_strands[j] = need_swap ? h : v;
		}
	}
}