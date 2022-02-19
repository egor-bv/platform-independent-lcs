#pragma once

#include "utility.hpp"

/// <summary>
/// Comb sticky braids in antidiagonal pattern using only standard C++
/// </summary>
/// <param name="q">command queue</param>
/// <param name="_a_rev">first sequence, reversed</param>
/// <param name="_b">second sequence, in original order</param>
/// <param name="m">length of first sequence</param>
/// <param name="n">length of second sequence</param>
/// <param name="_h_strands">horizontal strands, initialized in reverse order</param>
/// <param name="_v_strands">vertical strands, initialized in original order</param>
void StickyBraidComb_Reference(const int *a_rev, const int *b, int m, int n, int *h_strands, int *v_strands)
{
	int diag_count = m + n - 1;

	for (int i_diag = 0; i_diag < diag_count; ++i_diag)
	{
		int i_first = i_diag < m ? i_diag : m - 1;
		int j_first = i_diag < m ? 0 : i_diag - m + 1;

		// along antidiagonal, i goes down, j goes up
		int diag_len = Min(i_first + 1, n - j_first);
		int i_last = m - 1 - i_first;

		for (int steps = 0; steps < diag_len; ++steps)
		{
			// actual grid coordinates
			int i = i_last + steps;
			int j = j_first + steps;

			{
				int h_index = i;
				int v_index = j;
				int h_strand = h_strands[h_index];
				int v_strand = v_strands[v_index];

				bool need_swap = a_rev[i] == b[j] || h_strand > v_strand;

				h_strands[h_index] = need_swap ? v_strand : h_strand;
				v_strands[v_index] = need_swap ? h_strand : v_strand;

			}

		}
	}
}
