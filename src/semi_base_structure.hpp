#pragma once
#include "semi_lcs_types.hpp"
#include "permutation.hpp"
#include <CL/sycl.hpp>


/// <summary>
/// Common skeleton for semi-local LCS, using only standard C++
/// </summary>
/// <typeparam name="CombingProc">combing procedure type, see usage</typeparam>
/// <param name="comb">combing procedure</param>
/// <param name="given">input sequence pair</param>
/// <param name="reverse_a">whether to reverse first sequence before passing it to combing procedure</param>
/// <returns>permutation matrix representing semi-local LCS result</returns>
template<class CombingProc>
PermutationMatrix SemiLcsBase_NoSycl(CombingProc comb, const InputSequencePair &given, bool reverse_a = false)
{
	const int m = given.length_a;
	const int n = given.length_b;
	const int *a = given.a;
	if (reverse_a)
	{
		int *a_rev = new int[given.length_a];
		for (int i = 0; i < given.length_a; ++i)
		{
			a_rev[i] = given.a[given.length_a - 1 - i];
		}
		a = a_rev;
	}

	const int *b = given.b;

	// initialize strands
	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	// comb braid using given procedure
	comb(a, b, m, n, h_strands, v_strands);


	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands, m, v_strands, n);


	delete[] h_strands;
	delete[] v_strands;
	if (reverse_a) delete[] a;

	return result;
}


/// <summary>
/// Common skeleton for semi-local LCS, using DPC++
/// </summary>
/// <typeparam name="CombingProc">combing procedure type, see usage</typeparam>
/// <param name="comb">combing procedure</param>
/// <param name="q">command queue</param>
/// <param name="given">input sequence pair</param>
/// <param name="reverse_a">whether to reverse first sequence before passing it to combing procedure</param>
/// <returns>permutation matrix representing semi-local LCS result</returns>
template<class CombingProc>
PermutationMatrix SemiLcsBase_Sycl(CombingProc comb, sycl::queue &q, const InputSequencePair &given, bool reverse_a = false)
{
	const int m = given.length_a;
	const int n = given.length_b;

	const int *a = given.a;
	if (reverse_a)
	{
		int *a_rev = new int[given.length_a];
		for (int i = 0; i < given.length_a; ++i)
		{
			a_rev[i] = given.a[given.length_a - 1 - i];
		}
		a = a_rev;
	}

	const int *b = given.b;

	// initialize strands
	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	comb(q, a, b, m, n, h_strands, v_strands);

	// write resulting permutation matrix
	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands, m, v_strands, n);

	delete[] h_strands;
	delete[] v_strands;
	if (reverse_a) delete[] a;

	return result;
}

