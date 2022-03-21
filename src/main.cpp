#include <vector>
#include <random>

#include <stdio.h>
#include <inttypes.h>

#include "stopwatch.hpp"
#include "permutation.hpp"
#include "utility.hpp"




std::vector<int> RandomBinarySequence(int size, uint32_t seed)
{
	std::vector<int> result(size);
	std::mt19937 rng(seed);
	for (int i = 0; i < size; ++i)
	{
		result[i] = rng() & 1;
	}
	return result;
}


struct LcsProblem
{
	const int *input_a;
	const int *input_b;
	int size_a;
	int size_b;

	int llcs;
	int64_t perm_hash;
};

void semilocal_reference(LcsProblem &p)
{
	int m = p.size_a;
	int n = p.size_b;

	int *a = new int[m];
	int *b = new int[n];
	int *h_strands = new int[m];
	int *v_strands = new int[n];

	for (int i = 0; i < m; ++i) a[i] = p.input_a[m - 1 - i];
	for (int j = 0; j < n; ++j) b[j] = p.input_b[j];
	
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = m + j;

	// actual work
	int diag_count = m + n - 1;
	for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		int i_first = diag_idx < m ? m - 1 - diag_idx : 0;
		int j_first = diag_idx < m ? 0 : diag_idx - m + 1;
		int diag_len = Min(m - i_first, n - j_first);

		for (int step = 0; step < diag_len; ++step)
		{
			int i = i_first + step;
			int j = j_first + step;

			int h = h_strands[i];
			int v = v_strands[j];

			bool sym_equal = a[i] == b[j];
			bool has_crossing = h > v;
			bool need_swap = sym_equal || has_crossing;
			
			h_strands[i] = need_swap ? v : h;
			v_strands[j] = need_swap ? h : v;
		}
	}


	auto result = PermutationMatrix::FromStrands(h_strands, m, v_strands, n);

	p.perm_hash = result.hash();

	delete[] a;
	delete[] b;
	delete[] h_strands;
	delete[] v_strands;

}


struct TestSpec
{
	std::string a_path;
	std::string b_path;

	int a_size = 102400;
	int b_size = 102400;

	int num_iter = 4;

	std::string algo_name;
	
	bool verbose = true;
	bool validate = true;

	uint32_t random_seed = 1;

	void Execute()
	{
		std::vector<int> a = RandomBinarySequence(a_size, random_seed);
		std::vector<int> b = RandomBinarySequence(b_size, ~random_seed);
		int64_t total_size = int64_t(a_size) * int64_t(b_size);
		if (verbose)
		{
			printf("<<< beginning test >>>\n");
			printf("(%d * %d = %lld)\n", a_size, b_size, total_size);
		}

		// reference: run once...

		for (int iter = 0; iter < num_iter; ++iter)
		{
			LcsProblem p;
			p.input_a = &a[0];
			p.input_b = &b[0];
			p.size_a = a.size();
			p.size_b = b.size();

			Stopwatch sw;
			// run algo f(p)
			semilocal_reference(p);
			sw.stop();

			double cells_per_us = total_size / sw.elapsed_ms() / 1000.0; 
			printf("%f c/us;   %lld\n", cells_per_us, p.perm_hash);

			// validate
			// print results

		}
	}
};




int main(int argc, char **argv)
{
	TestSpec spec;
	spec.Execute();
	return 0;
}