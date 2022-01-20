#pragma once
#include "permutation.hpp"
#include "utility.hpp"


#include <CL/sycl.hpp>


// best so far
void AntidiagonalCombBottomUp(const int *a, const int *b, int m, int n, int *h_strands, int *v_strands)
{
	int diag_count = m + n - 1;

	for (int i_diag = 0; i_diag < diag_count; ++i_diag)
	{
		int i_first = i_diag < m ? i_diag : m - 1;
		int j_first = i_diag < m ? 0 : i_diag - m + 1;
		// along antidiagonal, i goes down, j goes up
		int diag_len = Min(i_first + 1, n - j_first);

		for (int steps = 0; steps < diag_len; ++steps)
		{
			// actual grid coordinates
			int i = i_first - steps;
			int j = j_first + steps;

			{
				int h_index = m - 1 - i;
				int v_index = j;
				int h_strand = h_strands[h_index];
				int v_strand = v_strands[v_index];

				bool need_swap = a[i] == b[j] || h_strand > v_strand;
#if 1
				h_strands[h_index] = need_swap ? v_strand : h_strand;
				v_strands[v_index] = need_swap ? h_strand : v_strand;
#else
				if (need_swap)
				{
					h_strands[h_index] = v_strand;
					v_strands[v_index] = h_strand;
				}
#endif
			}

		}
	}
}


void SingleTaskComb(sycl::queue q, const int *_a, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a(_a, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	int diag_count = m + n - 1;

	q.submit([&](auto &h)
		{
			auto a = buf_a.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			h.single_task([=]()
				{
					for (int i_diag = 0; i_diag < diag_count; ++i_diag)
					{
						int i_first = i_diag < m ? i_diag : m - 1;
						int j_first = i_diag < m ? 0 : i_diag - m + 1;
						int diag_len = Min(i_first + 1, n - j_first);

						for (int steps = 0; steps < diag_len; ++steps)
						{
							// actual grid coordinates
							int i = i_first - steps;
							int j = j_first + steps;

							{
								int h_index = m - 1 - i;
								int v_index = j;
								int h_strand = h_strands[h_index];
								int v_strand = v_strands[v_index];

								bool need_swap = a[i] == b[j] || h_strand > v_strand;

								{
									h_strands[h_index] = need_swap ? v_strand : h_strand;
									v_strands[v_index] = need_swap ? h_strand : v_strand;
								}

							}
						}
					}
				}
			);
		}
	);
}

void SingleWorkgroupComb(sycl::queue q, const int *_a, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a(_a, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	int diag_count = m + n - 1;

	q.submit([&](auto &h)
		{
			auto a = buf_a.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			int wg_size = 8;

			h.parallel_for(sycl::nd_range<1>{ wg_size, wg_size },
				[=](sycl::nd_item<1> item)
				[[cl::intel_reqd_sub_group_size(wg_size)]]
				{
					auto sg = item.get_sub_group();
					int sglid = sg.get_local_id()[0];
					int sgrange = sg.get_max_local_range()[0];
					// common code for all
					for (int i_diag = 0; i_diag < diag_count; ++i_diag)
					{
						int i_first = i_diag < m ? i_diag : m - 1;
						int j_first = i_diag < m ? 0 : i_diag - m + 1;
						int diag_len = Min(i_first + 1, n - j_first);

						// split between workgroup elements
						for (int steps = sglid; steps < diag_len; steps += sgrange)
						{
							// actual grid coordinates
							int i = i_first - steps;
							int j = j_first + steps;

							{
								int h_index = m - 1 - i;
								int v_index = j;
								int h_strand = h_strands[h_index];
								int v_strand = v_strands[v_index];

								bool need_swap = a[i] == b[j] || h_strand > v_strand;
#if 1
								{
									h_strands[h_index] = need_swap ? v_strand : h_strand;
									v_strands[v_index] = need_swap ? h_strand : v_strand;
								}
#else

								{
									if (need_swap) h_strands[h_index] = v_strand;
									if (need_swap) v_strands[v_index] = h_strand;
								}

#endif
							}
						}
						// sg.barrier();
					}
				}
				);
		}
	);
}


template<class CombingProc>
PermutationMatrix semi_local_lcs_cpu(CombingProc comb, const InputSequencePair &given)
{
	const int m = given.length_a;
	const int n = given.length_b;
	const int *a = given.a;
	const int *b = given.b;

	// initialize strands
	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	// comb braid using given procedure
	comb(a, b, m, n, h_strands, v_strands);


	PermutationMatrix result(m + n);
	for (int l = 0; l < m; l++) result.set_point(h_strands[l], n + l);
	for (int r = m; r < m + n; r++) result.set_point(v_strands[r - m], r - m);


	delete[] h_strands;
	delete[] v_strands;

	return result;
}

template<class CombingProc>
PermutationMatrix semi_local_lcs_sycl(CombingProc comb, sycl::queue &q, const InputSequencePair &given)
{
	const int m = given.length_a;
	const int n = given.length_b;
	const int *a = given.a;
	const int *b = given.b;

	// initialize strands
	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	comb(q, a, b, m, n, h_strands, v_strands);

	// write resulting permutation matrix
	PermutationMatrix result(m + n);
	for (int l = 0; l < m; l++) result.set_point(h_strands[l], n + l);
	for (int r = m; r < m + n; r++) result.set_point(v_strands[r - m], r - m);

	delete[] h_strands;
	delete[] v_strands;

	return result;
}


//
// Algorithm variations
//


PermutationMatrix semi_cpu_antidiag(const InputSequencePair &given)
{
	return semi_local_lcs_cpu(AntidiagonalCombBottomUp, given);
}

PermutationMatrix semi_parallel_single_task(sycl::queue q, const InputSequencePair &given)
{
	return semi_local_lcs_sycl(SingleTaskComb, q, given);
}

PermutationMatrix semi_parallel_single_sub_group(sycl::queue q, const InputSequencePair &given)
{
	return semi_local_lcs_sycl(SingleWorkgroupComb, q, given);
}


