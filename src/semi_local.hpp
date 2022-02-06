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

				h_strands[h_index] = need_swap ? v_strand : h_strand;
				v_strands[v_index] = need_swap ? h_strand : v_strand;

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

								h_strands[h_index] = need_swap ? v_strand : h_strand;
								v_strands[v_index] = need_swap ? h_strand : v_strand;

								//int r = need_swap;
								//h_strands[h_index] = (v_strand & (r - 1)) | ((-r) & h_strand);
								//v_strands[v_index] = (h_strand & (r - 1)) | ((-r) & v_strand);
							}
						}
					}
				}
			);
		}
	);
}

void SingleTaskCombRowMajor(sycl::queue q, const int *_a, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
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
					for (int i = 0; i < m; ++i)
						for (int j = 0; j < n; ++j)
						{

							int h_index = m - 1 - i;
							int v_index = j;
							int h_strand = h_strands[h_index];
							int v_strand = v_strands[v_index];

							bool need_swap = a[i] == b[j] || h_strand > v_strand;

							h_strands[h_index] = need_swap ? v_strand : h_strand;
							v_strands[v_index] = need_swap ? h_strand : v_strand;

						}
				}
			);
		}
	);
}

void SingleWorkgroupComb_(sycl::queue q, const int *_a, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
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

			const int wg_size = 8;

			h.parallel_for(sycl::nd_range<1>{ wg_size, wg_size },
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(wg_size)]]
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


								h_strands[h_index] = need_swap ? v_strand : h_strand;
								v_strands[v_index] = need_swap ? h_strand : v_strand;

							}

						}
						sg.barrier();
					}
				}
				);
		}
	);
}


// trying to improve performance
// NOTE: a is assumed reversed here!
void SingleWorkgroupComb(sycl::queue q, const int *_a_rev, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a_rev(_a_rev, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	constexpr size_t SG_POW2 = 3;
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
					using IndexType = int;
					auto sg = item.get_sub_group();
					IndexType sg_id = sg.get_local_id()[0];
					IndexType sg_range = sg.get_local_range()[0];

					for (IndexType diag_idx = 0; diag_idx < diag_count; ++diag_idx)
					{
						IndexType i_first = diag_idx < m ? diag_idx : m - 1;
						IndexType j_first = diag_idx < m ? 0 : diag_idx - m + 1;
						IndexType diag_len = Min(i_first + 1, n - j_first);
						IndexType i_last = m - 1 - i_first;

						for (IndexType step = sg_id; step < diag_len; step += sg_range)
						{
							IndexType i = i_last + step;
							IndexType j = j_first + step;
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


void NaiveSyclComb(sycl::queue q, const int *_a, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a(_a, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	int diag_count = m + n - 1;

	for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		q.submit([&](auto &h)
			{
				auto a = buf_a.get_access<sycl::access::mode::read>(h);
				auto b = buf_b.get_access<sycl::access::mode::read>(h);
				auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
				auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

				int i_first = diag_idx < m ? diag_idx : m - 1;
				int j_first = diag_idx < m ? 0 : diag_idx - m + 1;
				int diag_len = Min(i_first + 1, n - j_first);

				h.parallel_for(sycl::nd_range<1>{ diag_len, 1024},
					[=](sycl::nd_item<1> item)
					{
						int steps = item.get_global_linear_id();
						int i = i_first - steps;
						int j = j_first + steps;

						int h_index = m - 1 - i;
						int v_index = j;
						int h_strand = h_strands[h_index];
						int v_strand = v_strands[v_index];

						bool need_swap = a[i] == b[j] || h_strand > v_strand;
						h_strands[h_index] = need_swap ? v_strand : h_strand;
						v_strands[v_index] = need_swap ? h_strand : v_strand;

					}
				);
			}
		);
	}
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

PermutationMatrix semi_parallel_single_task_row_major(sycl::queue q, const InputSequencePair &given)
{
	return semi_local_lcs_sycl(SingleTaskCombRowMajor, q, given);
}

PermutationMatrix semi_parallel_single_sub_group(sycl::queue q, const InputSequencePair &given)
{
	return semi_local_lcs_sycl(SingleWorkgroupComb, q, given);
}

PermutationMatrix semi_parallel_naive_sycl(sycl::queue q, const InputSequencePair &given)
{
	return semi_local_lcs_sycl(NaiveSyclComb, q, given);
}



// Iterate in rectangular blocks to reduce number of dispatches
// It's also probably not correct w.r.t. memory writes/reads
long long StickyBraidParallelBlockwise(sycl::queue &q, InputSequencePair p)
{
	int m = p.length_a;
	int n = p.length_b;
	const int *a = p.a;
	const int *b = p.b;

	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	size_t total_num_threads = 0;
	{
		sycl::buffer<int, 1> buf_a(p.a, p.length_a);
		sycl::buffer<int, 1> buf_b(p.b, p.length_b);

		sycl::buffer<int, 1> buf_h_strands(h_strands, m);
		sycl::buffer<int, 1> buf_v_strands(v_strands, n);

		int original_m = m;
		int original_n = n;


		const int block_m = 256;
		const int block_n = 256;
		m = (m + block_m - 1) / block_m;
		n = (n + block_n - 1) / block_n;


		int diag_count = m + n - 1;
		// int max_diag_length = m; // say it's vertical!

		int block_diag_count = block_m + block_n - 1;
		// int block_diag_length = block_m;

		for (int i_diag = 0; i_diag < diag_count; ++i_diag)
		{
			int i_first = i_diag < m ? i_diag : m - 1;
			int j_first = i_diag < m ? 0 : i_diag - m + 1;
			// along antidiagonal, i goes down, j goes up
			int diag_len = std::min(i_first + 1, n - j_first);

			total_num_threads += diag_len;
			// each block antidiagonal is a separate kernel
			q.submit(
				[&](auto &h)
				{
					// accessors are defined here
					auto acc_h_strands = buf_h_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto acc_v_strands = buf_v_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto acc_a = buf_a.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);
					auto acc_b = buf_b.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);

#if 1 // DEBUG: no inner loop
					h.parallel_for(sycl::range<1>(diag_len),
						[=](auto iter_steps)
						// [[intel::kernel_args_restrict]]
						{
#if 1 // DEBUG: no work inside inner loop
							int steps = iter_steps;
							// block coordinates
							int i_block = i_first - steps;
							int j_block = j_first + steps;
#if 1 // DEBUG: row-major iteration
							// row-major iteration
							for (int i_rel = 0; i_rel < block_m; ++i_rel)
							{
								for (int j_rel = 0; j_rel < block_n; j_rel++)
								{
									int i = i_block * block_m + i_rel;
									int j = j_block * block_n + j_rel;
									if (i >= original_m || j >= original_n)
									{
										continue;
									}

									int h_index = original_m - 1 - i;
									int v_index = j;
									int h_strand = acc_h_strands[h_index];
									int v_strand = acc_v_strands[v_index];
									bool need_swap = acc_a[i] == acc_b[j] || h_strand > v_strand;
#if 1 // DEBUG: remove if
									{
										// maybe a little faster?
										int r = (int)need_swap;
										acc_h_strands[h_index] = (h_strand & (r - 1)) | ((-r) & v_strand);
										acc_v_strands[v_index] = (v_strand & (r - 1)) | ((-r) & h_strand);
									}
#else
									if (need_swap)
									{
										// swap
										acc_h_strands[h_index] = v_strand;
										acc_v_strands[v_index] = h_strand;
									}
#endif
								}
							}
#else // DEBUG: anti-diagonal iteration
							for (int i_block_diag = 0; i_block_diag < block_diag_count; ++i_block_diag)
							{
								int i_rel_first = i_block_diag < block_m ? i_block_diag : block_m - 1;
								int j_rel_first = i_block_diag < block_m ? 0 : i_block_diag - block_m + 1;
								int block_diag_len = Min(i_rel_first + 1, block_n - j_rel_first);
								for (int block_steps = 0; block_steps < block_diag_len; ++block_steps)
								{
									int i_rel = i_rel_first - block_steps;
									int j_rel = j_rel_first + block_steps;
									int i_row0 = i_block * block_m + i_rel;
									int j = j_block * block_n + j_rel;
									bool inside = i_row0 < original_m &&j < original_n;


									int h_index = original_m - 1 - i_row0;
									int v_index = j;

									int h_strand = inside ? acc_h_strands[h_index] : 0;
									int v_strand = inside ? acc_v_strands[v_index] : 0;
									int a_sym = inside ? acc_a[i_row0] : 0;
									int b_sym = inside ? acc_b[j] : 0;
									bool need_swap = (a_sym == b_sym || h_strand > v_strand) && inside;

									{
										if (need_swap) acc_h_strands[h_index] = v_strand;
										if (need_swap) acc_v_strands[v_index] = h_strand;
									}
								}
							}
#endif
#endif
						}
					);
#endif

				}
			).wait();

		}

		m = original_m;
		n = original_n;
	}

	//std::cout << "blockwise, total_num_threads: " << total_num_threads << "\n";
	{
		PermutationMatrix p(m + n);
		for (int l = 0; l < m; l++) p.set_point(h_strands[l], n + l);
		for (int r = m; r < m + n; r++) p.set_point(v_strands[r - m], r - m);

		//if (p.size < 100)
		//{
		//	std::cout << "DEBUG INFO:" << std::endl;
		//	for (int row_i = 0; row_i < p.size; ++row_i)
		//	{
		//		std::cout << p.get_row_by_col(row_i) << " ";
		//	}
		//	std::cout << std::endl;

		//	for (int row_i = 0; row_i < p.size; ++row_i)
		//	{
		//		std::cout << p.get_col_by_row(row_i) << " ";
		//	}
		//	std::cout << std::endl;
		//}



		long long result = hash(p, p.size);
		delete[] h_strands;
		delete[] v_strands;
		return result;
	}
}
