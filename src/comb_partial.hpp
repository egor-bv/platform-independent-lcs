#pragma once

//
// Separate combing of individual antidiagonals for easier perf testing
//


template<class CombingProc>
double test_comb_partial_cpu(CombingProc comb_some, const InputSequencePair &given)
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

	Stopwatch sw;
	comb_some(a, b, m, n, h_strands, v_strands);
	sw.stop();

	delete[] h_strands;
	delete[] v_strands;

	return sw.elapsed_ms();
}


double test_comb_partial_sycl(sycl::queue &q, const InputSequencePair &given, int iter_count)
{
	const int m = given.length_a;
	const int n = given.length_b;
	const int *_a = given.a;
	const int *_b = given.b;

	// initialize strands
	int *_h_strands = new int[m];
	int *_v_strands = new int[n];
	for (int i = 0; i < m; ++i) _h_strands[i] = i;
	for (int j = 0; j < n; ++j) _v_strands[j] = j + m;

	Stopwatch sw;
	{
		sycl::buffer<int, 1> buf_a(_a, m);
		sycl::buffer<int, 1> buf_b(_b, n);
		sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
		sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

		int wg_size = 4096;
		// actual combing happens here
		for (int diag_idx = m; diag_idx < m + iter_count; ++diag_idx)
		{
			int i_diag = diag_idx;
			int i_first = i_diag < m ? i_diag : m - 1;
			int j_first = i_diag < m ? 0 : i_diag - m + 1;

			int diag_len = Min(i_first + 1, n - j_first);

			int i_last = m - 1 - i_first;
			q.submit(
				[&](auto &h)
				{
					auto a = buf_a.get_access<sycl::access::mode::read>(h);
					auto b = buf_b.get_access<sycl::access::mode::read>(h);
					auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
					auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

					const size_t global_size = diag_len;
					h.parallel_for(sycl::range<1>{ global_size },
						[=](sycl::item<1> item)
						{
							//int steps = item;
							//// int i = i_first - steps;
							//// int j = j_first + steps;
							//int h_index = i_last + steps;
							//int v_index = j_first + steps;
							//int i = i_first - steps;
							//int j = v_index;
							//{
							//	// int h_index = m - 1 - i;
							//	// int v_index = j;
							//	int h_strand = h_strands[h_index];
							//	int v_strand = v_strands[v_index];

							//	bool need_swap = a[i] == b[j] || h_strand > v_strand;
							//	{
							//		h_strands[h_index] = need_swap ? v_strand : h_strand;
							//		v_strands[v_index] = need_swap ? h_strand : v_strand;
							//	}
							//}
						}
					);
				}
			);
		}
	}
	sw.stop();

	delete[] _h_strands;
	delete[] _v_strands;

	return sw.elapsed_ms();
}


double test_comb_partial_sycl_iter(sycl::queue &q, const InputSequencePair &given, int iter_count)
{
	const int m = given.length_a;
	const int n = given.length_b;
	const int *_a = given.a;
	const int *_b = given.b;

	// initialize strands
	int *_h_strands = new int[m];
	int *_v_strands = new int[n];
	for (int i = 0; i < m; ++i) _h_strands[i] = i;
	for (int j = 0; j < n; ++j) _v_strands[j] = j + m;

	Stopwatch sw;
	{
		sycl::buffer<int, 1> buf_a(_a, m);
		sycl::buffer<int, 1> buf_b(_b, n);
		sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
		sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

		int internal_iter = 16;
		// actual combing happens here
		for (int diag_idx = m; diag_idx < m + iter_count; ++diag_idx)
		{
			int i_diag = diag_idx;
			int i_first = i_diag < m ? i_diag : m - 1;
			int j_first = i_diag < m ? 0 : i_diag - m + 1;

			int diag_len = Min(i_first + 1, n - j_first);

			int i_last = m - 1 - i_first;
			q.submit(
				[&](auto &h)
				{
					auto a = buf_a.get_access<sycl::access::mode::read>(h);
					auto b = buf_b.get_access<sycl::access::mode::read>(h);
					auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
					auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

					const size_t global_size = diag_len / internal_iter;
					h.parallel_for(sycl::range<1>{ global_size },
						[=](sycl::item<1> item)
						{
							//int iter = 0;
							for(int iter = 0; iter < internal_iter; ++iter)
							{
								int steps = item;// *internal_iter + iter;
								// int i = i_first - steps;
								// int j = j_first + steps;
								int h_index = i_last + steps;
								int v_index = j_first + steps;
								int i = i_first - steps;
								int j = v_index;
								{
									// int h_index = m - 1 - i;
									// int v_index = j;
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
					);
				}
			);
		}
	}
	sw.stop();

	delete[] _h_strands;
	delete[] _v_strands;

	return sw.elapsed_ms();
}


double test_comb_partial_sycl_single(sycl::queue &q, const InputSequencePair &given, int iter_count)
{
	const int m = given.length_a;
	const int n = given.length_b;
	const int *_a = given.a;
	const int *_b = given.b;

	// initialize strands
	int *_h_strands = new int[m];
	int *_v_strands = new int[n];
	for (int i = 0; i < m; ++i) _h_strands[i] = i;
	for (int j = 0; j < n; ++j) _v_strands[j] = j + m;

	Stopwatch sw;
	{
		sycl::buffer<int, 1> buf_a(_a, m);
		sycl::buffer<int, 1> buf_b(_b, n);
		sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
		sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

		int wg_size = 4096;
		// actual combing happens here
		for (int diag_idx = m; diag_idx < m + iter_count; ++diag_idx)
		{
			int i_diag = diag_idx;
			int i_first = i_diag < m ? i_diag : m - 1;
			int j_first = i_diag < m ? 0 : i_diag - m + 1;

			int diag_len = Min(i_first + 1, n - j_first);

			q.submit(
				[&](auto &h)
				{
					auto a = buf_a.get_access<sycl::access::mode::read>(h);
					auto b = buf_b.get_access<sycl::access::mode::read>(h);
					auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
					auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

					h.single_task(
						[=]()
						{
							for (int steps = 0; steps < diag_len; ++steps)
							{
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
					);
				}
			);
		}
	}
	sw.stop();

	delete[] _h_strands;
	delete[] _v_strands;

	return sw.elapsed_ms();
}

void CombPartialCpu(const int *a, const int *b, int m, int n, int *h_strands, int *v_strands)
{
	int diag_count = m + n - 1;

	// need to pick antidiagonal such that it has some known length
	// say, the longest (length m)
	// int i_diag = m;

	for (int i_diag = m; i_diag < m + 100; ++i_diag)
	{
		int i_first = i_diag < m ? i_diag : m - 1;
		int j_first = i_diag < m ? 0 : i_diag - m + 1;

		// along antidiagonal, i goes down, j goes up
		int diag_len = Min(i_first + 1, n - j_first);

		// std::cout << diag_len << "\n";
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

void CombPartialCpuColumnMajor(const int *a, const int *b, int m, int n, int *h_strands, int *v_strands)
{

	for (int j = 0; j < 100; ++j)
	{
		for (int i = 0; i < m; ++i)
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

void CombPartialStairsCpu(const int *a, const int *b, int m, int n, int *h_strands, int *v_strands)
{
	int diag_count = m + n - 1;

	// need to pick antidiagonal such that it has some known length
	// say, the longest (length m)
	// int i_diag = m;

	// staircased execution now?
}
