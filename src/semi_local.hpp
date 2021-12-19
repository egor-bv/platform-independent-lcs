#pragma once
#include "permutation.hpp"



#include <CL/sycl.hpp>
long long StickyBraidParallel(sycl::queue &q, InputSequencePair p)
{
	int m = p.length_a;
	int n = p.length_b;
	const int *a = p.a;
	const int *b = p.b;

	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	{
		sycl::buffer<int, 1> buf_a(p.a, p.length_a);
		sycl::buffer<int, 1> buf_b(p.b, p.length_b);

		sycl::buffer<int, 1> buf_h_strands(h_strands, m);
		sycl::buffer<int, 1> buf_v_strands(v_strands, n);

		int diag_count = m + n - 1;
		int max_diag_length = m; // say it's vertical!

		for (int i_diag = 0; i_diag < diag_count; ++i_diag)
		{
			int i_first = i_diag < m ? i_diag : m - 1;
			int j_first = i_diag < m ? 0 : i_diag - m + 1;
			// along antidiagonal, i goes down, j goes up
			int diag_len = std::min(i_first + 1, n - j_first);

			if (0)
			{
				if (i_diag % 100 == 99)
				{
					std::cout << "Submitting work on diag #" << i_diag << " of length " << diag_len << "\n";
				}
			}
			// each antidiagonal is a separate kernel
			q.submit(
				[&](auto &h)
				{
					// accessors are defined here
					auto acc_h_strands = buf_h_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto acc_v_strands = buf_v_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto acc_a = buf_a.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);
					auto acc_b = buf_b.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);

#if 1
					h.parallel_for(sycl::range<1>(diag_len),
						[=](auto iter_steps)
						{
#if 1
							int steps = iter_steps;
							// actual grid coordinates
							int i = i_first - steps;
							int j = j_first + steps;

							// now the core of the algo (and of course it doesn't work!)
							{
								int h_index = m - 1 - i;
								int v_index = j;
								int h_strand = acc_h_strands[h_index];
								int v_strand = acc_v_strands[v_index];
								bool need_swap = acc_a[i] == acc_b[j] || h_strand > v_strand;
								if (need_swap)
								{
									// swap
									acc_h_strands[h_index] = v_strand;
									acc_v_strands[v_index] = h_strand;
								}
							}
#endif
						}
					);
#endif

				}
			);

			if (0)
			{
				if (i_diag % 100 == 99)
				{
					std::cout << "Waiting on the batch..." << "\n";
					q.wait_and_throw();
				}
			}
		}
	}
	std::cout << "Now waiting..." << std::endl;
	// q.wait();

	{
		PermutationMatrix p(m + n);
		for (int l = 0; l < m; l++) p.set_point(h_strands[l], n + l);
		for (int r = m; r < m + n; r++) p.set_point(v_strands[r - m], r - m);

		if (p.size < 100)
		{
			std::cout << "DEBUG INFO:" << std::endl;
			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_row_by_col(row_i) << " ";
			}
			std::cout << std::endl;

			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_col_by_row(row_i) << " ";
			}
			std::cout << std::endl;
		}



		long long result = hash(p, p.size);
		delete[] h_strands;
		delete[] v_strands;
		return result;
	}
}

// Iterate in rectangular blocks to reduce number of dispatches
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

	{
		sycl::buffer<int, 1> buf_a(p.a, p.length_a);
		sycl::buffer<int, 1> buf_b(p.b, p.length_b);

		sycl::buffer<int, 1> buf_h_strands(h_strands, m);
		sycl::buffer<int, 1> buf_v_strands(v_strands, n);

		int original_m = m;
		int original_n = n;


		const int block_m = 2;
		const int block_n = 2;
		m = (m + block_m - 1) / block_m;
		n = (n + block_n - 1) / block_n;


		int diag_count = m + n - 1;
		int max_diag_length = m; // say it's vertical!

		for (int i_diag = 0; i_diag < diag_count; ++i_diag)
		{
			int i_first = i_diag < m ? i_diag : m - 1;
			int j_first = i_diag < m ? 0 : i_diag - m + 1;
			// along antidiagonal, i goes down, j goes up
			int diag_len = std::min(i_first + 1, n - j_first);

			// each block antidiagonal is a separate kernel
			q.submit(
				[&](auto &h)
				{
					// accessors are defined here
					auto acc_h_strands = buf_h_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto acc_v_strands = buf_v_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto acc_a = buf_a.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);
					auto acc_b = buf_b.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);

#if 1
					h.parallel_for(sycl::range<1>(diag_len),
						[=](auto iter_steps)
						{
#if 1
							int steps = iter_steps;
							// block coordinates
							int i_block = i_first - steps;
							int j_block = j_first + steps;

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
									if (need_swap)
									{
										// swap
										acc_h_strands[h_index] = v_strand;
										acc_v_strands[v_index] = h_strand;
									}
								}
							}
#endif
						}
					);
#endif

				}
			);

		}

		m = original_m;
		n = original_n;
	}


	{
		PermutationMatrix p(m + n);
		for (int l = 0; l < m; l++) p.set_point(h_strands[l], n + l);
		for (int r = m; r < m + n; r++) p.set_point(v_strands[r - m], r - m);

		if (p.size < 100)
		{
			std::cout << "DEBUG INFO:" << std::endl;
			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_row_by_col(row_i) << " ";
			}
			std::cout << std::endl;

			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_col_by_row(row_i) << " ";
			}
			std::cout << std::endl;
		}



		long long result = hash(p, p.size);
		delete[] h_strands;
		delete[] v_strands;
		return result;
	}
}

long long StickyBraidSimplest(const InputSequencePair &p)
{
	// NOTE(Egor): the order matters in this task, so make sure it works fine!
	int m = p.length_a;
	int n = p.length_b;
	const int *a = p.a;
	const int *b = p.b;

	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	// row-major order
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			int h_index = m - 1 - i;
			int v_index = j;
			int h_strand = h_strands[h_index];
			int v_strand = v_strands[v_index];


			if (a[i] == b[j] || h_strand > v_strand)
			{
				// swap
				h_strands[h_index] = v_strand;
				v_strands[v_index] = h_strand;
			}
		}
	}

	{
		PermutationMatrix p(m + n);
		for (int l = 0; l < m; l++) p.set_point(h_strands[l], n + l);
		for (int r = m; r < m + n; r++) p.set_point(v_strands[r - m], r - m);

		if (p.size < 100)
		{
			std::cout << "DEBUG INFO:" << std::endl;
			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_row_by_col(row_i) << " ";
			}
			std::cout << std::endl;

			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_col_by_row(row_i) << " ";
			}
			std::cout << std::endl;
		}



		long long result = hash(p, p.size);
		delete[] h_strands;
		delete[] v_strands;
		return result;
	}


}


long long StickyBraidAntidiagonal(const InputSequencePair &p)
{
	int m = p.length_a;
	int n = p.length_b;
	const int *a = p.a;
	const int *b = p.b;

	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	int diag_count = m + n - 1;
	int max_diag_length = m; // say it's vertical!

	for (int i_diag = 0; i_diag < diag_count; ++i_diag)
	{
		int i_first = i_diag < m ? i_diag : m - 1;
		int j_first = i_diag < m ? 0 : i_diag - m + 1;
		// along antidiagonal, i goes down, j goes up
		int diag_len = std::min(i_first + 1, n - j_first);
		for (int steps = 0; steps < diag_len; ++steps)
		{
			// actual grid coordinates
			int i = i_first - steps;
			int j = j_first + steps;

			// now the core of the algo (and of course it doesn't work!)
			{
				int h_index = m - 1 - i;
				int v_index = j;
				int h_strand = h_strands[h_index];
				int v_strand = v_strands[v_index];
				if (a[i] == b[j] || h_strand > v_strand)
				{
					// swap
					h_strands[h_index] = v_strand;
					v_strands[v_index] = h_strand;
				}
			}
		}
	}


	{
		PermutationMatrix p(m + n);
		for (int l = 0; l < m; l++) p.set_point(h_strands[l], n + l);
		for (int r = m; r < m + n; r++) p.set_point(v_strands[r - m], r - m);

		if (p.size < 100)
		{
			std::cout << "DEBUG INFO:" << std::endl;
			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_row_by_col(row_i) << " ";
			}
			std::cout << std::endl;

			for (int row_i = 0; row_i < p.size; ++row_i)
			{
				std::cout << p.get_col_by_row(row_i) << " ";
			}
			std::cout << std::endl;
		}



		long long result = hash(p, p.size);
		delete[] h_strands;
		delete[] v_strands;
		return result;
	}


}