#pragma once
#include "permutation.hpp"
#include "utility.hpp"


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

	{
		sycl::buffer<int, 1> buf_a(p.a, p.length_a);
		sycl::buffer<int, 1> buf_b(p.b, p.length_b);

		sycl::buffer<int, 1> buf_h_strands(h_strands, m);
		sycl::buffer<int, 1> buf_v_strands(v_strands, n);

		int original_m = m;
		int original_n = n;


		const int block_m = 32;
		const int block_n = 64;
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

									{
										int r = (int)need_swap;
										acc_h_strands[h_index] = (h_strand & (r - 1)) | ((-r) & v_strand);
										acc_v_strands[v_index] = (v_strand & (r - 1)) | ((-r) & h_strand);
									}
								}
							}
#endif
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




long long StickyBraidParallelStairs(sycl::queue &q, InputSequencePair p)
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

		const int stairs_m = 2;
		const int stairs_n = 2;

		const int big_m = SmallestMultipleToFit(stairs_m, m);
		const int overall_leftmost = -m + big_m;
		const int big_n = SmallestMultipleToFit(stairs_n, n - overall_leftmost);

		const int big_diag_count = big_m + big_n - 1;
		std::cout << big_diag_count;

		for (int big_diag_idx = 0; big_diag_idx < big_diag_count; ++big_diag_idx)
		{
			const bool starts_on_left = big_diag_idx < big_m;
			int i_big_diag_first = starts_on_left ? big_diag_idx : big_m - 1;
			int j_big_diag_first = starts_on_left ? 0 : big_diag_idx - big_m + 1;
			int big_diag_len = Min(i_big_diag_first + 1, big_n - j_big_diag_first);

			// clip big diagonals
			{
				const int horz_coverage = stairs_m + stairs_n - 1;
				const int horz_stride = stairs_m - 1;

				const int i_big_diag_end = i_big_diag_first + big_diag_len;
				const int j_big_diag_end = j_big_diag_first + big_diag_len;

				const int diag_leftmost = -horz_stride * (i_big_diag_first + 1) + j_big_diag_first * stairs_n;
				const int diag_rightmost = -horz_stride * (i_big_diag_end + 1) + j_big_diag_end * stairs_n;

				const int diag_extra_left = Max(0, -diag_leftmost);
				const int diag_extra_right = Max(0, diag_rightmost - n);
				const int clip_first = diag_extra_left / horz_coverage;
				const int clip_last = diag_extra_right / horz_coverage;

				i_big_diag_first -= clip_first;
				j_big_diag_first += clip_first;
				big_diag_len -= clip_first + clip_last;
			}

			const int global_size = stairs_m * big_diag_len;
			const int local_size = stairs_m;
			// if (global_size == 0) continue;
			// std::cout << global_size << " " << local_size << std::endl;

#if 0
			q.submit(
				[&](auto &h)
				{
					auto global_h_strands = buf_h_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto global_v_strands = buf_v_strands.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
					auto global_a = buf_a.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);
					auto global_b = buf_b.get_access<sycl::access::mode::read, sycl::access::target::global_buffer>(h);

					sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> local_h_strands({ local_size }, h);
					sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> local_v_strands({ local_size }, h);

					h.parallel_for(sycl::nd_range<1>{ global_size, local_size },
						[=](auto item)
						{
							int group_id = item.get_group_linear_id();
							int local_id = item.get_local_linear_id();

							int i_stairs = i_big_diag_first - group_id;
							int j_stairs = j_big_diag_first + group_id;

							int i = i_stairs * stairs_m + local_id;
							int j = j_stairs * stairs_n - i_stairs * (stairs_n - 1) - local_id;

							int h_index = m - 1 - i;
							int v_index = j;
							int local_h_index = h_index % local_size;

							int safe_i = Clamp(i, 0, m - 1);
							int safe_h_index = h_index % m;
							int safe_v_index = v_index % n;

							// 1. initialize local(shared) memory
							local_h_strands[local_h_index] = global_h_strands[safe_h_index];
							local_v_strands[v_index % local_size] = global_v_strands[safe_v_index];

							int a_sym = global_a[safe_i];

							item.barrier();

							// 2. iterate
							for (int step = 0; step < stairs_n; ++step)
							{
								int safe_j = Clamp(j, 0, n - 1);
								int b_sym = global_b[safe_j];
								int local_v_index = v_index % local_size;
								int safe_v_index = v_index % n;

								// load rightmost
								if (local_id == 0 && safe_h_index == h_index)
								{
									local_h_strands[local_h_index] = global_h_strands[h_index];
								}

								// compute frontier
								int h_strand = local_h_strands[local_h_index];
								int v_strand = local_v_strands[local_v_index];

								bool need_swap = (a_sym == b_sym || h_strand > v_strand) && safe_v_index == v_index && safe_h_index == h_index;
								if (need_swap)
								{
									local_h_strands[local_h_index] = need_swap ? v_strand : h_strand;
									local_v_strands[local_v_index] = need_swap ? h_strand : v_strand;
								}

								// store leftmost
								if (local_id + 1 == local_size && safe_h_index == h_index)
								{
									global_h_strands[h_index] = local_h_strands[local_h_index];
								}
								item.barrier();
								++v_index;
								++j;
							}

							// 3. flush remaining entries to global memory
							if (h_index == h_index % m) global_h_strands[h_index] = local_h_strands[h_index % local_size];
							if (v_index == v_index % n) global_v_strands[v_index] = local_v_strands[v_index % local_size];

						}
					);
				}
			);
#else
			for (int group_id = 0; group_id < big_diag_len; ++group_id)
			{
				for (int local_id = 0; local_id < local_size; ++local_id)
				{

					int i_stairs = i_big_diag_first - group_id;
					int j_stairs = j_big_diag_first + group_id;

					int i = i_stairs * stairs_m + local_id;
					int j = j_stairs * stairs_n - i_stairs * (stairs_m - 1) - local_id;

					// if (i < 0 || i >= m) continue;

					int h_index = m - 1 - i;
					int v_index = j;
					// int local_h_index = h_index % local_size;

					int safe_i = Clamp(i, 0, m - 1);
					int a_sym = a[safe_i];

					if (h_index < 0 || h_index >= m) continue;
					// 2. iterate
					for (int step = 0; step < stairs_n; ++step)
					{


						if (v_index < 0 || v_index >= n) { std::cout << v_index << std::endl; continue; }

						int safe_j = Clamp(j, 0, n - 1);
						int b_sym = b[safe_j];

						// compute frontier
						int h_strand = h_strands[h_index];
						int v_strand = v_strands[v_index];

						bool need_swap = (a_sym == b_sym || h_strand > v_strand);
						{
							h_strands[h_index] = need_swap ? v_strand : h_strand;
							v_strands[v_index] = need_swap ? h_strand : v_strand;
						}

						++v_index;
						++j;
					}

				}
			}


#endif
		}

	}
	// return 0;
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


long long
StickyBraidAntidiagonalStairs(const InputSequencePair &pair)
{
	int m = pair.length_a;
	int n = pair.length_b;
	const int *a = pair.a;
	const int *b = pair.b;
	int *h_strands = new int[m];
	int *v_strands = new int[n];

	for (int i = 0; i < m; ++i) h_strands[i] = i;
	for (int j = 0; j < n; ++j) v_strands[j] = j + m;

	int diag_count = m + n - 1;

	// should pass these in
	constexpr int block_m = 1;
	constexpr int block_n = 1;

	int big_m = SmallestMultipleToFit(block_m, m);
	int overall_leftmost = -m + big_m;
	int big_n = SmallestMultipleToFit(block_n, n - overall_leftmost);
	int block_diag_count = big_m + big_n - 1;

	size_t total_iter_count = 0;
	for (int block_diag_idx = 0; block_diag_idx < block_diag_count; ++block_diag_idx)
	{
		bool starts_on_left = block_diag_idx < big_m;
		int block_i_first = starts_on_left ? block_diag_idx : big_m - 1;
		int block_j_first = starts_on_left ? 0 : block_diag_idx - big_m + 1;
		int block_count = Min(block_i_first + 1, big_n - block_j_first);

		// clip against rectangle
		{
			int horz_coverage = block_m + block_n - 1;
			int horz_stride = block_m - 1;

			int i_end = block_i_first - block_count;
			int j_end = block_j_first + block_count;

			int leftmost = -horz_stride * (block_i_first + 1) + block_j_first * block_n;
			int rightmost = -horz_stride * (i_end + 1) + j_end * block_n;

			int extra_left = Max(0, -leftmost);
			int extra_right = Max(0, rightmost - n);
			int clip_first = extra_left / horz_coverage;
			int clip_last = extra_right / horz_coverage;

			block_i_first -= clip_first;
			block_j_first += clip_first;
			block_count -= clip_first + clip_last;
		}


		for (int block_idx = 0; block_idx < block_count; ++block_idx)
		{
			int i_first = block_i_first * block_m;
			int j_first = block_j_first * block_n - block_i_first * (block_m - 1);

			for (int iter = 0; iter < block_n; ++iter)
			{
				for (int local_id = 0; local_id < block_m; ++local_id)
				{

					int i = i_first + local_id;
					int j = j_first - local_id;

					bool inside = i >= 0 && i < m && j >= 0 && j < n;
					if (!inside) continue;

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
					++total_iter_count;
				}
			}
		}
	}
	std::cout << "(" << total_iter_count <<")\n";
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
				{
					h_strands[h_index] = need_swap ? v_strand : h_strand;
					v_strands[v_index] = need_swap ? h_strand : v_strand;
				}

			}
		}
	}
}

void AntidiagonalCombTest(const int *a, const int *b, int m, int n, int *h_strands, int *v_strands)
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
				{
					h_strands[h_index] = need_swap ? v_strand : h_strand;
					v_strands[v_index] = need_swap ? h_strand : v_strand;
				}

			}
		}
	}
}

void AntidiagonalCombStairs(const int *a, const int *b, int m, int n, int *h_strands, int *v_strands)
{
	const int block_m = 16;
	const int block_n = 128;

	int diag_count = m + n - 1;

	for (int i_diag = 0; i_diag < diag_count; ++i_diag)
	{
		int i_first = i_diag < m ? i_diag : m - 1;
		int j_first = i_diag < m ? 0 : i_diag - m + 1;
		// along antidiagonal, i goes down, j goes up
		int diag_len = Min(i_first + 1, n - j_first);
		
		

		// this is the best candidate for innel loop
		// just need to make an outer loop to match
		for (int local_id = 0; local_id < diag_len; ++local_id)
		{
			// actual grid coordinates
			int i = i_first - local_id;
			int j = j_first + local_id;

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



// this is somewhat slower just because order is switched
void AntidiagonalCombTopDown(const int *a, const int *b, int m, int n, int *h_strands, int *v_strands)
{
	int diag_count = m + n - 1;

	for (int diag_idx = 0; diag_idx < diag_count; ++diag_idx)
	{
		int i_first = diag_idx < n ? 0 : diag_idx - n + 1;
		int j_first = diag_idx < n ? diag_idx : n - 1;
		// along antidiagonal, i goes UP, j goes DOWN
		int diag_len = Min(j_first + 1, m - i_first);

		for (int steps = 0; steps < diag_len; ++steps)
		{
			// actual grid coordinates
			int i = i_first + steps;
			int j = j_first - steps;

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

	 // AntidiagonalCombBottomUp(a, b, m, n, h_strands, v_strands);

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
				 {
					 h_strands[h_index] = need_swap ? v_strand : h_strand;
					 v_strands[v_index] = need_swap ? h_strand : v_strand;
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