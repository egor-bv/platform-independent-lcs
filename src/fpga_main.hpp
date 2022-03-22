#pragma once
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "stopwatch.hpp"
#include "testing_utility.hpp"

#include "permutation.hpp"
#include "semi_lcs_types.hpp"
#include "utility.hpp"


struct LcsProblem
{
	const int *input_a;
	const int *input_b;
	int size_a;
	int size_b;

	int llcs;
	int64_t perm_hash;
};

static sycl::queue *FPGA_QUEUE_GLOBAL = nullptr;
static bool USE_FPGA = false;
static bool USE_FPGA_EMULATOR = false;


sycl::queue &GetFpgaQueue()
{
	if (!FPGA_QUEUE_GLOBAL)
	{
		if (USE_FPGA_EMULATOR)
		{
			FPGA_QUEUE_GLOBAL = new sycl::queue(sycl::ext::intel::fpga_emulator_selector());
		}
		else if (USE_FPGA)
		{
			FPGA_QUEUE_GLOBAL = new sycl::queue(sycl::ext::intel::fpga_selector());
		}
		else
		{
			FPGA_QUEUE_GLOBAL = new sycl::queue(sycl::cpu_selector());
		}
	}
	return *FPGA_QUEUE_GLOBAL;
}

#if 0
void semi_simple(LcsProblem &p)
{
	auto &q = GetFpgaQueue();

	int m = p.size_a;
	int n = p.size_b;

	int *a_data = new int[m];
	int *b_data = new int[n];

	int *h_strands_data = new int[m];
	int *v_strands_data = new int[n];

	for (int i = 0; i < m; ++i) a_data[i] = p.input_a[m - i - 1];
	for (int j = 0; j < n; ++j) b_data[j] = p.input_b[j];

	for (int i = 0; i < m; ++i) h_strands_data[i] = i;
	for (int j = 0; j < n; ++j) v_strands_data[j] = m + j;

	int diag_count = m + n - 1;
	{
		sycl::buffer<int, 1> buf_a(a_data, m);
		sycl::buffer<int, 1> buf_b(b_data, n);
		sycl::buffer<int, 1> buf_h_strands(h_strands_data, m);
		sycl::buffer<int, 1> buf_v_strands(v_strands_data, n);


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

									bool need_swap = a[i] == b[j] || h_strand > v_strand;

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

	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands_data, m, v_strands_data, n);

	delete[] a_data;
	delete[] b_data;
	delete[] h_strands_data;
	delete[] v_strands_data;

	p.perm_hash = result.hash();
}
#endif


void semi_simple(LcsProblem &p)
{
	auto &q = GetFpgaQueue();

	int m = p.size_a;
	int n = p.size_b;

	int *a_data = new int[m];
	int *b_data = new int[n];

	int *h_strands_data = new int[m];
	int *v_strands_data = new int[n];

	for (int i = 0; i < m; ++i) a_data[i] = p.input_a[m - i - 1];
	for (int j = 0; j < n; ++j) b_data[j] = p.input_b[j];

	for (int i = 0; i < m; ++i) h_strands_data[i] = i;
	for (int j = 0; j < n; ++j) v_strands_data[j] = m + j;

	constexpr int ROW_M_BITS = 5;
	constexpr int ROW_M = 1 << ROW_M_BITS;
	constexpr int ROW_M_MASK = ROW_M - 1;

	int row_count = m / ROW_M;

	{
		sycl::buffer<int, 1> buf_a(a_data, m);
		sycl::buffer<int, 1> buf_b(b_data, n);
		sycl::buffer<int, 1> buf_h_strands(h_strands_data, m);
		sycl::buffer<int, 1> buf_v_strands(v_strands_data, n);


		q.submit([&](auto &h)
			{
				auto a = buf_a.get_access<sycl::access::mode::read>(h);
				auto b = buf_b.get_access<sycl::access::mode::read>(h);
				auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
				auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

				h.single_task([=]()
					{

						for (int row = row_count; row > 0; --row)
						{
							int i_bottom = (row - 1) * ROW_M;

							// local registers
							int Hs[ROW_M];
							int As[ROW_M];

							// shift register
							int Vs[ROW_M];
							int Bs[ROW_M];


							#pragma unroll
							[[intel::ivdep]]
							for (int i = 0; i < ROW_M; ++i)
							{
								Hs[i] = h_strands[i_bottom + i];
							}

							#pragma unroll
							[[intel::ivdep]]
							for (int i = 0; i < ROW_M; ++i)
							{
								As[i] = a[i_bottom + i];
							}

							for (int horz_step = 1 - ROW_M; horz_step < n; ++horz_step)
							{
								// load another 
								int j_right = horz_step + ROW_M - 1;
								int j_left = horz_step;

								if (j_right < n)
								{
									Vs[ROW_M - 1] = v_strands[j_right];
									Bs[ROW_M - 1] = b[j_right];
								}

								#pragma unroll
								[[intel::ivdep]]
								for (int step = 0; step < ROW_M; ++step)
								{
									int i = i_bottom + step;
									int j = j_left + step;

									int ii = step;
									int jj = step;

									bool inside = 0 <= j && j < n;

									int h = Hs[ii];
									int v = Vs[jj];
									bool need_swap = (As[ii] == Bs[jj] || h > v) && inside;

									Hs[ii] = need_swap ? v : h;
									Vs[jj] = need_swap ? h : v;
								}

								// store V
								if (j_left >= 0)
								{
									v_strands[j_left] = Vs[0];
								}

								// shift register
								#pragma unroll
								for (int jj = 0; jj < ROW_M - 1; ++jj)
								{
									Vs[jj] = Vs[jj + 1];
									Bs[jj] = Bs[jj + 1];
								}
							}

							// store h_strands back
							#pragma unroll
							for (int i = 0; i < ROW_M; ++i)
							{
								h_strands[i_bottom + i] = Hs[i];
							}

						}
					}
				);
			}
		);
	}

	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands_data, m, v_strands_data, n);

	delete[] a_data;
	delete[] b_data;
	delete[] h_strands_data;
	delete[] v_strands_data;

	p.perm_hash = result.hash();
}
