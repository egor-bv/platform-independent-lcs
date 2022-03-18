#pragma once
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "stopwatch.hpp"
#include "testing_utility.hpp"

#include "permutation.hpp"
#include "semi_lcs_types.hpp"
#include "utility.hpp"

static sycl::queue *FPGA_QUEUE_GLOBAL = nullptr;


sycl::queue &GetSomeQueue()
{
	if (!FPGA_QUEUE_GLOBAL)
	{
#ifdef  REAL_FPGA
		{
			FPGA_QUEUE_GLOBAL = new sycl::queue(sycl::ext::intel::fpga_selector());
		}
#else
		{
			FPGA_QUEUE_GLOBAL = new sycl::queue(sycl::cpu_selector());
		}
#endif //  REAL_FPGA
	}
	return *FPGA_QUEUE_GLOBAL;
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

void semi_simple(LcsProblem &p)
{
	auto &q = GetSomeQueue();

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


// using this to only compile a single kernel that works for fpga
int fpga_main(int argc, char **argv)
{
	auto &q = GetSomeQueue();

	int input_size = 1024;
	int num_iterations = 4;

	auto example = ExampleInput(input_size, input_size);


	std::cout << "doing something...\n";
	for (int iter = 0; iter < num_iterations; ++iter)
	{
		LcsProblem problem = {};
		problem.input_a = example.a;
		problem.input_b = example.b;
		problem.size_a = example.length_a;
		problem.size_b = example.length_b;

		Stopwatch sw;
		// do something

		semi_simple(problem);

		sw.stop();
		double epus = double(problem.size_a) *double(problem.size_b) / sw.elapsed_ms() / 1000.0;
		std::cout << epus << "c/us,   " << sw.elapsed_ms() << " ms\n";

	}
	return 0;
}