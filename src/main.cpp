#include <chrono>
#include <cmath>
#include <iostream>

#include "CL/sycl.hpp"
#include "dpc_common.hpp"

#include "lcs.hpp"
#include "semi_local.hpp"

#include "stopwatch.hpp"
#include "testing_utility.hpp"
#include "comb_partial.hpp"


void compare_all(sycl::queue &q, const InputSequencePair &input)
{
	auto print_test = [&](const char *test_name, auto f)
	{
		int num_iterations = 3;
		for (int iteration = 0; iteration < num_iterations; iteration++)
		{
			Stopwatch sw;
			auto p = f(input);
			sw.stop();

			double elem_per_us = (double)input.length_a * (double)input.length_b / sw.elapsed_ms() / 1000.0f;

			std::cout << test_name << ":  ";
			std::cout << elem_per_us << " e/us\n";
		}
		std::cout << "\n";
	};

	auto with_queue = [&](auto f)
	{
		return [f, &q](auto p) { return f(q, p); };
	};

	// print_test("antidiagonal_cpu", semi_cpu_antidiag);
	// print_test("single_task", with_queue(semi_parallel_single_task));
	// print_test("single_task_row_major", with_queue(semi_parallel_single_task_row_major));
	print_test("single_subgroup", with_queue(semi_parallel_single_sub_group));
	// print_test("blockwise", with_queue(StickyBraidParallelBlockwise));
	// print_test("naive_sycl", with_queue(semi_parallel_naive_sycl));
}


void compare_combing(sycl::queue &q, InputSequencePair input)
{
	ResultCsvWriter csv_writer("test");
	for (int i = 0; i < 2; ++i)
	{
		if (1)
		{
			Stopwatch sw;
			auto p = semi_cpu_antidiag(input);
			long long hsh = hash(p, p.size);


			sw.stop();

			auto elements_per_ms = (double)input.length_a * input.length_b / sw.elapsed_ms();
			std::cout << "\n===============================================================\n";
			std::cout << "semi-local finished!" << "\n";
			std::cout << "time taken  = " << sw.elapsed_ms() << "ms" << "\n";

			std::cout << "elements/us = " << elements_per_ms / 1000.0 << "\n";
			std::cout << "lcs hash (my)  = " << hsh << "\n";

			csv_writer.AppendResult("session", "serial", "some a", "some b", input.length_a, input.length_b, sw.elapsed_ms(), hsh);
		}
	}

	for (int i = 0; i < 4; ++i)
	{
		if (1)
		{

			long long hsh = 0;
			// semi_parallel_single_task(q, input);
			Stopwatch sw;
			if (0)
			{
				auto p = semi_parallel_single_task(q, input);
				long long hsh = hash(p, p.size);
			}
			else
			{
				hsh = StickyBraidParallelBlockwise(q, input);
			}

			sw.stop();

			auto elements_per_ms = (double)input.length_a * input.length_b / sw.elapsed_ms();
			std::cout << "\n===============================================================\n";
			std::cout << "sycl finished!" << "\n";
			std::cout << "time taken  = " << sw.elapsed_ms() << "ms" << "\n";

			std::cout << "elements/us = " << elements_per_ms / 1000.0 << "\n";
			std::cout << "lcs hash (my)  = " << hsh << "\n";

			csv_writer.AppendResult("session", "parallel", "some a", "some b", input.length_a, input.length_b, sw.elapsed_ms(), hsh);
		}
	}
}

void test_partial_combing(sycl::queue q, int size)
{
	auto given = ExampleInput(size, size + 10000);
	std::cout << "\n== PARTIAL COMBING ==\n";

	{
		auto time_ms = test_comb_partial_cpu(CombPartialCpu, given);
		auto elements_per_us = 100 * given.length_a / time_ms / 1000.0f;
		std::cout << size << ": t = " << time_ms << "; e/us = " << elements_per_us << "\n";
	}

	std::cout << "\n== SYCL ==\n";
	{
		auto time_ms = test_comb_partial_sycl_iter(q, given, 100);
		auto elements_per_us = 100.0 * given.length_a / time_ms / 1000.0;
		std::cout << size << ": t = " << time_ms << "; e/us = " << elements_per_us << "\n";
	}

}

sycl::queue create_queue(char device)
{
	if (device == 'g')
	{
		return sycl::queue(sycl::gpu_selector(), dpc_common::exception_handler);
	}
	else
	{
		return sycl::queue(sycl::cpu_selector(), dpc_common::exception_handler);
	}
}

void test_triad(sycl::queue &q, int num_elements)
{
	std::cout << "count = " << num_elements << "\n";
	{
		double cpu_ms = triad_cpu_ms(num_elements);
		double elem_per_us = num_elements / 1000.0f / cpu_ms;
		std::cout << "CPU: " << elem_per_us << "\n";
	}

	{
		double sycl_ms = triad_sycl_pinit_ms(q, num_elements);
		double elem_per_us = num_elements / 1000.0f / sycl_ms;
		std::cout << "SYCL: " << elem_per_us << "\n";
	}

	{
		double sycl_ms = triad_sycl_single_ms(q, num_elements);
		double elem_per_us = num_elements / 1000.0f / sycl_ms;
		std::cout << "SYCL(single): " << elem_per_us << "\n";
	}

	std::cout << "========\n";
}

void simd_test(char mode, int input_size, int num_iterations)
{
	auto input = ExampleInput(input_size, input_size);
	uint64_t total_elements = input_size * (uint64_t)input_size;
	std::cout << "Total number of elements: " << total_elements << "\n";

	auto print_test = [&](const char *test_name, auto f)
	{
		for (int iteration = 0; iteration < num_iterations; iteration++)
		{
			Stopwatch sw;
			auto p = f(input);
			sw.stop();

			auto result_hash = hash(p, p.size);
			double elem_per_us = (double)input.length_a * (double)input.length_b / sw.elapsed_ms() / 1000.0f;

			std::cout << test_name << ":  ";
			std::cout << elem_per_us << " e/us,   " << result_hash << "\n";
		}
		std::cout << "\n";
	};



	if (mode == 'h') // host
	{
		std::cout << "Ordinary cpu code:\n";
		print_test("antidiagonal_cpu", semi_cpu_antidiag);

	}
	else if (mode == 's') // sycl single subgroup
	{
		auto q = create_queue('c');

		auto with_queue = [&](auto f)
		{
			return [f, &q](auto p) { return f(q, p); };
		};

		std::cout << "SYCL single sub-group:\n";
		print_test("single_subgroup", with_queue(semi_parallel_single_sub_group));

	}
	else if (mode == 'm')
	{

		auto q = create_queue('c');

		auto with_queue = [&](auto f)
		{
			return [f, &q](auto p) { return f(q, p); };
		};

		std::cout << "SYCL single sub-group:\n";
		print_test("multithreaded", with_queue(semi_parallel_antidiag));
	}
}

int main(int argc, char **argv)
{
	char mode = 'h';
	int input_size = 80000;
	int num_iterations = 4;
	if (argc >= 1 && argv[1])
	{
		mode = argv[1][0];
	}
	if (argc >= 2 && argv[2])
	{
		input_size = atoi(argv[2]);
	}
	if (argc >= 3 && argv[3])
	{
		num_iterations = atoi(argv[3]);
	}

	//simd_test(mode, input_size, num_iterations);
	simd_test('h', input_size, num_iterations);
	simd_test('s', input_size, num_iterations);
	//simd_test('m', input_size, num_iterations);
}

