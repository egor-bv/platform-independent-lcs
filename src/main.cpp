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


void compare_all(sycl::queue q, const InputSequencePair &input)
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
		return [f,&q](auto p) { return f(q, p); };
	};

	print_test("antidiagonal_cpu", semi_cpu_antidiag);
	print_test("single_task", with_queue(semi_parallel_single_task));
	print_test("single_task_row_major", with_queue(semi_parallel_single_task_row_major));
	print_test("single_subgroup", with_queue(semi_parallel_single_sub_group));
	print_test("blockwise", with_queue(StickyBraidParallelBlockwise));
}


void compare_combing(sycl::queue q, InputSequencePair input)
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
		auto time_ms = test_comb_partial_sycl(q, given, 10000);
		auto elements_per_us = 10000.0 * given.length_a / time_ms / 1000.0;
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

int main(int argc, char **argv)
{
	char device_type = 'c';
	if (argc > 0 && argv[0][0] == 'g')
	{
		device_type = 'g';
		std::cout << "using gpu device...\n\n";
	}
	else
	{
		std::cout << "using cpu device\n\n";
	}

	auto q = create_queue(device_type);

	compare_all(q, ExampleInput(40001, 50003));
}

