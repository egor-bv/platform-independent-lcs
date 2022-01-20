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

	for (int i = 0; i < 2; ++i)
	{
		if (1)
		{

			// semi_parallel_single_task(q, input);
			Stopwatch sw;

			auto p = semi_parallel_single_task(q, input);
			long long hsh = hash(p, p.size);


			sw.stop();

			auto elements_per_ms = (double)input.length_a * input.length_b / sw.elapsed_ms();
			std::cout << "\n===============================================================\n";
			std::cout << "semi-local finished!" << "\n";
			std::cout << "time taken  = " << sw.elapsed_ms() << "ms" << "\n";

			std::cout << "elements/us = " << elements_per_ms / 1000.0 << "\n";
			std::cout << "lcs hash (my)  = " << hsh << "\n";

			csv_writer.AppendResult("session", "parallel", "some a", "some b", input.length_a, input.length_b, sw.elapsed_ms(), hsh);
		}
	}
}

void test_partial_combing(sycl::queue q, int size)
{
	auto given = ExampleInput(size, size + 1000);
	std::cout << "\n== PARTIAL COMBING ==\n";

	{
		auto time_ms = test_comb_partial_cpu(CombPartialCpuColumnMajor, given);
		auto elements_per_us = 100*given.length_a / time_ms / 1000.0f;
		std::cout << size << ": t = " << time_ms << "; e/us = "<< elements_per_us << "\n";
	}

	std::cout << "\n== SYCL ==\n";
	{
		auto time_ms = test_comb_partial_sycl(q, given, 100);
		auto elements_per_us = 100.0 * given.length_a / time_ms / 1000.0;
		std::cout << size << ": t = " << time_ms << "; e/us = " << elements_per_us << "\n";
	}

}

int main(int argc, char **argv)
{


	// create a sycl queue
	sycl::cpu_selector device_selector;
	sycl::queue q(device_selector, dpc_common::exception_handler);
	
	// InputSequencePair input = ExampleInput(18105, 28201);

	for (int i = 0; i < 2; ++i)
	{
		test_partial_combing(q, 10000);
		test_partial_combing(q, 100000);
		test_partial_combing(q, 200000);
		test_partial_combing(q, 250000);
		test_partial_combing(q, 300000);
		test_partial_combing(q, 3000000);
		//test_partial_combing(q, 300000);
		//test_partial_combing(q, 300000);
	}

}

