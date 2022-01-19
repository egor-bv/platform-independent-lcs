#include <chrono>
#include <cmath>
#include <iostream>

#include "CL/sycl.hpp"
#include "dpc_common.hpp"

#include "lcs.hpp"
#include "semi_local.hpp"

#include "stopwatch.hpp"
#include "testing_utility.hpp"


int main(int argc, char **argv)
{
	// NOTE(Egor): for test purposes, hardcoded filenames here
	const char *file_a = "1.fna";
	const char *file_b = "2.fna";

	// loading input files
	int dataset_a_size = 0;
	int *dataset_a = LoadIntArrayFromFna(file_a, &dataset_a_size);

	int dataset_b_size = 0;
	int *dataset_b = LoadIntArrayFromFna(file_b, &dataset_b_size);

	// InputSequencePair input(dataset_a, dataset_a_size, dataset_b, dataset_b_size);
	InputSequencePair input = ExampleInput(8105, 8201);

	ResultCsvWriter csv_writer("test");

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



	if (1)
	{
		// create a queue
		sycl::cpu_selector device_selector;
		sycl::queue q(device_selector, dpc_common::exception_handler);

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

