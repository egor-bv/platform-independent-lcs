#include <chrono>
#include <cmath>
#include <iostream>

#include "CL/sycl.hpp"
#include "dpc_common.hpp"

#include "semi_lcs_types.hpp"
#include "semi_lcs_api.hpp"

#include "stopwatch.hpp"
#include "testing_utility.hpp"





template<class F>
void RunBenchmark2(F func, int input_size, int num_iterations)
{
	auto given = ExampleInput(input_size, input_size);
	auto &q = GetOrCreateQueue("cpu");
	int64_t total_size = (int64_t)input_size * input_size;
	std::cout << "... ... ...\n";
	for (int iter = 0; iter < num_iterations; ++iter)
	{
		Stopwatch sw;
		auto result = func(q, given);
		sw.stop();

		int64_t result_hash = result.hash();
		double time_taken_ms = sw.elapsed_ms();
		double cells_per_us = total_size / time_taken_ms / 1000.0;

		std::cout << cells_per_us << " cells/us" << "; hash = " << result_hash << "\n";


	}
}


void RunBenchmarkCpu(std::string algo_name, int input_size, int num_iterations)
{
	auto given = ExampleInput(input_size, input_size);

	auto run_tests = [&](std::string header, auto func)
	{
		std::cout << header << "\n";
		int64_t total_size = (int64_t)given.length_a * (int64_t)given.length_b;
		std::cout << "(total_size = " << total_size << ")\n";
		std::cout << "===\n";

		for (int iter = 0; iter < num_iterations; ++iter)
		{
			Stopwatch sw;
			auto result = func(given);
			sw.stop();

			int64_t result_hash = result.hash();
			double time_taken_ms = sw.elapsed_ms();
			double cells_per_us = total_size / time_taken_ms / 1000.0;

			std::cout << cells_per_us << " cells/us" << "; hash = " << result_hash << "\n";


		}
		std::cout << "===\n\n";
	};



	if (algo_name == "reference")
	{
		run_tests(algo_name, SemiLcs_Reference);
	}
	else
	{
		auto &q = GetOrCreateQueue("cpu");

		auto with_queue = [&](auto f)
		{
			return [f, &q](auto p) { return f(q, p); };
		};

		auto run_tests_with_q = [&](std::string header, auto func)
		{
			run_tests(header, with_queue(func));
		};

		if (algo_name == "antidiagonal8")
		{
			run_tests_with_q(algo_name, SemiLcs_SubgroupAntidiagonal8);
		}
		else if (algo_name == "antidiagonal16")
		{
			run_tests_with_q(algo_name, SemiLcs_SubgroupAntidiagonal16);
		}
		else if (algo_name == "staircase-global8")
		{
			run_tests_with_q(algo_name, SemiLcs_SubgroupStaircaseGlobal8);
		}
		else if (algo_name == "staircase-global16")
		{
			run_tests_with_q(algo_name, SemiLcs_SubgroupStaircaseGlobal16);
		}
		else if (algo_name == "staircase-crosslane")
		{
			run_tests_with_q(algo_name, SemiLcs_SubgroupStaircaseCrosslane);
		}
		else if (algo_name == "staircase-local")
		{
			run_tests_with_q(algo_name, SemiLcs_SubgroupStaircaseLocal);
		}
		else if (algo_name == "tiled")
		{
			run_tests_with_q(algo_name, SemiLcs_Tiled);
		}
		else if (algo_name == "tiled-mt")
		{
			run_tests_with_q(algo_name, SemiLcs_Tiled_MT);
		}
		else if (algo_name == "tiled-new")
		{
			run_tests_with_q(algo_name, SemiLcs_Tiled_Universal<16, 3, 3, 24>);
		}
	}
}


#define COMPATIBLE_TESTING_INTERFACE 1
#if COMPATIBLE_TESTING_INTERFACE

#include "compatible_performance_benchmark.hpp"

int main(int argc, char **argv)
{
	RunSingleTest(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]), std::string(argv[4]));
	return 0;
}

#else
int main(int argc, char **argv)
{
	int input_size = 32 * 1024;
	int num_iterations = 4;
	std::string algo_name = "tiled-mt";

	// parse arguments to replace defaults
	if (argc > 1 && argv[1])
	{
		algo_name = std::string(argv[1]);
		std::cout << "\nreceived argument: " << algo_name << "\n";
	}
	if (argc > 2 && argv[2])
	{
		input_size = atoi(argv[2]);
	}
	if (argc > 3 && argv[3])
	{
		num_iterations = atoi(argv[3]);
	}


#if 0
	if (algo_name == "all")
	{
		RunBenchmarkCpu("reference", input_size, num_iterations);
		RunBenchmarkCpu("antidiagonal8", input_size, num_iterations);
		RunBenchmarkCpu("antidiagonal16", input_size, num_iterations);
		// RunBenchmarkCpu("staircase-global8", input_size, num_iterations);
		// RunBenchmarkCpu("staircase-global16", input_size, num_iterations);
		// RunBenchmarkCpu("staircase-crosslane", input_size, num_iterations);
		RunBenchmarkCpu("tiled", input_size, num_iterations);
		RunBenchmarkCpu("tiled-mt", input_size, num_iterations);
	}
	else
	{
		RunBenchmarkCpu(algo_name, input_size, num_iterations);
	}
#else
	{

#define BENCHMARK(SG_SIZE, TILE_M, TILE_N, TARGET_THREADS) \
		std::cout << SG_SIZE << " " << TILE_M << " " << TILE_N << " " << TARGET_THREADS << "\n";\
	RunBenchmark2(SemiLcs_Tiled_Universal<SG_SIZE, TILE_M, TILE_N, TARGET_THREADS>, input_size, num_iterations);

		BENCHMARK(16, 1, 1, 16);
		BENCHMARK(16, 2, 2, 16);
		BENCHMARK(16, 4, 4, 16);
		BENCHMARK(16, 8, 8, 16);

	}
#endif
	ShutdownQueues();
	return 0;
}

#endif