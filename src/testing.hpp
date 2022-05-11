#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_device_selector.hpp>

#include "dpc_common.hpp"

#include "lcs_types.hpp"
#include "lcs_registry.hpp"
#include "testing_fna_parser.hpp"
#include "testing_utility.hpp"

struct CliOptions
{
	std::string algorithm;
	std::string device = "cpu";

	std::string a_file;
	std::string b_file;

	int a_size = 1024;
	int b_size = 1024;

	int a_seed = 1;
	int b_seed = 2;

	int iterations = 3;

	std::string test;
	bool verbose = true;
};

sycl::queue *GLOBAL_CPU_QUEUE;
sycl::queue *GLOBAL_GPU_QUEUE;
sycl::queue *GLOBAL_FPGA_QUEUE;
sycl::queue *GLOBAL_FPGA_EMU_QUEUE;
sycl::queue *GLOBAL_HOST_QUEUE;

sycl::queue *create_queue_for_device(const std::string &device)
{
	if (device == "cpu")
	{
		if (!GLOBAL_CPU_QUEUE)
		{
			GLOBAL_CPU_QUEUE = new sycl::queue(sycl::cpu_selector());
		}
		return GLOBAL_CPU_QUEUE;
	}
	if (device == "gpu")
	{
		if (!GLOBAL_GPU_QUEUE)
		{
			GLOBAL_GPU_QUEUE = new sycl::queue(sycl::gpu_selector());
		}
		return GLOBAL_GPU_QUEUE;
	}
	if (device == "fpga")
	{
		if (!GLOBAL_FPGA_QUEUE)
		{
			GLOBAL_FPGA_QUEUE = new sycl::queue(sycl::ext::intel::fpga_selector());
		}
		return GLOBAL_FPGA_QUEUE;
	}
	if (device == "fpga_emu")
	{
		if (!GLOBAL_FPGA_EMU_QUEUE)
		{
			GLOBAL_FPGA_EMU_QUEUE = new sycl::queue(sycl::ext::intel::fpga_emulator_selector());
		}
		return GLOBAL_FPGA_EMU_QUEUE;
	}
	if (device == "host")
	{
		if (!GLOBAL_HOST_QUEUE)
		{
			GLOBAL_HOST_QUEUE = new sycl::queue(sycl::host_selector());
		}
		return GLOBAL_HOST_QUEUE;
	}
	return nullptr;
}


struct TestResult
{
	double elapsed_ms;
	bool succeeded = false;
	bool checked = false;
	bool correct = false;
	double similarity = 0.0f;
	int64_t hash = 0;
};

struct CorrectnessStats
{
	int num_total = 0;
	int num_correct = 0;
};


TestResult run_single_test_case(sycl::queue *q, const std::string &algo_name, const LcsInput &input, bool checked)
{
	// TODO: handle runtime failure somewhere around here

	auto *reg = get_global_lcs_registry();
	auto entry = reg->Solvers().find(algo_name);

	if (entry == reg->Solvers().end())
	{
		// Algorithm name not in registry
		// return empty result (test failed)
		return TestResult{};
	}
	else
	{
		auto algo = entry->second;
		algo.SetQueue(q);


		dpc_common::TimeInterval timer;
		auto perm = algo.Semilocal(input);
		double elapsed_ms = timer.Elapsed() * 1000.0;

		TestResult result = {};

		result.succeeded = true;
		result.elapsed_ms = elapsed_ms;
		result.hash = perm.hash();

		if (checked)
		{
			result.checked = true;
			// TODO: compare to reference
			auto perm_ref = reg->Solvers().find("reference")->second.Semilocal(input);
			double sim = perm_ref.Similarity(perm);
			if (perm_ref == perm)
			{
				result.correct = true;
			}
			else
			{
				printf("  (Sim: %f)  ", sim);
			}
		}

		return result;
	}
}


void run_correctness_tests(sycl::queue *q, const std::string &algo_name, bool print_wrong, bool print_correct)
{
	CorrectnessStats stats = {};

	// subgroup size will be <= 32, so this should catch most bugs
	// related to alignment
	{
		for (int m = 1; m < 66; ++m)
		{
			for (int n = 1; n < 66; ++n)
			{
				auto a = generate_random_binary_sequence(m, 1111);
				auto b = generate_random_binary_sequence(n, 2222);
				auto input = LcsInput(a.data(), a.size(), b.data(), b.size());
				auto result = run_single_test_case(q, algo_name, input, true);
				if (result.correct)
				{
					stats.num_correct++;
					if (print_correct)
					{
						printf("[%d] (%d x %d) correct!\n", stats.num_total, input.a_size, input.b_size);
					}
				}
				else
				{
					if (print_wrong)
					{
						printf("[%d] (%d x %d) WRONG ANSWER\n", stats.num_total, input.a_size, input.b_size);
					}
				}
				stats.num_total++;
			}
		}
	}

	// print summary
	{
		printf("Passed: %d/%d (%f%%)\n", stats.num_correct, stats.num_total,
			double(stats.num_correct) / stats.num_total * 100.0);
	}
}

void cli_test_run(CliOptions &opts)
{
	#define VERBOSE(...) if (opts.verbose) printf(__VA_ARGS__);
	VERBOSE("================\n");
	VERBOSE("Requested device: %s\n", opts.device.c_str());
	if (opts.device != "cpu" && opts.device != "gpu")
	{
		VERBOSE("Unknown device type, try: cpu | gpu\n");
		return;
	}
	auto *queue = create_queue_for_device(opts.device);
	auto *reg = get_global_lcs_registry();
	// TODO: print device info

	if (opts.test.empty())
	{
		VERBOSE("Requested custom test case...\n");
		std::vector<int> seq_a;
		std::vector<int> seq_b;

		if (opts.a_file.empty())
		{
			seq_a = generate_random_binary_sequence(opts.a_size, opts.a_seed);
			VERBOSE("> a source: random(seed=%d), size: %d\n", opts.a_seed, (int)seq_a.size());
		}
		else
		{
			seq_a = load_fna_file(opts.a_file);
			VERBOSE("> a source: file %s, size: %d\n", opts.a_file.c_str(), (int)seq_a.size());
		}
		if (opts.b_file.empty())
		{
			seq_b = generate_random_binary_sequence(opts.b_size, opts.b_seed);
			VERBOSE("> b source: random(seed=%d), size: %d\n", opts.b_seed, (int)seq_b.size());
		}
		else
		{
			seq_b = load_fna_file(opts.b_file);
			VERBOSE("> b source: file %s, size: %d\n", opts.b_file.c_str(), (int)seq_b.size());
		}

		int64_t total_cells = (int64_t)seq_a.size() * (int64_t)seq_b.size();
		{
			VERBOSE("Using algorithm: %s\n", opts.algorithm.c_str());
			VERBOSE("Doing %d iterations\n", opts.iterations);
			VERBOSE("Total cells to process: %lld\n", total_cells);
			VERBOSE("================\n\n");
		}

		auto input = LcsInput(seq_a.data(), seq_a.size(), seq_b.data(), seq_b.size());
		for (int iter = 0; iter < opts.iterations; ++iter)
		{
			auto result = run_single_test_case(queue, opts.algorithm, input, false);
			double cells_per_us = total_cells / result.elapsed_ms / 1000.0;
			
			const char *correctness_msg = result.checked ? (result.correct ? "+" : "-") : "?";
			printf("[%s] time = %f ms, speed = %f cell/us, #%lld\n", 
				correctness_msg, result.elapsed_ms, cells_per_us, result.hash);
		}
		printf("\n");
	}
	else
	{
		VERBOSE("Requested named test case: %s\n", opts.test.c_str());
		if (opts.test == "benchmark_100k")
		{
		}
		if (opts.test == "benchmark_1000k")
		{
		}
		if (opts.test == "correctness")
		{
			if (opts.algorithm == "all")
			{
				// check correctness for all algorithms
			}
			else
			{
				run_correctness_tests(queue, opts.algorithm, true, true);
			}
		}
	}

	#undef VERBOSE
}