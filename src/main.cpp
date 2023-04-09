#include "test_setup.hpp"
#include "test_utility.hpp"

#include "lcs_types.hpp"

#include <CL/sycl.hpp>
#include "dpc_common.hpp"

#include "algorithm_registry.hpp"

using dpc_common::TimeInterval;

#define PrintError(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__)


int main(int argc, char **argv)
{
	const char *in_filename = argc >= 2 ? argv[1] : "test_default.txt";
	const char *out_filename = argc >= 3 ? argv[2] : nullptr;

	TextFileData script(in_filename);
	LcsAlgorithmRegistry reg;

	auto commands = ParseEntireScript(script.text.c_str());
	int global_seed_counter = 1000;

	TestResultWriter out(out_filename);
	out.WriteCsvHeader();

	int counter = 0;
	for (auto cmd : commands)
	{
		++counter;

		// Prepare test case options
		auto opts = cmd.ParseOptions();
		// Prepare inputs
		if (opts.seed_a == -1) opts.seed_a = global_seed_counter++;
		if (opts.seed_b == -1) opts.seed_b = global_seed_counter++;

		auto a = generate_random_binary_sequence(opts.size_a, opts.seed_a);
		auto b = generate_random_binary_sequence(opts.size_b, opts.seed_b);
		LcsInput input(a.data(), a.size(), b.data(), b.size());

		// Find algorithm for device
		auto impl = reg.Get(opts.algorithm, opts.device_type);
		if (!impl.ok)
		{
			PrintError("Error: Unable to find algorihm '%s'\n", opts.algorithm.c_str());
			continue;
		}

		for (int iter = 0; iter < opts.iterations; ++iter)
		{
			try
			{
				// Start timer
				TimeInterval timer;

				// Produce permutation matrix
				auto perm = impl(input);

				// Stop timer
				double elapsed_ms = timer.Elapsed() * 1000.0;
				int64_t num_cells_total = (int64_t)opts.size_a * (int64_t)opts.size_b;
				double speed_cells_per_us = (double)num_cells_total / (elapsed_ms * 1000.0);

				int64_t hash = perm.hash();

				TestCaseResult res = {};
				res.device = impl.queue->get_info<sycl::info::queue::device>().get_info<sycl::info::device::name>();


				res.algorithm = opts.algorithm;
				res.device_type = opts.device_type;
				res.size_a = opts.size_a;
				res.size_b = opts.size_b;
				res.elapsed_ms = elapsed_ms;
				res.hash = hash;

				out.WriteLine(opts, res);
			} 
			catch (sycl::exception e)
			{
				PrintError("Sycl exception: \n");
				PrintError(">>> %s\n", e.what());
				PrintError("---------------\n");
			}
			catch (std::exception e)
			{
				PrintError("exception: \n");
				PrintError(">>> %s\n", e.what());
				PrintError("---------------\n");
			}
		}
		out.Flush();
	}

	return 0;
}