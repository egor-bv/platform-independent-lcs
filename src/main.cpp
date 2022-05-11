#include <stdio.h>


#include "testing_argument_parser.hpp"
#include "testing.hpp"

#include "lcs_registry.hpp"


int main0(int argc, const char **argv)
{
	auto *reg = get_global_lcs_registry();
	for (auto kv : reg->Solvers())
	{
		auto name = kv.second.Name();
		printf("Algorithm: %s\n", name.c_str());
	}
}

int main(int argc, const char **argv)
{
	CliArgumentParser args(argc, argv);
	CliOptions opts;

	args.opt_string(opts.a_file, "a_file");
	args.opt_string(opts.b_file, "b_file");

	args.opt_string(opts.algorithm, "algorithm");
	args.opt_string(opts.device, "device");
	args.opt_string(opts.test, "test");

	args.opt_int(opts.a_size, "a_size");
	args.opt_int(opts.b_size, "b_size");
	args.opt_int(opts.a_seed, "a_seed");
	args.opt_int(opts.b_seed, "b_seed");
	args.opt_int(opts.iterations, "iterations");

	args.opt_bool(opts.verbose, "verbose");

	// opts.test = "correctness"; opts.algorithm = "tiled_mt_test";  cli_test_run(opts);

	opts.test = "";
	opts.iterations = 3;
	opts.a_size = 3*32 * 1024;
	opts.b_size = 3*32 * 1024;


	opts.algorithm = "tiled_mt_test"; cli_test_run(opts);
	// opts.algorithm = "tiled_st_test"; cli_test_run(opts);
	// opts.algorithm = "tiled_st_ref"; cli_test_run(opts);


	printf("Main working!\n");
	return 0;
}