#if 0

#include <stdio.h>

#include "lcs_types.hpp"
#include "test_utility.hpp"

#include <CL/sycl.hpp>

#include "lcs_reference.hpp"
#include "lcs_validation.hpp"

#include "lcs_tiled.hpp"

sycl::queue *LcsContext::CPU_QUEUE = nullptr;
sycl::queue *LcsContext::GPU_QUEUE = nullptr;


#if 0
#include "dpc_common.hpp"
void steady_ant_test()
{
	for (int i = 1; i < 11; ++i)
	{
		int a_size = 202500;
		int b_size = 202300;

		auto a = RandomBinarySequence(a_size, i);
		auto b = RandomBinarySequence(b_size, i * 1000);


		auto input_full = LcsInput(a.data(), a.size(), b.data(), b.size());
		auto ctx_full = LcsContext::Cpu();
		Lcs_Semi_Antidiagonal(input_full, ctx_full);

		//auto input00 = LcsInput(a.data(), a.size() / 2, b.data(), b.size());
		//auto input01 = LcsInput(a.data() + a.size() / 2, a.size() / 2, b.data(), b.size());

		int splitter = a_size / 2;
		int a_half_size0 = splitter;
		int a_half_size1 = a_size - a_half_size0;

		dpc_common::TimeInterval t_big_stuff;

		auto input00 = LcsInput(a.data(), a_half_size0, b.data(), b.size());
		auto input01 = LcsInput(a.data() + a_half_size0, a_half_size1, b.data(), b.size());

		auto ctx00 = LcsContext::Cpu();
		auto ctx01 = LcsContext::Cpu();

		Lcs_Semi_Antidiagonal(input00, ctx00);
		Lcs_Semi_Antidiagonal(input01, ctx01);

		double big_stuff_seconds = t_big_stuff.Elapsed();

		dpc_common::TimeInterval t_fixing;
		// Make permutation matrices...
		auto p00 = PermutationMatrix::FromStrands(ctx00.h_strands, ctx00.m, ctx00.v_strands, ctx00.n);
		auto p01 = PermutationMatrix::FromStrands(ctx01.h_strands, ctx01.m, ctx01.v_strands, ctx01.n);

		auto p_full = PermutationMatrix::FromStrands(ctx_full.h_strands, ctx_full.m, ctx_full.v_strands, ctx_full.n);

		auto prod = staggered_multiply<true>(p00, p01, b_size);

		double fixing_seconds = t_fixing.Elapsed();
		double sim = prod.Similarity(p_full);

		printf("Similarity = %f, big_stuff = %f ms, fixing = %f ms\n", sim, big_stuff_seconds * 1000.0, fixing_seconds * 1000.0);
	}
	printf("done something...\n");
}


void steady_ant_test_2(LcsInput &input, bool horizontal)
{
	int splitter_horz = input.size_a / 2;
	int splitter_vert = input.size_b / 2;

	int a_size0 = horizontal ? splitter_horz : input.size_a;
	int a_size1 = horizontal ? input.size_a - splitter_horz : input.size_a;
	int b_size0 = horizontal ? input.size_b : splitter_vert;
	int b_size1 = horizontal ? input.size_b : input.size_b - splitter_vert;

	int a_off1 = horizontal ? splitter_horz : 0;
	int b_off1 = horizontal ? 0 : splitter_vert;

	dpc_common::TimeInterval t_full;
	auto ctx_full = LcsContext::Cpu();
	Lcs_Semi_Antidiagonal(input, ctx_full);
	auto p_full = PermutationMatrix::FromStrands(ctx_full.h_strands, ctx_full.m, ctx_full.v_strands, ctx_full.n);
	double full_ms = t_full.Elapsed() * 1000.0;



	dpc_common::TimeInterval t_total;

	// split into two subproblems
	auto input0 = LcsInput(input.seq_a, a_size0, input.seq_b, b_size0);
	auto input1 = LcsInput(input.seq_a + a_off1, a_size1, input.seq_b + b_off1, b_size1);

	auto ctx0 = LcsContext::Cpu();
	auto ctx1 = LcsContext::Cpu();

	Lcs_Semi_Antidiagonal(input0, ctx0);
	Lcs_Semi_Antidiagonal(input1, ctx1);

	auto p0 = PermutationMatrix::FromStrands(ctx0.h_strands, ctx0.m, ctx0.v_strands, ctx0.n);
	auto p1 = PermutationMatrix::FromStrands(ctx1.h_strands, ctx1.m, ctx1.v_strands, ctx1.n);
	
	// combine subresults
	dpc_common::TimeInterval t_multiplication;
	int k_horz = input.size_b;
	int k_vert = input.size_a;
	auto prod = horizontal ? staggered_multiply<true>(p0, p1, k_horz) : staggered_multiply<false>(p0, p1, k_vert);
	

	double multiplication_ms = t_multiplication.Elapsed() * 1000.0;
	double total_ms = t_total.Elapsed() * 1000.0;
	double similarity = p_full.Similarity(prod);
	// report results

	printf("taken: %f ms total, %f ms multiplying; %f correct; mult: %f%%\n", total_ms, multiplication_ms, similarity, multiplication_ms / total_ms * 100.0);

}
#endif

template<typename F>
void benchmark(F f, LcsInput &input, LcsContext &ctx)
{
	ctx.Reset();
	dpc_common::TimeInterval t;
	f(input, ctx);
	uint64_t total_size = uint64_t(input.size_a) * uint64_t(input.size_b);
	double elapsed_ms = t.Elapsed() * 1000.0;
	double cells_per_us = total_size / elapsed_ms / 1000.0f;
	printf("elapsed: %f ms,    speed: %f c/us\n", elapsed_ms, cells_per_us);
}


void CorrectnessTest(LcsInput &input)
{
	auto ref_ctx = LcsContext();
	Lcs_Semi_Reference(input, ref_ctx);

	auto test_thing = [&](auto f)
	{
		auto ctx = LcsContext::Cpu();
		f(input, ctx);
		auto result = ValidateLcsContext(ctx, &ref_ctx);
		if (result.equals_reference)
		{
			printf("ok!\n");
		}
		else
		{
			double unique_percent = double(result.unique_strand_indices) / result.total_strands;
			auto exact = result.unique_strand_indices == result.total_strands ? " (exact)" : "";
			printf("no! first_invalid_index: %u, percent: %.3f%s\n", result.first_invalid_strand_index, unique_percent * 100.0, exact);
		}
	};
	printf("size: %d x %d\n", input.size_a, input.size_b);
	// test_thing(Lcs_Semi_Antidiagonal);
	printf("---\n");
}

#include "test_spec.hpp"
#include "test_argument_parser.hpp"

int main(int argc, char **argv)
{
	// parsing arguments
	CliArguments args(argc, (const char **)argv);
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



	// predefined tests

	// RunPerformanceBenchmark("stripes", Lcs_Semi_Stripes);
	RunCorrectnessTests("tiled", Lcs_Tiled_1);
	// RunPerformanceBenchmark("tiled2", Lcs_Tiled_2);
	// RunPerformanceBenchmark("tiled1", Lcs_Tiled_1);
	
	//auto a_ex = RandomBinarySequence(70245, 1);
	//auto b_ex = RandomBinarySequence(80667, 2);
	//auto ex = LcsInput(a_ex.data(), a_ex.size(), b_ex.data(), b_ex.size());
	//steady_ant_test_2(ex, false);
	//steady_ant_test_2(ex, true);

	// custom tests


	return 0;
}

#endif