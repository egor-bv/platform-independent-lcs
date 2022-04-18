#include <stdio.h>

#include "lcs_types.hpp"
#include "test_utility.hpp"

#include <CL/sycl.hpp>

#include "lcs_reference.hpp"
#include "lcs_validation.hpp"
#include "lcs_antidiagonal.hpp"
#include "lcs_hybrid.hpp"

sycl::queue *LcsContext::CPU_QUEUE = nullptr;
sycl::queue *LcsContext::GPU_QUEUE = nullptr;


#include "dpc_common.hpp"
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
	test_thing(Lcs_Semi_Antidiagonal);
	printf("---\n");
}


int main(int argc, char **argv)
{
	bool use_gpu = false;
	if (argc >= 1 && argv[1])
	{
		if (argv[1][0] == 'g')
		{
			use_gpu = true;
		}
	}
	if(0)
	{
		auto a = RandomBinarySequence(64, 1);
		auto b = RandomBinarySequence(64, 2);
		
		for (int i = 0; i < a.size(); ++i)
		{
			a[i] = i;
		}

		for (int j = 0; j < b.size(); ++j)
		{
			b[j] = -j;
		}

		auto input = LcsInput(a.data(), a.size(), a.data(), a.size());
		auto ctx = LcsContext::Cpu();

		Lcs_Hybrid(input, ctx);


	}
	
	for (int i = 0; i < 5; ++i)
	{
		auto a = RandomBinarySequence(64000, i * 4);
		auto b = RandomBinarySequence(64000, i * 40);

		auto input = LcsInput(a.data(), a.size(), b.data(), b.size());
		auto ctx = use_gpu ? LcsContext::Gpu() : LcsContext::Cpu();

		int64_t total_size = int64_t(input.size_a) * int64_t(input.size_b);
		dpc_common::TimeInterval t;

		Lcs_Hybrid(input, ctx);
		
		double elapsed_ms = t.Elapsed() * 1000.0;
		double cells_per_us = total_size / elapsed_ms / 1000.0f;
		printf("elapsed: %f ms,    speed: %f c/us\n", elapsed_ms, cells_per_us);
	}

	if (0)
	{
		for (int seed = 10; seed < 20; ++seed)
		{
			auto a_pr = RandomBinarySequence(32 * 56, seed);
			auto b_pr = RandomBinarySequence(32 * 56, seed + 6);

			auto input = LcsInput(a_pr.data(), a_pr.size(), b_pr.data(), b_pr.size());
			auto ctx = LcsContext::Cpu();

			CorrectnessTest(input);
		}
	}

	printf("\n=== END ===\n");
	return 0;
}