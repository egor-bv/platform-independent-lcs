#include <stdio.h>

#include "lcs_types.hpp"
#include "test_utility.hpp"

#include <CL/sycl.hpp>

#include "lcs_reference.hpp"
#include "lcs_validation.hpp"
#include "lcs_antidiagonal.hpp"


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


int main(int argc, char **atgv)
{

	for (int seed = 10; seed < 20; ++seed)
	{
		auto a_pr = RandomBinarySequence(32*1000, seed);
		auto b_pr = RandomBinarySequence(32*1000, seed + 6);
		
		auto input = LcsInput(a_pr.data(), a_pr.size(), b_pr.data(), b_pr.size());
		auto ctx = LcsContext::Cpu();

		printf("=======\n");
		benchmark(Lcs_Prefix_Reference, input, ctx);
		benchmark(Lcs_Prefix_Binary_Reference, input, ctx);
		benchmark(Lcs_Semi_Antidiagonal, input, ctx);
		benchmark(Lcs_Semi_Antidiagonal, input, ctx);
		
		{
			auto result = ValidateLcsContext(ctx);
			double unique_percent = double(result.unique_strand_indices) / result.total_strands;
			auto exact = result.unique_strand_indices == result.total_strands ? " (exact)" : "";
			printf("first_invalid_index: %u, percent: %.3f%s\n", result.first_invalid_strand_index, unique_percent * 100.0, exact);
		}
	}


	printf("\n=== END ===\n");
	return 0;
}