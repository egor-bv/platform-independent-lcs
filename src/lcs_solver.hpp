#pragma once

#include "lcs_common.hpp"
#include <CL/sycl.hpp>

#include "braid_multiplication.hpp"

// Public interface for LCS algorithms -- stateless
class LcsSolver
{
	LcsProcedure procedure;
	LcsContext context;
public:

	PermutationMatrix operator()(const LcsInput &input)
	{
		context.Reset();
		procedure(input, context);
	}

	PermutationMatrix operator()(const int *a_ptr, int a_size, const int *b_ptr, int b_size)
	{
		context.Reset();
		auto input = LcsInput(a_ptr, a_size, b_ptr, b_size);
		procedure(input, context);
	}

	PermutationMatrix operator()(const std::vector<int> &a, const std::vector<int> &b)
	{
		context.Reset();
		auto input = LcsInput(a.data(), a.size(), b.data(), b.size());
		procedure(input, context);
	}
};
