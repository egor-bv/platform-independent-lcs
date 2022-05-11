#pragma once

#include <CL/sycl.hpp>
#include <inttypes.h>

#include "utility.hpp"

// Canonical type for LCS inputs -- two sequences & their sizes
struct LcsInput
{
	const int *a_data;
	const int *b_data;
	int a_size;
	int b_size;

	LcsInput(const int *a_data, int a_size, const int *b_data, int b_size)
		: a_data(a_data)
		, b_data(b_data)
		, a_size(a_size)
		, b_size(b_size)
	{
	}
};


// Intermediate and final results of LCS computation, shared by different algorithms
struct LcsContext
{
	sycl::queue *queue;

	int *h_strands = 0;
	int *v_strands = 0;

	int h_strands_size = -1;
	int v_strands_size = -1;

	int llcs = -1;

	LcsContext(sycl::queue *q = nullptr)
	{
		queue = q;
	}

	~LcsContext()
	{
		delete[] h_strands;
		delete[] v_strands;
	}
};

// all LCS implementation functions have this signature
typedef void(*LcsFunction)(const LcsInput &input, LcsContext &ctx);