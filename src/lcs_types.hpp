#pragma once

#include <CL/sycl.hpp>
#include <inttypes.h>

// view into input sequences to pass to LCS algorithm
struct LcsInput
{
	int *seq_a; // not owned
	int *seq_b; // not owned
	int size_a;
	int size_b;

	LcsInput(int *seq_a, int size_a, int *seq_b, int size_b)
		: seq_a(seq_a), seq_b(seq_b), size_a(size_a), size_b(size_b)
	{
	}
};


// intermediate and final results of LCS computation, shared by different algorithms
struct LcsContext
{
	// TODO: find a way to deal with different types here
	using Symbol = uint32_t;
	using Index = uint32_t;

	sycl::queue *queue;
	
	uint32_t m;
	uint32_t n;

	// want to keep this unsigned
	Symbol *a;
	Symbol *b;


	// how do we handle different strand sizes?
	Index *h_strands;
	Index *v_strands;

	bool is_prefix = false;
	bool is_binary_prefix = false;

	// prefix-specific data
	Index *diag0;
	Index *diag1;
	Index *diag2;
	uint32_t diag_len;

	uint32_t llcs;

	LcsContext() = default;

	// construct with a CPU queue in place
	static LcsContext Cpu()
	{
		if (!CPU_QUEUE)
		{
			CPU_QUEUE = new sycl::queue(sycl::cpu_selector());
		}
		auto result = LcsContext();
		result.queue = CPU_QUEUE;
		return result;
	}

	Index num_h_strands()
	{
		return m;
	}

	Index num_v_strands()
	{
		return n;
	}

	void Reset()
	{
		m = 0;
		n = 0;
		a = nullptr;
		b = nullptr;
		delete[] h_strands; h_strands = nullptr;
		delete[] v_strands; v_strands = nullptr;
		delete[] diag0; diag0 = nullptr;
		delete[] diag1; diag1 = nullptr;
		delete[] diag2; diag2 = nullptr;
	}

	static sycl::queue *CPU_QUEUE;
	static sycl::queue *GPU_QUEUE;
};


// all LCS implementation functions have this signature
typedef void(*LcsProcedure)(LcsInput &input, LcsContext &ctx);