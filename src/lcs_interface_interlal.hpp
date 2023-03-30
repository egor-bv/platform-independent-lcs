// Common interface to be implemented by various LCS solver variations.
// For internal use of the library only

#pragma once

#include <CL/sycl.hpp>
#include <inttypes.h>

#include "permutation.hpp"


// All sequences are to be converted to sequences of 32 bit words
// this size is best supported by both CPU and GPU hardware
// sequences with small alphabets and 16-bit strands may be implemented on top
#if 1
struct Word
{
	union 
	{
		int value; // NOTE: we'd prefer to work with unsigned only, but code generation suffers without this
		uint32_t uvalue;
	};

	Word() : value(0x0f0f0f0f) {}
	Word(int x) : value(x) {}

	bool operator==(const Word &other) const { return value == other.value; }
	bool operator>(const Word &other) const { return value > other.value; }
};
#else
using Word = uint32_t;
#endif

// Bundles input sequences, additional input parameters, intermediate and final results
struct LcsProblemContext
{
	// Fields filled in by the caller
	const Word *a_given;
	const Word *b_given;
	int a_length;
	int b_length;

	sycl::queue *queue;

	// Intermediate state managed by solvers
	Word *a_prepared; // NOTE: reversed
	Word *b_prepared;
	


	Word *h_strands;
	Word *v_strands;
	int h_strands_length;
	int v_strands_length;

	// Final results
	int length_of_lcs = -1;
	PermutationMatrix permutation_result;

	~LcsProblemContext()
	{
		delete[] h_strands;
		delete[] v_strands;
		delete[] a_prepared;
		delete[] b_prepared;
	}
};


// All LCS implementation functions should use this signature
typedef void LcsProcedure(LcsProblemContext &ctx);

