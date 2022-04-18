#pragma once
// common types and functions used by LCS implementations

#include "lcs_types.hpp"
#include "utility.hpp"

// copy input sequences, reversing the first one
void InitInputs(LcsInput &input, LcsContext &ctx)
{
	Assert(!ctx.a);
	Assert(!ctx.b);

	ctx.m = input.size_a;
	ctx.a = new LcsContext::Symbol[ctx.m];
	for (int i = 0; i < ctx.m; ++i)
	{
		ctx.a[i] = input.seq_a[ctx.m - i - 1];
	}

	ctx.n = input.size_b;
	ctx.b = new LcsContext::Symbol[ctx.n];
	for (int j = 0; j < ctx.n; ++j)
	{
		ctx.b[j] = input.seq_b[j];
	}
}


// init strand indices, must be called after InitInputs()
void InitStrands(LcsContext &ctx)
{
	Assert(!ctx.h_strands);
	Assert(!ctx.v_strands);

	ctx.is_prefix = false;
	ctx.h_strands = new LcsContext::Index[ctx.num_h_strands()];
	for (int i = 0; i < ctx.num_h_strands(); ++i)
	{
		ctx.h_strands[i] = i;
	}

	ctx.v_strands = new LcsContext::Index[ctx.num_v_strands()];
	for (int j = 0; j < ctx.num_v_strands(); ++j)
	{
		ctx.v_strands[j] = ctx.num_h_strands() + j;
	}
}

template<int TILE_M, int TILE_N>
void InitDeinterleaved(LcsInput &input, LcsContext &ctx)
{
	Assert(!ctx.a);
	Assert(!ctx.b);
	Assert(!ctx.h_strands);
	Assert(!ctx.v_strands);

	int m = input.size_a;
	int n = input.size_b;
}



void InitInputsPrefix(LcsInput &input, LcsContext &ctx)
{
	Assert(!ctx.a);
	Assert(!ctx.b);

	ctx.m = input.size_a;
	ctx.a = new LcsContext::Symbol[ctx.m];
	for (int i = 0; i < ctx.m; ++i)
	{
		ctx.a[i] = input.seq_a[ctx.m - i - 1];
	}

	ctx.n = input.size_b;
	ctx.b = new LcsContext::Symbol[ctx.n];
	for (int j = 0; j < ctx.n; ++j)
	{
		ctx.b[j] = input.seq_b[j];
	}

	if (ctx.m > ctx.n)
	{
		Swap(ctx.m, ctx.n);
		Swap(ctx.a, ctx.b);
	}
}



void InitDiagsPrefix(LcsContext &ctx)
{
	Assert(!ctx.diag0);
	Assert(!ctx.diag1);
	Assert(!ctx.diag2);

	ctx.is_prefix = true;
	LcsContext::Index diag_max_len = Min(ctx.n, ctx.m) + 1;

	ctx.diag_len = diag_max_len;
	// init diagonals to zero
	ctx.diag0 = new LcsContext::Index[diag_max_len]{};
	ctx.diag1 = new LcsContext::Index[diag_max_len]{};
	ctx.diag2 = new LcsContext::Index[diag_max_len]{};
}


// binary
void InitInputsPrefixBinary(LcsInput &input, LcsContext &ctx)
{
	static_assert(std::is_same<LcsContext::Index, uint32_t>::value, "Bad type");
	static_assert(std::is_same<LcsContext::Index, uint32_t>::value, "Bad type");

	Assert(!ctx.a);
	Assert(!ctx.b);

	Assert(IsMultipleOf(input.size_a, 32));
	Assert(IsMultipleOf(input.size_b, 32));

	ctx.m = input.size_a / 32;
	ctx.a = new LcsContext::Symbol[ctx.m];

	ctx.n = input.size_b / 32;
	ctx.b = new LcsContext::Symbol[ctx.n];

	for (int i = 0; i < ctx.m; ++i)
	{
		uint32_t packed = 0;
		for (int step = 0; step < 32; ++step)
		{
			uint32_t src = input.seq_a[input.size_a - 1 - (32 * i + step)];
			Assert(src == 0 || src == 1);

			// packed = (packed << 1) | src;
			packed = packed | (src << step);
		}
		ctx.a[i] = packed;
	}

	for (int j = 0; j < ctx.n; ++j)
	{
		uint32_t packed = 0;
		for (int step = 0; step < 32; ++step)
		{
			uint32_t src = input.seq_b[32 * j + step];
			Assert(src == 0 || src == 1);
			packed = packed | (src << step);
			// packed = (packed << 1) | src;
		}
		ctx.b[j] = packed;
	}
}

void InitStrandsPrefixBinary(LcsContext &ctx)
{
	Assert(!ctx.h_strands);
	Assert(!ctx.v_strands);

	ctx.is_prefix = true;
	ctx.h_strands = new LcsContext::Index[ctx.num_h_strands()];
	for (int i = 0; i < ctx.num_h_strands(); ++i)
	{
		ctx.h_strands[i] = -1;
	}

	ctx.v_strands = new LcsContext::Index[ctx.num_v_strands()];
	for (int j = 0; j < ctx.num_v_strands(); ++j)
	{
		ctx.v_strands[j] = 0;
	}
}

