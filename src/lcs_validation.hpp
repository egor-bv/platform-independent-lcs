#pragma once

#include <map>
#include <inttypes.h>
#include "lcs_types.hpp"

// describes in what's wrong with output
struct LcsValidationResult
{
	uint32_t total_strands = 0;
	uint32_t unique_strand_indices = 0;

	// first index that falls that's >= total_strands, otherwise 0
	uint32_t first_invalid_strand_index = 0;
	uint32_t invalid_strand_index_count = 0;

	bool equals_reference = false;
};


LcsValidationResult ValidateLcsContext(LcsContext &ctx, LcsContext *ref = nullptr)
{
	LcsValidationResult result;
	result.total_strands = ctx.num_h_strands() + ctx.num_v_strands();
	if (ref)
	{
		if (ctx.num_h_strands() == ref->num_h_strands() && ctx.num_v_strands() == ref->num_v_strands())
		{
			bool equals = true;
			for (int i = 0; i < ctx.num_h_strands(); ++i)
			{
				if (ctx.h_strands[i] != ref->h_strands[i])
				{
					equals = false;
					break;
				}
			}

			for (int j = 0; j < ctx.num_v_strands(); ++j)
			{
				if (ctx.v_strands[j] != ref->v_strands[j])
				{
					equals = false;
					break;
				}
			}

			result.equals_reference = equals;
		}
		else
		{
			result.equals_reference = false;
		}
	}

	std::map<uint32_t, int> unique_indices;


	auto validate_strand = [&](uint32_t strand)
	{
		if (strand >= result.total_strands)
		{
			++result.invalid_strand_index_count;
			if (result.first_invalid_strand_index == 0)
			{
				result.first_invalid_strand_index = strand;
			}
		}

		if (unique_indices.count(strand) == 0)
		{
			unique_indices[strand] = 1;
			++result.unique_strand_indices;
		}
		else
		{
			++unique_indices[strand];
		}
	};

	for (int i = 0; i < ctx.num_h_strands(); ++i)
	{
		auto strand = ctx.h_strands[i];
		validate_strand(strand);
	}

	for (int j = 0; j < ctx.num_v_strands(); ++j)
	{
		auto strand = ctx.v_strands[j];
		validate_strand(strand);
	}

	return result;
}