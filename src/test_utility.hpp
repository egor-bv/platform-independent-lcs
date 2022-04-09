#pragma once

#include <vector>
#include <random>

#include "lcs_types.hpp"


std::vector<int> RandomBinarySequence(int size, int seed)
{
	std::vector<int> result(size);
	std::mt19937 rng(seed);

	for (int i = 0; i < size; ++i)
	{
		result[i] = rng() & 1;
	}

	return result;
}

std::vector<int> RandomSequence(int size, int range, int seed)
{
	std::vector<int> result(size);
	std::mt19937 rng(seed);

	for (int i = 0; i < size; ++i)
	{
		result[i] = rng() % range;
	}

	return result;
}

