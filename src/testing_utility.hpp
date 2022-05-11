#pragma once

#include <vector>
#include <random>

std::vector<int> generate_random_binary_sequence(int size, int seed)
{
	std::vector<int> result(size);
	std::mt19937 rng(seed);

	for (int i = 0; i < size; ++i)
	{
		result[i] = rng() & 1;
	}

	return result;
}

std::vector<int> generate_random_uniform_sequence(int size, int range, int seed)
{
	std::vector<int> result(size);
	std::mt19937 rng(seed);

	for (int i = 0; i < size; ++i)
	{
		result[i] = rng() % range;
	}

	return result;
}

