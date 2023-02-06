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


#include <string>
#include <fstream>

struct TextFileData
{
	std::string text;

	TextFileData(const char *filename)
	{
		std::ifstream file(filename);
		text = std::string((std::istreambuf_iterator<char>(file)),
						   (std::istreambuf_iterator<char>()));
	}
};