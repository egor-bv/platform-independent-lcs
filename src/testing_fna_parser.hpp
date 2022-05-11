#pragma once
#include "lcs_types.hpp"

// Parses custom file format for sequences
std::vector<int> load_fna_file(std::string filename)
{
	std::ifstream file;
	file.open(filename);
	
	std::string dataset_name;
	int stated_count = 0;
	
	std::vector<int> result;

	if (file.is_open())
	{
		std::getline(file, dataset_name);
		file >> stated_count;

		int symbol = 0;
		for (int i = 0; i < stated_count; ++i)
		{
			file >> symbol;
			result.push_back(symbol);
			file.ignore(); // ignore comma
		}
	}
	else
	{
		printf("failed to load file \"%s\"\n", filename.c_str());
	}

	return result;
}