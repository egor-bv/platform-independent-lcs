#pragma once
#include <fstream>
#include <iostream>
#include <string>

int *LoadIntArrayFromFna(const char *filename, int *array_size)
{
	int *result = nullptr;
	std::ifstream file;
	file.open(filename);

	std::string dataset_name;
	int total_count = 0;
	int number = 0;
	if (file.is_open())
	{
		std::getline(file, dataset_name);
		file >> total_count;
		std::cout << total_count << " ";
		*array_size = total_count;
		result = new int[total_count];
		for (int i = 0; i < total_count; ++i)
		{
			file >> number;
			result[i] = number;
			file.ignore();
		}
	}
	else
	{
		std::cout << "Failed to load file " << filename << "\n";
	}

	return result;
}

