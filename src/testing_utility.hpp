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

void short_form_report_header()
{
	std::cout
		<< std::setw(16) << "(algo_name)" << "|"
		<< std::setw(16) << "(mn)" << "|"
		<< std::setw(16) << "(time_ms)" << "|"
		<< std::setw(16) << "(elements/us)"
		;
}

void short_form_report_results(const char *algo_name, int64_t mn, double time_ms)
{
	double elements_per_us = mn / 1000.0f / time_ms;

	std::cout
		<< std::setw(16) << algo_name << "|"
		<< std::setw(16) << mn << "|"
		<< std::setw(16) << time_ms << "|"
		<< std::setw(16) << elements_per_us
		;
}

struct ResultCsvWriter
{
	std::ofstream fout;

	ResultCsvWriter(std::string filename = "")
	{
		std::string when = std::to_string(std::time(nullptr));
		fout.open(filename + when + ".csv", std::ofstream::out);

		fout.setstate(std::ios::left);
		fout
			<< std::setw(16) << "session_name" << "|"
			<< std::setw(16) << "algo_name" << "|"
			<< std::setw(16) << "a_name" << "|"
			<< std::setw(16) << "b_name" << "|"
			<< std::setw(9) << "a_length" << "|"
			<< std::setw(9) << "b_length" << "|"
			<< std::setw(8) << "time_ms" << "|"
			<< std::setw(20) << "result_hash" << "\n"
			;
	}

	~ResultCsvWriter()
	{
		fout.close();
	}

	void AppendResult(const char *session_name, const char *algo_name, const char *a_name, const char *b_name,
		int a_length, int b_length, double time_ms, int64_t result_hash)
	{
		fout
			<< std::setw(16) << session_name << "|"
			<< std::setw(16) << algo_name << "|"
			<< std::setw(16) << a_name << "|"
			<< std::setw(16) << b_name << "|"
			<< std::setw(9) << a_length << "|"
			<< std::setw(9) << b_length << "|"
			<< std::setw(8) << (int)time_ms << "|"
			<< std::setw(20) << result_hash << "\n"
			;
	}

};