#pragma once
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>

struct InputSequencePair
{
	const int *a;
	const int *b;
	const int length_a;
	const int length_b;
	InputSequencePair(int *a, int length_a, int *b, int length_b) :
		a(a), b(b), length_a(length_a), length_b(length_b)
	{
	}
};

InputSequencePair ExampleInput(int m, int n, int seed = 1)
{
	int *a = new int[m];
	int *b = new int[n];

	std::mt19937 rng(seed);
	for (int i = 0; i < m; ++i) a[i] = rng() % 2;
	for (int j = 0; j < n; ++j) b[j] = rng() % 2;

	return InputSequencePair(a, m, b, n);
}


struct ResultCsvWriter
{
	std::ofstream fout;

	ResultCsvWriter(std::string filename = "")
	{
		std::string when = std::to_string(std::time(nullptr));
		fout.open(filename + when + ".csv", std::ofstream::out);

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