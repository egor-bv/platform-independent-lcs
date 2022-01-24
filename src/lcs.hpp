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


