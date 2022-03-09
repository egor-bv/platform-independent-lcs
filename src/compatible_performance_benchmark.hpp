#pragma once

#include "semi_lcs_api.hpp"

#include "testing_utility.hpp"
#include "stopwatch.hpp"

void RunSingleTest(std::string device, std::string algo_name, std::string file_a, std::string file_b)
{
	assert(device == "cpu" || device == "gpu");
	auto q = GetOrCreateQueue(device);

	auto select_algo = [](std::string name)
	{
		if (name == "ad8")
		{
			return SemiLcs_SubgroupAntidiagonal8;
		}
		else if (name == "ad16")
		{
			return SemiLcs_SubgroupAntidiagonal16;
		}
		else if (name == "tiled")
		{
			return SemiLcs_Tiled;
		}
		else if (name == "tiled_8_4_4_16")
		{
			return SemiLcs_Tiled_Universal<8, 4, 4, 16>;
		}
		else if (name == "tiled_16_4_4_16")
		{
			return SemiLcs_Tiled_Universal<16, 4, 4, 16>;
		}
		else if (name == "tiled_16_8_8_16")
		{
			return SemiLcs_Tiled_Universal<16, 8, 8, 16>;
		}
		else if (name == "tiled_16_16_16_16")
		{
			return SemiLcs_Tiled_Universal<16, 16, 16, 16>;
		}
		else if (name == "tiled_16_8_8_16")
		{
			return SemiLcs_Tiled_Universal<16, 8, 8, 32>;
		}
		else if (name == "tiled_16_8_8_16")
		{
			return SemiLcs_Tiled_Universal<16, 8, 8, 32>;
		}

	};

	auto f = select_algo(algo_name);

	auto warmup_input = ExampleInput(1024, 1024);

	int a_size = 0;
	int b_size = 0;
	int *a = LoadIntArrayFromFna(file_a.c_str(), &a_size);
	int *b = LoadIntArrayFromFna(file_b.c_str(), &b_size);

	auto real_input = InputSequencePair(a, a_size, b, b_size);

	auto warmup_result = f(q, warmup_input);

	Stopwatch sw;
	auto real_result = f(q, real_input);
	sw.stop();

	std::cout << 0 << "\n";
	std::cout << int(sw.elapsed_ms()) << "\n";
	std::cout << real_result.hash() << "\n";
	std::cout << a_size << "\n";
	std::cout << b_size << "\n";
	std::cout << "...\n";
	std::cout << "...\n";
}