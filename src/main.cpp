#include <chrono>
#include <cmath>
#include <iostream>

#include "CL/sycl.hpp"

#include "dpc_common.hpp"


#include "lcs.hpp"
#include "semi_local.hpp"
#include "combing.hpp"

#include "stopwatch.hpp"
#include "file_utility.hpp"




int main(int argc, char **argv)
{
	// NOTE(Egor): for test purposes, hardcoded filenames here
	const char *file_a = "1.fna";
	const char *file_b = "2.fna";

	// loading code
	int dataset_a_size = 0;
	int *dataset_a = LoadIntArrayFromFna(file_a, &dataset_a_size);

	int dataset_b_size = 0;
	int *dataset_b = LoadIntArrayFromFna(file_b, &dataset_b_size);

	std::cout << "\n-----------------------------------------------------------------------------\n";

	InputSequencePair input(dataset_a, dataset_a_size, dataset_b, dataset_b_size);

	// InputSequencePair input = ExampleInput(11, 19);
	// InputSequencePair input = ExampleInput(17, 17);
	// InputSequencePair input = ExampleInput(4, 5);
	// InputSequencePair input = ExampleInput(30002, 40001);
	// InputSequencePair input = ExampleInput(3*30002, 3*40001);

	if (1)
	{
		if (1)
		{
			Stopwatch sw;

			// long long hsh_my = StickyBraidSequentialHash(input.a, input.b, input.length_a, input.length_b);
			long long hsh_my = StickyBraidSimplest(input);
			long long hsh_his = 0;// StickyBraidSimplest(input);
			// long long hsh_his = sticky_braid_sequential(input.a, input.length_a, input.b, input.length_b);
			sw.stop();

			auto elements_per_ms = (double)input.length_a * input.length_b / sw.elapsed_ms();
			std::cout << "\n===============================================================\n";
			std::cout << "semi-local finished!" << "\n";
			std::cout << "time taken  = " << sw.elapsed_ms() << "ms" << "\n";
			// TODO: set precision better
			std::cout << "elements/us = " << elements_per_ms / 1000.0 << "\n";
			std::cout << "lcs hash (my)  = " << hsh_my << "\n";
			std::cout << "lcs hash (his) = " << hsh_his << "\n";
		}

		if (1)
		{
			try
			{
				auto sel = sycl::cpu_selector();
				sycl::queue q(sel, dpc_common::exception_handler);

				auto warmup = StickyBraidParallelBlockwise(q, input);
				Stopwatch sw;
				long long hsh = StickyBraidParallelBlockwise(q, input);
				sw.stop();
				auto elements_per_ms = (double)input.length_a * input.length_b / sw.elapsed_ms();
				// long long hsh = StickyBraidSycl(q, input);
				std::cout << "\n===============================================================\n";
				std::cout << "parallel algo finished" << "\n";
				std::cout << "time taken = " << sw.elapsed_ms() << "ms" << "\n";
				std::cout << "elements/us = " << elements_per_ms / 1000.0 << "\n";
				std::cout << "new hash   = " << hsh  << "\n";
			}
			catch (sycl::exception e)
			{
				std::cout << "SYCL exception caught: " << e.what() << "\n";
				return 1;
			}
		}
	}


}

