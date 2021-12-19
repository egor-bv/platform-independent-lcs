#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>

#include "CL/sycl.hpp"
#include "device_selector.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"
#include "lcs.hpp"

#include "semi_local.hpp"

class Stopwatch
{
private:
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point latest_time;
	bool stopped;

	void record_latest()
	{
		latest_time = std::chrono::high_resolution_clock::now();
	}

public:
	Stopwatch()
	{
		restart();
	}

	void restart()
	{
		stopped = false;
		start_time = std::chrono::high_resolution_clock::now();
	}

	void stop()
	{
		record_latest();
		stopped = true;
	}

	double elapsed_ms()
	{
		if (!stopped)
		{
			record_latest();
		}
		auto duration = latest_time - start_time;
		return std::chrono::duration<double, std::milli>(duration).count();
	}

};

void print_ms(const char *text, double ms)
{
	std::cout << text << ms << std::endl;
}



struct InputPair
{
	const int *a;
	const int *b;
	const int len_a;
	const int len_b;

	InputPair(int *a, int *b, int len_a, int len_b) :
		a(a), b(b), len_a(len_a), len_b(len_b)
	{
	}
};




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
		std::cout << dataset_name;
		std::cout << total_count;
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

	// for compatibility reasons!
	// result[*array_size - 1] = -1;
	return result;
}


int *LoadExample(int *size)
{
	*size = 2;
	int *result = new int[*size];
	result[0] = 111;
	result[1] = 888;
	return result;
}

int LcsRowMajor(int *a, int m, int *b, int n, int *prev_row, int *curr_row)
{
	for (int i = 0; i < n; ++i)
	{
		prev_row[i] = 0;
		curr_row[i] = 0;
	}
	for (int i = 1; i < m; ++i)
	{
		int L = 0;
		for (int j = 1; j < n; ++j)
		{
			curr_row[j] = std::max(
				std::max(prev_row[j], L),
				(a[i - 1] == b[j - 1]) ? prev_row[j - 1] + 1 : prev_row[j - 1]
			);
			L = curr_row[j];
		}
		std::swap(prev_row, curr_row);
	}
	return prev_row[n - 1];
}

int LcsAntidiagonal(int *a, int m, int *b, int n)
{
	assert(m >= n);
	int full_diag_len = n + 1;
	int *d1 = new int[full_diag_len] {};
	int *d2 = new int[full_diag_len] {};
	int *d3 = new int[full_diag_len] {};

	// NOTE(Egor): i here does not correspond to column index, but to diagonal index
	for (int i = 0; i < m + n - 1; ++i)
	{
		int begin_j = (i < m) ? 1 : (2 + i - m);
		int end_j = (i >= n - 1) ? (n + 1) : (i + 2);
		for (int j = begin_j; j < end_j; ++j)
		{
			int e_n = d2[j - 1];
			int e_w = d2[j];
			int e_nw = d1[j - 1] + (int)(a[i - j + 1] == b[j - 1]);

			d3[j] = std::max(e_nw, std::max(e_n, e_w));
		}
		std::swap(d1, d2);
		std::swap(d2, d3);
	}
	int result = d2[full_diag_len - 1];
	delete[] d1;
	delete[] d2;
	delete[] d3;
	return result;
}



struct SequencePair
{
	std::vector<int> seq_a;
	std::vector<int> seq_b;

	void init_random_binary(int size_a, int size_b)
	{
		for (int i = 0; i < size_a; ++i) seq_a.push_back(rand() % 2);
		for (int i = 0; i < size_b; ++i) seq_b.push_back(rand() % 2);
	}
};


// Reference implementation by Mishin
int prefix_lcs_sequential(int *a, int a_size, int *b, int b_size)
{

	int *input_a;
	int *input_b;
	int m, n;

	if (a > b) {
		m = a_size + 1;
		n = b_size + 1;
		input_a = a;
		input_b = b;
	}
	else {
		n = a_size + 1;
		m = b_size + 1;
		input_b = a;
		input_a = b;
	}

	auto prev_row = new int[n];
	auto cur_row = new int[n];
	for (int i = 0; i < n; ++i) {
		cur_row[i] = 0;
		prev_row[i] = 0;
	}

	for (int i = 1; i < m; ++i) {
		auto l = 0;
		for (int j = 1; j < n; ++j) {
			cur_row[j] = std::max(
				std::max(prev_row[j], l),
				(input_a[i - 1] == input_b[j - 1]) ? prev_row[j - 1] + 1 : prev_row[j - 1]
			);
			l = cur_row[j];

		}
		std::swap(prev_row, cur_row);
	}

	return prev_row[n - 1];
}


static void ReportTime(const std::string &msg, sycl::event e)
{
	cl_ulong time_start =
		e.get_profiling_info<sycl::info::event_profiling::command_start>();

	cl_ulong time_end =
		e.get_profiling_info<sycl::info::event_profiling::command_end>();

	double elapsed = (time_end - time_start) / 1e6;
	std::cout << msg << elapsed << " milliseconds\n";
}



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

	//if (dataset_a_size < dataset_b_size)
	//{
	//	std::swap(dataset_a_size, dataset_b_size);
	//	std::swap(dataset_a, dataset_b);
	//}

	InputSequencePair input(dataset_a, dataset_a_size, dataset_b, dataset_b_size);

	// semilocal hash!
	if (1)
	{
		{
			Stopwatch sw;

			// long long hsh_my = StickyBraidSequentialHash(input.a, input.b, input.length_a, input.length_b);
			long long hsh_my = StickyBraidAntidiagonal(input);
			long long hsh_his = 0;// StickyBraidSimplest(input);
			// long long hsh_his = sticky_braid_sequential(input.a, input.length_a, input.b, input.length_b);
			sw.stop();

			std::cout << "\n===============================================================\n";
			std::cout << "semi-local finished!" << "\n";
			std::cout << "time taken = " << sw.elapsed_ms() << "ms" << "\n";
			std::cout << "lcs hash (my)  = " << hsh_my << "\n";
			std::cout << "lcs hash (his) = " << hsh_his << "\n";
		}

		if (1)
		{
			try
			{
				auto sel = sycl::cpu_selector();
				sycl::queue q(sel, dpc_common::exception_handler);

				Stopwatch sw;
				long long hsh = StickyBraidParallelBlockwise(q, input);

				std::cout << "\n===============================================================\n";
				std::cout << "parallel algo finished" << "\n";
				std::cout << "time taken = " << sw.elapsed_ms() << "ms" << "\n";
				std::cout << "new hash   = " << hsh << "\n";
			}
			catch (sycl::exception e)
			{
				std::cout << "SYCL exception caught: " << e.what() << "\n";
				return 1;
			}
		}
	}


	if (0)
	{
		// now ready for testing!
		PrefixLcsSequential lcs_seq;

		Stopwatch sw;
		lcs_seq.Prepare(input);
		int score = lcs_seq.Run(input);
		sw.stop();

		std::cout << "\n===============================================================\n";
		std::cout << "sequential algo finished" << "\n";
		std::cout << "time taken = " << sw.elapsed_ms() << "ms" << "\n";
		std::cout << "lcs score  = " << score << "\n";
	}

	//if (0)
	//{
	//	try
	//	{
	//		auto sel = sycl::cpu_selector();
	//		sycl::queue q(sel, dpc_common::exception_handler);

	//		PrefixLcsParallel lcs_par;
	//		Stopwatch sw;
	//		lcs_par.Prepare(input);
	//		int score = lcs_par.RunNaive(q, input);

	//		std::cout << "\n===============================================================\n";
	//		std::cout << "parallel algo finished" << "\n";
	//		std::cout << "time taken = " << sw.elapsed_ms() << "ms" << "\n";
	//		std::cout << "lcs score  = " << score << "\n";
	//	}
	//	catch (sycl::exception e)
	//	{
	//		std::cout << "SYCL exception caught: " << e.what() << "\n";
	//		return 1;
	//	}
	//}
}

//int old_main(int argc, char **argv)
//{
//	// NOTE(Egor): for test purposes, hardcoded filenames here
//	const char *file_a = "a.fna";
//	const char *file_b = "b.fna";
//
//	// loading code
//	int dataset_a_size = 0;
//	int *dataset_a = LoadIntArrayFromFna("a.fna", &dataset_a_size);
//
//	int dataset_b_size = 0;
//	int *dataset_b = LoadIntArrayFromFna("b.fna", &dataset_b_size);
//
//	if (dataset_a_size < dataset_b_size)
//	{
//		std::swap(dataset_a_size, dataset_b_size);
//		std::swap(dataset_a, dataset_b);
//	}
//
//	int m = dataset_a_size + 1;
//	int n = dataset_b_size + 1;
//
//	// create temporary data store for the algorithm
//	int *prev_row = new int[n];
//	int *curr_row = new int[n];
//
//	int *diagonal1 = new int[n] {};
//	int *diagonal2 = new int[n] {};
//	int *diagonal3 = new int[n] {};
//
//	// cpu-based execution
//	if (1)
//	{
//		Stopwatch serial_sw;
//
//		// int lcs_score = prefix_lcs_sequential(dataset_a, dataset_a_size, dataset_b, dataset_b_size);
//		int lcs_score = LcsAntidiagonal(dataset_a, m - 1, dataset_b, n - 1);
//
//		auto elapsed_ms = serial_sw.elapsed_ms();
//		std::cout << "\nLCS score = " << lcs_score << "\n";
//		std::cout << "ms elapsed = " << elapsed_ms << "\n";
//		std::cout << "iterations = " << (m + n - 1) << "\n";
//
//	}
//
//
//	// TODO(Egor): use default cpu selector instead
//	sycl::cpu_selector sel;
//	sycl::event e1;
//
//	// Wrap main SYCL API calls into a try/catch to diagnose potential errors
//	if (1)
//	{
//		auto begin_stamp = std::chrono::high_resolution_clock::now();
//		try
//		{
//			// Create a command queue using the device selector and request profiling
//			auto prop_list = sycl::property_list{ };
//			sycl::queue q(sel, dpc_common::exception_handler, prop_list);
//
//			sycl::buffer<int, 1> buf_a(dataset_a, sycl::range<1>(dataset_a_size));
//			sycl::buffer<int, 1> buf_b(dataset_b, sycl::range<1>(dataset_b_size));
//
//			sycl::buffer<int, 1> buf_d1(diagonal1, sycl::range<1>(n));
//			sycl::buffer<int, 1> buf_d2(diagonal2, sycl::range<1>(n));
//			sycl::buffer<int, 1> buf_d3(diagonal3, sycl::range<1>(n));
//
//			std::cout << "Submitting first kernel..." << std::endl;
//			begin_stamp = std::chrono::high_resolution_clock::now();
//			int KERNEL_EXEC_COUNT = m + n - 1;
//			for (int i = 0; i < KERNEL_EXEC_COUNT; ++i)
//			{
//				if (i == 1) begin_stamp = std::chrono::high_resolution_clock::now();
//				int begin_j = (i < m) ? 1 : (2 + i - m);
//				int end_j = (i >= n - 1) ? (n + 1) : (i + 2);
//
//				if (i > 0)
//				{
//					std::swap(buf_d1, buf_d2);
//					std::swap(buf_d2, buf_d3);
//				}
//
//				Stopwatch sw;
//				q.submit(
//					[&](auto &h)
//					{
//						auto acc_a = buf_a.get_access<sycl_read, sycl_global_buffer>(h);
//						auto acc_b = buf_b.get_access<sycl_read, sycl_global_buffer>(h);
//
//						auto acc_d1 = buf_d1.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
//						auto acc_d2 = buf_d2.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
//						auto acc_d3 = buf_d3.get_access<sycl::access::mode::write, sycl_global_buffer>(h);
//
//
//						h.parallel_for(sycl::range<1>(end_j - begin_j),
//							[=](auto j_iter)
//							{
//								/*
//								int j = j_iter + begin_j;
//								int e_n = acc_d2[j - 1];
//								int e_w = acc_d2[j];
//								int e_nw = acc_d1[j - 1] + (int)(acc_a[i - j + 1] == acc_b[j - 1]);
//								acc_d3[j] = std::max(e_nw, std::max(e_n, e_w));
//								*/
//							}
//						);
//					});
//				// std::cout << sw.elapsed_ms() << std::endl; 
//			}
//
//			{
//
//			}
//
//
//			//q.wait_and_throw();
//			std::cout << "Waiting for execution to complete...\n";
//
//			auto elapsed = std::chrono::high_resolution_clock::now() - begin_stamp;
//			auto elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
//
//			std::cout << "Before sync\n";
//			std::cout << elapsed_ms << "ms\n";
//		}
//
//		catch (sycl::exception e) {
//			std::cout << "SYCL exception caught: " << e.what() << "\n";
//			return 1;
//		}
//
//		auto elapsed = std::chrono::high_resolution_clock::now() - begin_stamp;
//		auto elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
//
//		std::cout << "Execution completed\n";
//		std::cout << elapsed_ms << "ms\n";
//
//		// Print results
//		std::cout << "1:" << diagonal1[n - 1] << std::endl;
//		std::cout << "2:" << diagonal2[n - 1] << std::endl;
//		std::cout << "3:" << diagonal2[n - 1] << std::endl;
//	}
//
//
//	return 0;
//}
