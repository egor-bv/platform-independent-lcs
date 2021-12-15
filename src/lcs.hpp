#pragma once
#include <algorithm>
#include <assert.h>

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

class PrefixLcsSolver
{
protected:
	// temporary storage
	int *d1 = nullptr;
	int *d2 = nullptr;
	int *d3 = nullptr;
	int diagonal_length = 0;

	void Free()
	{
		diagonal_length = 0;
		delete[] d1;
		delete[] d2;
		delete[] d3;
	}

	void Alloc(int _diagonal_length)
	{
		diagonal_length = _diagonal_length;
		d1 = new int[diagonal_length] {};
		d2 = new int[diagonal_length] {};
		d3 = new int[diagonal_length] {};
	}

	void PrepareStorage(const InputSequencePair &pair)
	{
		int length = std::min(pair.length_a, pair.length_b) + 1;
		if (diagonal_length != length)
		{
			Free();
			Alloc(length);
		}
	}

public:
	PrefixLcsSolver()
	{
	}

	~PrefixLcsSolver()
	{
		if (diagonal_length) Free();
	}



};

class PrefixLcsSequential : public PrefixLcsSolver
{
public:
	void Prepare(const InputSequencePair &p)
	{
		PrepareStorage(p);
	}

	int Run(const InputSequencePair &p)
	{
		const int *a = p.a;
		const int *b = p.b;
		int m = p.length_a;
		int n = p.length_b;
		if (m < n)
		{
			std::swap(a, b);
			std::swap(m, n);
		}
		assert(m >= n);

		int full_diag_len = n + 1;
		assert(diagonal_length = full_diag_len);
		for (int i = 0; i < (m + n - 1); ++i)
		{
			int begin_j = (i < m) ? 1 : (2 + i - m);
			int end_j = (i >= n - 1) ? (n + 1) : (i + 2);
			for (int j = begin_j; j < end_j; ++j)
			{
				// TOOD(Egor): fix weird warnings here (reading invalid data)
				int e_n = d2[j - 1];
				int e_w = d2[j];
				int e_nw = d1[j - 1] + (int)(a[i - j + 1] == b[j - 1]);

				d3[j] = std::max(e_nw, std::max(e_n, e_w));
			}
			std::swap(d1, d2);
			std::swap(d2, d3);
		}
		int result = d2[full_diag_len - 1];
		return result;
	}
};

#include "CL/sycl.hpp"

// Few useful acronyms.
constexpr auto sycl_read = sycl::access::mode::read;
constexpr auto sycl_write = sycl::access::mode::write;
constexpr auto sycl_global_buffer = sycl::access::target::global_buffer;

#if 0
class PrefixLcsParallel : public PrefixLcsSolver
{
	using IntBuffer = sycl::buffer<int, 1>;

	// NOTE(Egor): can't pull these here, they don't have default constructors
	//IntBuffer buf_a;
	//IntBuffer buf_b;

	//IntBuffer buf_d1;
	//IntBuffer buf_d2;
	//IntBuffer buf_d3;
public:

	void Warmup()
	{
		// TODO(Egor): so you can run kernel once before measuring time
	}

	void Prepare(const InputSequencePair &p)
	{
		PrepareStorage(p);
	}

	int RunNaive(sycl::queue &q, const InputSequencePair &p)
	{
		IntBuffer buf_a = IntBuffer(p.a, p.length_a);
		IntBuffer buf_b = IntBuffer(p.b, p.length_b);

		IntBuffer buf_d1 = IntBuffer(d1, diagonal_length);
		IntBuffer buf_d2 = IntBuffer(d2, diagonal_length);
		IntBuffer buf_d3 = IntBuffer(d3, diagonal_length);

		int m = p.length_a;
		int n = p.length_b;
		if (m < n)
		{
			std::swap(buf_a, buf_b);
			std::swap(m, n);
		}
		assert(m >= n);

		int full_diag_len = n + 1;
		assert(diagonal_length = full_diag_len);

		const int kernel_exec_count = m + n - 1;
		for (int i = 0; i < kernel_exec_count; ++i)
		{
			int begin_j = (i < m) ? 1 : (2 + i - m);
			int end_j = (i >= n - 1) ? (n + 1) : (i + 2);
			q.submit(
				[&](auto &h)
				{
					auto acc_a = buf_a.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
					auto acc_b = buf_b.get_access<sycl::access::mode::read, sycl_global_buffer>(h);

					auto acc_d1 = buf_d1.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
					auto acc_d2 = buf_d2.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
					auto acc_d3 = buf_d3.get_access<sycl::access::mode::write, sycl_global_buffer>(h);


					h.parallel_for(sycl::range<1>(end_j - begin_j),
						[=](auto j_iter)
						{
							int j = j_iter + begin_j;
							int e_n = acc_d2[j - 1];
							int e_w = acc_d2[j];
							int e_nw = acc_d1[j - 1] + (int)(acc_a[i - j + 1] == acc_b[j - 1]);
							acc_d3[j] = std::max(e_nw, std::max(e_n, e_w));

						}
					);
				});
			std::swap(buf_d1, buf_d2);
			std::swap(buf_d2, buf_d3);
		}
		q.wait_and_throw();
		// FIXME(Egor): Hack! I swap buffers, but can't tell where actual data should be?
		// maybe just get pointer from buffer directly?
		int result = std::max(d1[n - 1], std::max(d2[n-1], d3[n-1]));
		return result;
	}

	//int RunGrouped(sycl::queue &q, const InputSequencePair &p)
	//{
	//	IntBuffer buf_a = IntBuffer(p.a, p.length_a);
	//	IntBuffer buf_b = IntBuffer(p.b, p.length_b);

	//	IntBuffer buf_d1 = IntBuffer(d1, diagonal_length);
	//	IntBuffer buf_d2 = IntBuffer(d2, diagonal_length);
	//	IntBuffer buf_d3 = IntBuffer(d3, diagonal_length);

	//	int m = p.length_a;
	//	int n = p.length_b;
	//	if (m < n)
	//	{
	//		std::swap(buf_a, buf_b);
	//		std::swap(m, n);
	//	}
	//	assert(m >= n);

	//	int full_diag_len = n + 1;
	//	assert(diagonal_length = full_diag_len);

	//	// Everything's the same so far...
	//	// Now compute how to split work into workgroups
	//	// And also use local memory instead of global!

	//	const int vert_size = 16; // == workgroup size
	//	const int horz_size = 1000;

	//	const int diagonal_count = m + n - 1;
	//	const int kernel_exec_count = (diagonal_count/horz_size) + 1;

	//	for (int i_big = 0; i_big < kernel_exec_count; ++i_big)
	//	{
	//		// TODO(Egor): Complete this code so that it at least works at all
	//		int begin_j = (i < m) ? 1 : (2 + i - m);
	//		int end_j = (i >= n - 1) ? (n + 1) : (i + 2);
	//		q.submit(
	//			[&](auto &h)
	//			{
	//				auto acc_a = buf_a.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
	//				auto acc_b = buf_b.get_access<sycl::access::mode::read, sycl_global_buffer>(h);

	//				auto acc_d1 = buf_d1.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
	//				auto acc_d2 = buf_d2.get_access<sycl::access::mode::read, sycl_global_buffer>(h);
	//				auto acc_d3 = buf_d3.get_access<sycl::access::mode::write, sycl_global_buffer>(h);

	//				sycl::accessor<int, 1>(sycl::range<1>(vert_size)

	//				int workgroup_count = ...;
	//				int total_size = vert_size * workgroup_count;
	//				h.parallel_for(sycl::nd_range<1>(total_size, vert_size),
	//					[=](auto j_iter)
	//					{
	//						int j = j_iter + begin_j;
	//						int e_n = acc_d2[j - 1];
	//						int e_w = acc_d2[j];
	//						int e_nw = acc_d1[j - 1] + (int)(acc_a[i - j + 1] == acc_b[j - 1]);
	//						acc_d3[j] = std::max(e_nw, std::max(e_n, e_w));

	//					}
	//				);
	//			});
	//		std::swap(buf_d1, buf_d2);
	//		std::swap(buf_d2, buf_d3);
	//	}
	//	q.wait_and_throw();
	//	// FIXME(Egor): Hack! I swap buffers, but can't tell where actual data should be?
	//	// maybe just get pointer from buffer directly?
	//	int result = std::max(d1[n - 1], std::max(d2[n - 1], d3[n - 1]));
	//	return result;
	//}
};

#endif