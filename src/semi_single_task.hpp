#pragma once
#include <CL/sycl.hpp>

struct VirtualDiagonal
{
	int i_first;
	int j_first;

	void IndicesAt(int idx, int &i, int &j)
	{
		i = i_first + idx;
		j = j_first + idx;
	}
};

struct VirtualGrid
{
	int m;
	int n;

	int DiagonalCount()
	{
		return m + n;
	}

	VirtualDiagonal DiagonalAt(int idx)
	{
		int i_first = 0;
		int j_first = 0;
		return { i_first, j_first };
	}
};




void StickyBraidComb_Task(sycl::queue &q, const int *_a_rev, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a_rev(_a_rev, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	q.submit([&](auto h)
		{
			auto a_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			h.single_task([=]()
				{
					for ...;
					for ...;
					for ...;
				}
			);
		}
	);
}