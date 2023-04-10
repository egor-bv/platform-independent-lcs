#pragma once
#include "lcs_grid.hpp"

#include <vector>

struct GridBuffersSeparate 
{
	using Buffer = sycl::buffer<Word, 1>;

	Buffer a;
	Buffer b;

	Buffer v_strands_in;
	Buffer v_strands_out;

	std::vector<Buffer> h_strands_inout;

	GridBuffersSeparate(Word *a_data, int a_len,
						Word *b_data, int b_len,
						Word *h_strands_data, int h_strands_len,
						Word *v_strands_data, int v_strands_len,
						Word *v_strands_out,
						int num_sections)
		: a(a_data, a_len)
		, b(b_data, b_len)
		, v_strands_in(v_strands_data, v_strands_len)
		, v_strands_out(v_strands_out, v_strands_len)
	{
		int section_len = h_strands_len / num_sections;
		for (int section_idx = 0; section_idx < num_sections; ++section_idx)
		{
			int offset = section_idx * section_len;
			// printf("Creating buffer subrange from %d to %d\n", offset, offset + section_len);
			h_strands_inout.push_back(Buffer(h_strands_data + offset, section_len));
		}
	}

	void Swap()
	{
		::Swap(v_strands_in, v_strands_out);
	}
};

GridBuffersSeparate make_buffers_separate(GridEmbeddingSimple &g, int num_sections)
{
	g.v_strands_out = new Word[g.shape.n_aligned]{};
	return GridBuffersSeparate(g.a, g.shape.m_aligned,
							   g.b, g.shape.n_aligned,
							   g.h_strands, g.shape.m_aligned,
							   g.v_strands, g.shape.n_aligned,
							   g.v_strands_out,
							   num_sections);
}