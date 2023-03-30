// LCS algorithms perform dynamic programming computations on a grid
// 

#pragma once

#include <CL/sycl.hpp>

#include "lcs_interface_interlal.hpp"
#include "lcs_residual_fixup.hpp"

#include "utility.hpp"




struct GridShapeSimple
{
	int m_given;
	int n_given;

	int m_aligned;
	int n_aligned;

	GridShapeSimple(int _m_given, int _n_given,
					int m_alignment = 1, int n_alignment = 1)
		: m_given(_m_given)
		, n_given(_n_given)
		, m_aligned(AlignedToMultiple(m_given, m_alignment))
		, n_aligned(AlignedToMultiple(n_given, n_alignment))
	{
	}
};

struct GridEmbeddingSimple
{
	GridShapeSimple shape;

	Word *a;
	Word *b;
	Word *h_strands;
	Word *v_strands;

	GridEmbeddingSimple(Word *a_raw, int a_len,
						Word *b_raw, int b_len)
		: shape(a_len, b_len)
	{
		int m = shape.m_aligned;
		int n = shape.n_aligned;

		a = new Word[m];
		b = new Word[n];

		for (int i = 0; i < m; ++i)
		{
			a[i] = a_raw[m - i - 1];
		}
		for (int j = 0; j < n; ++j)
		{
			b[j] = b_raw[j];
		}

		h_strands = new Word[m];
		v_strands = new Word[n];
		for (int i = 0; i < m; ++i)
		{
			h_strands[i] = i;
		}
		for (int j = 0; j < n; ++j)
		{
			v_strands[j] = m + j;
		}

	}

	~GridEmbeddingSimple()
	{
		delete[] a;
		delete[] b;
		delete[] h_strands;
		delete[] v_strands;
	}
};


struct TiledArrayDescriptor
{
	int len_given;
	int tile_size;
	int tile_count;
	int stride;

	TiledArrayDescriptor(int _len_given, int _tile_size, int _block_size)
	{
		len_given = _len_given;
		tile_size = _tile_size;

		tile_count = len_given / tile_size;
		tile_count = tile_count / _block_size * _block_size;
		stride = tile_count;

	}

	int LenLinear() const
	{
		return tile_count * tile_size;
	}

	int LenAllocated() const
	{
		return stride * tile_size;
	}

	int Deinterleave(int interleaved_idx) const
	{
		int mod = interleaved_idx % tile_size;
		int off = interleaved_idx / tile_size;
		int idx = mod * stride + off;
		return idx;
	}

	int Reinterleave(int deinterleaved_idx) const
	{
		int mod = deinterleaved_idx % tile_size;
		int off = deinterleaved_idx / tile_size;
		int idx = mod * tile_size + off;
		return idx;
	}
};

struct GridShapeTiled
{
	int m_given;
	int n_given;

	TiledArrayDescriptor h_desc;
	TiledArrayDescriptor v_desc;

	int remainder_m;
	int remainder_n;

	GridShapeTiled(int _m_given, int _tile_m, int _block_m,
				   int _n_given, int _tile_n, int _block_n)
		: m_given(_m_given)
		, n_given(_n_given)
		, h_desc(_m_given, _tile_m, _block_m)
		, v_desc(_n_given, _tile_n, _block_n)
	{
		remainder_m = m_given - h_desc.LenLinear();
		remainder_n = n_given - v_desc.LenLinear();
	}
};



struct GridEmbeddingTiled
{
	GridShapeTiled shape;

	Word *a;
	Word *b;
	Word *h_strands;
	Word *v_strands;

	GridEmbeddingTiled(const Word *a_raw, int a_len, int tile_m, int block_m,
					   const Word *b_raw, int b_len, int tile_n, int block_n)
		: shape(a_len, tile_m, block_m,
				b_len, tile_n, block_n)
	{
		int m_alloc = shape.h_desc.LenAllocated();
		int n_alloc = shape.v_desc.LenAllocated();;

		int m = shape.h_desc.LenLinear();
		int n = shape.v_desc.LenLinear();

		int i_offset = shape.remainder_m;

		a = new Word[m_alloc];
		b = new Word[n_alloc];

		for (int i = 0; i < m; ++i)
		{
			int i_di = shape.h_desc.Deinterleave(i);
			a[i_di] = a_raw[m - i - 1];
		}
		for (int j = 0; j < n; ++j)
		{
			int j_di = shape.v_desc.Deinterleave(j);
			b[j_di] = b_raw[j];
		}

		h_strands = new Word[m_alloc];
		v_strands = new Word[n_alloc];

		for (int i = 0; i < m; ++i)
		{
			int i_di = shape.h_desc.Deinterleave(i);
			h_strands[i_di] = i_offset + i;
		}
		for (int j = 0; j < n; ++j)
		{
			int j_di = shape.v_desc.Deinterleave(j);
			v_strands[j_di] = i_offset + m + j;
		}
	}

	~GridEmbeddingTiled()
	{
		delete[] a;
		delete[] b;
		delete[] h_strands;
		delete[] v_strands;
	}
};


struct GridShapeGeneral
{
	int m_given;
	int n_given;

	int sub_count_m;
	int sub_count_n;

	TiledArrayDescriptor sub_h_desc;
	TiledArrayDescriptor sub_v_desc;
};



struct GridBuffers
{
	sycl::buffer<Word, 1> a;
	sycl::buffer<Word, 1> b;

	sycl::buffer<Word, 1> h_strands;
	sycl::buffer<Word, 1> v_strands;


	GridBuffers(const GridBuffers &other) = delete;

	GridBuffers(Word *a_data, int a_len,
				Word *b_data, int b_len,
				Word *h_strands_data, int h_strands_len,
				Word *v_strands_data, int v_strands_len)
		: a(a_data, a_len)
		, b(b_data, b_len)
		, h_strands(h_strands_data, h_strands_len)
		, v_strands(v_strands_data, v_strands_len)
	{
	}

};


struct GridAccessors
{
	sycl::accessor<Word, 1, sycl::access_mode::read> a;
	sycl::accessor<Word, 1, sycl::access_mode::read> b;

	sycl::accessor<Word, 1, sycl::access_mode::read_write> h_strands;
	sycl::accessor<Word, 1, sycl::access_mode::read_write> v_strands;

	GridAccessors(GridBuffers &buf, sycl::handler &cgh)
		: a(buf.a, cgh)
		, b(buf.b, cgh)
		, h_strands(buf.h_strands, cgh)
		, v_strands(buf.v_strands, cgh)
	{
	}
};


// Utility functions

GridEmbeddingTiled make_grid_embedding_tiled(LcsProblemContext &ctx, int tile_m, int block_m, int tile_n, int block_n)
{
	return GridEmbeddingTiled(ctx.a_given, ctx.a_length, tile_m, block_m,
							  ctx.b_given, ctx.b_length, tile_n, block_n);
}

GridBuffers make_buffers(GridEmbeddingTiled &g)
{
	return GridBuffers(g.a, g.shape.h_desc.LenAllocated(),
					   g.b, g.shape.v_desc.LenAllocated(),
					   g.h_strands, g.shape.h_desc.LenAllocated(),
					   g.v_strands, g.shape.v_desc.LenAllocated());
}

GridAccessors make_accessors(GridBuffers &buf, sycl::handler &cgh)
{
	return GridAccessors(buf, cgh);
}


// Rectangular subregion in the grid
struct BlockShape
{
	int i0;
	int isize;
	int j0;
	int jsize;
};



struct StripeBlocks
{
	int jsize_total;
	int i0;
	int isize;
	int i_stride;
	int block_count;

	BlockShape block_at(int idx) const
	{
		BlockShape result = {};
		result.i0 = i0 + i_stride * idx;
		result.isize = isize;
		result.j0 = 0;
		result.jsize = jsize_total;
		return result;
	}
};

StripeBlocks divide_tiled_into_stripes(GridShapeTiled &shape, int stripe_height)
{
	int m = shape.h_desc.tile_count;
	int n = shape.v_desc.tile_count;

	StripeBlocks result = {};
	result.i0 = m - stripe_height;
	result.isize = stripe_height;
	result.i_stride = -stripe_height;
	result.jsize_total = n;
	result.block_count = m / stripe_height;

	return result;
}

struct Diagonal
{
	int i_first;
	int j_first;
	int len;
};

struct RectangleBlocks
{
	int block_count_m;
	int block_count_n;

	BlockShape block0;
	int i_stride;
	int j_stride;

	int pass_count() const
	{
		return block_count_m + block_count_n - 1;
	}

	Diagonal diagonal_at(int pass_idx) const
	{
		Diagonal d = {};
		int m = block_count_m;
		int n = block_count_n;
		d.i_first = pass_idx < m ? (m - pass_idx - 1) : 0;
		d.j_first = pass_idx < m ? 0 : (pass_idx - m + 1);
		d.len = Min(m - d.i_first, n - d.j_first);
		return d;
	}

	BlockShape block_at(Diagonal d, int step) const
	{
		int block_i = d.i_first + step;
		int block_j = d.j_first + step;

		BlockShape result = block0;
		result.i0 += i_stride * block_i;
		result.j0 += j_stride * block_j;

		return result;
	}
};

RectangleBlocks divide_tiled_into_blocks(GridShapeTiled &shape, int block_m, int block_n)
{
	int m = shape.h_desc.tile_count;
	int n = shape.v_desc.tile_count;

	RectangleBlocks result = {};
	result.block_count_m = m / block_m;
	result.block_count_n = n / block_n;

	BlockShape block0 = {};
	block0.i0 = 0;
	block0.isize = block_m;

	block0.j0 = 0;
	block0.jsize = block_n;

	result.i_stride = block_m;
	result.j_stride = block_n;

	result.block0 = block0;
	return result;
}

void prepare_symbols(LcsProblemContext &ctx)
{
	int m = ctx.a_length;
	int n = ctx.b_length;

	ctx.a_prepared = new Word[m];
	ctx.b_prepared = new Word[n];

	for (int i = 0; i < m; ++i)
	{
		ctx.a_prepared[i] = ctx.a_given[m - i - 1];
	}
	for (int j = 0; j < n; ++j)
	{
		ctx.b_prepared[j] = ctx.b_given[j];
	}
}

void copy_strands_and_fixup_tiled(LcsProblemContext &ctx, GridEmbeddingTiled &grid)
{
	int m_given = grid.shape.h_desc.len_given;
	int n_given = grid.shape.v_desc.len_given;

	ctx.h_strands_length = m_given;
	ctx.h_strands = new Word[m_given];

	ctx.v_strands_length = n_given;
	ctx.v_strands = new Word[n_given];

	prepare_symbols(ctx);
	int i_offset = grid.shape.remainder_m;

	int m_done = grid.shape.h_desc.LenLinear();
	int n_done = grid.shape.v_desc.LenLinear();

	for (int i = 0; i < i_offset; ++i)
	{
		ctx.h_strands[i] = i;
	}

	for (int i = 0; i < m_done; ++i)
	{
		int i_di = grid.shape.h_desc.Deinterleave(i);
		ctx.h_strands[i + i_offset] = grid.h_strands[i_di];
	}

	for (int j = 0; j < n_done; ++j)
	{
		int j_di = grid.shape.v_desc.Deinterleave(j);
		ctx.v_strands[j] = grid.v_strands[j_di];
	}

	for (int j = 0; j < grid.shape.remainder_n; ++j)
	{
		ctx.v_strands[j + n_done] = m_given + n_done + j;
	}

	Lcs_Semi_Fixup(ctx, 0, grid.shape.remainder_m, 0, n_done);
	Lcs_Semi_Fixup(ctx, 0, m_given, n_done, grid.shape.remainder_n);

}