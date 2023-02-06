#pragma once

#include <CL/sycl.hpp>
#include "lcs_common.hpp"


// Arrays are laid out as-is with sequence a reversed
template<typename Symbol, typename Strand>
class GridEmbeddingSimple
{
public:
	Symbol *a_data;
	Symbol *b_data;
	Strand *h_strands;
	Strand *v_strands;

	int m;
	int n;

	GridEmbeddingSimple(const Symbol *a_raw, int a_len, const Symbol *b_raw, int b_len)
	{
		m = a_len;
		n = b_len;

		a_data = new Symbol[m];
		b_data = new Symbol[n];

		for (int i = 0; i < m; ++i)
		{
			a_data[i] = a_raw[m - i - 1];
		}

		for (int j = 0; j < n; ++j)
		{
			b_data[j] = b_raw[j];
		}

		h_strands = new Strand[m];
		v_strands = new Strand[n];

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
		delete[] a_data;
		delete[] b_data;
		delete[] h_strands;
		delete[] v_strands;
	}
};


// Defines how indices are transformed in (de)interleaving
// where:
// Interleaved   ~ ABCDABCDABCDABCD
// Deinterleaved ~ AAAABBBBCCCCDDDD
// also same as transposing a 2d array with some padding
template<int TileSize>
class DeinterleavedArrayDescriptor
{
	int len_original;
	int len_allocated;
	int stride;

public:

	DeinterleavedArrayDescriptor(int len)
	{
		len_original = len;
		int num_tiles = SmallestMultipleToFit(len_original, TileSize);
		// want each segment to be aligned to cacheline boundary
		// but since we don't know element size here & TileSize is small, just align to 64 elements
		constexpr int cacheline_elements = 64;
		stride = SmallestMultipleToFit(num_tiles, cacheline_elements) * cacheline_elements;
		len_allocated = stride * TileSize;
	}


	int Len()
	{
		return len_original;
	}

	int LenAllocated()
	{
		return len_allocated;
	}

	int Stride()
	{
		return stride;
	}

	int RegionUsefulSize()
	{
		return CeilDiv(len_original, TileSize);
	}
	
	int IncompleteTileLimit()
	{
		int result = len_original % TileSize;
		if (result == 0)
		{
			result = TileSize;
		}
		return result;
	}


	int Deinterleave(int interleaved_idx)
	{
		int mod = interleaved_idx % TileSize;
		int off = interleaved_idx / TileSize;
		int idx = mod * stride + off;
		return idx;
	}

	int Interleave(int deinterleaved_idx)
	{
		int mod = deinterleaved_idx % stride;
		int off = deinterleaved_idx / stride;
		int idx = mod * TileSize + off;
		return idx;
	}

	bool Inside(int deinterleaved_idx)
	{
		int mod = deinterleaved_idx % stride;
	}
};


// Arrays are laid out in deinterleaved fashion with TileM x TileN sized tiles
template<typename Symbol, typename Strand, int TileM, int TileN>
class GridEmbeddingDeinterleaved
{
public:
	Symbol *a_data;
	Symbol *b_data;
	Strand *h_strands;
	Strand *v_strands;

	DeinterleavedArrayDescriptor<TileM> h_desc;
	DeinterleavedArrayDescriptor<TileN> v_desc;

	GridEmbeddingDeinterleaved(const Symbol *a_raw, int a_len, const Symbol *b_raw, int b_len)
		: h_desc(a_len)
		, v_desc(b_len)
	{
		int h_len = h_desc.LenAllocated();
		int v_len = v_desc.LenAllocated();

		a_data = new Symbol[h_len];
		b_data = new Symbol[v_len];

		for (int i = 0; i < h_desc.Len(); ++i)
		{
			int i_di = h_desc.Deinterleave(i);
			a_data[i_di] = a_raw[h_desc.Len() - i - 1];
		}

		for (int j = 0; j < v_desc.Len(); ++j)
		{
			int j_di = v_desc.Deinterleave(j);
			b_data[j_di] = b_raw[j];
		}

		h_strands = new Strand[h_len];
		v_strands = new Strand[v_len];

		for (int i = 0; i < h_desc.Len(); ++i)
		{
			int i_di = h_desc.Deinterleave(i);
			h_strands[i_di] = i;
		}

		for (int j = 0; j < v_desc.Len(); ++j)
		{
			int j_di = v_desc.Deinterleave(j);
			v_strands[j_di] = j + h_desc.Len();
		}
	}

	~GridEmbeddingDeinterleaved()
	{
		delete[] a_data;
		delete[] b_data;
		delete[] h_strands;
		delete[] v_strands;
	}
};


// Structure holding sycl buffers to avoid repetition
// Doesn't know anything about in-buffer layout
template<typename Symbol, typename Strand>
class GridBuffers
{
public:
	sycl::buffer<Symbol, 1> a;
	sycl::buffer<Symbol, 1> b;
	sycl::buffer<Strand, 1> h_strands;
	sycl::buffer<Strand, 1> v_strands;

	GridBuffers(const GridBuffers<Symbol, Strand> &other) = delete;

	GridBuffers(Symbol *a_data, int a_len, Symbol *b_data, int b_len,
		Strand *h_strands_data, int h_strands_len, Strand *v_strands_data, int v_strands_len)
		: a(a_data, a_len)
		, b(b_data, b_len)
		, h_strands(h_strands_data, h_strands_len)
		, v_strands(v_strands_data, v_strands_len)
	{
	}

	static GridBuffers FromSimple(GridEmbeddingSimple<Symbol, Strand> &g)
	{
		return GridBuffers(
			g.a_data, g.m,
			g.b_data, g.n,
			g.h_strands, g.m,
			g.v_strands, g.n
		);
	}

	template<int TileM, int TileN>
	static GridBuffers FromDeinterleaved(GridEmbeddingDeinterleaved<Symbol, Strand, TileM, TileN> &g)
	{
		return GridBuffers(
			g.a_data, g.h_desc.LenAllocated(),
			g.b_data, g.v_desc.LenAllocated(),
			g.h_strands, g.h_desc.LenAllocated(),
			g.v_strands, g.v_desc.LenAllocated()
		);
	}
};


// Structure holding sycl accessors to avoid repetition
// Doesn't know anything about in-buffer layout
template<typename Symbol, typename Strand>
class GridAccessors
{
public:
	sycl::accessor<Symbol, 1, sycl::access_mode::read> a;
	sycl::accessor<Symbol, 1, sycl::access_mode::read> b;
	sycl::accessor<Strand, 1, sycl::access_mode::read_write> h_strands;
	sycl::accessor<Strand, 1, sycl::access_mode::read_write> v_strands;

	GridAccessors(GridBuffers<Symbol, Strand> &g, sycl::handler &cgh)
		: a(g.a, cgh)
		, b(g.b, cgh)
		, h_strands(g.h_strands, cgh)
		, v_strands(g.v_strands, cgh)
	{
	}
};



GridEmbeddingSimple<int, int>
make_embedding(const LcsInput &input)
{
	return GridEmbeddingSimple<int, int>(
		input.a_data, input.a_size, input.b_data, input.b_size);
}


template<int TILE_M, int TILE_N>
GridEmbeddingDeinterleaved<int, int, TILE_M, TILE_N> 
make_embedding_deinterleaved(const LcsInput &input)
{
	return GridEmbeddingDeinterleaved<int, int, TILE_M, TILE_N>(
		input.a_data, input.a_size,
		input.b_data, input.b_size);
}

GridBuffers<int, int>
make_buffers(GridEmbeddingSimple<int, int> &grid)
{
	return GridBuffers<int, int>::FromSimple(grid);
}

template<int TILE_M, int TILE_N>
GridBuffers<int, int> 
make_buffers(GridEmbeddingDeinterleaved<int, int, TILE_M, TILE_N> &grid)
{
	return GridBuffers<int, int>::FromDeinterleaved(grid);
}

GridAccessors<int, int> make_accessors(GridBuffers<int, int> &buf, sycl::handler &cgh)
{
	return GridAccessors<int, int>(buf, cgh);
}


struct AntdiagonalDescriptor
{
	int diag_len;
	int i_first;
	int j_first;
};


AntdiagonalDescriptor antidiag_at(int diag_idx, int m, int n)
{
	AntdiagonalDescriptor d = {};
	d.i_first = diag_idx < m ? (m - diag_idx - 1) : 0;
	d.j_first = diag_idx < m ? 0 : (diag_idx - m + 1);
	d.diag_len = Min(m - d.i_first, n - d.j_first);
	return d;
}


void copy_strands(LcsContext &ctx_dst, GridEmbeddingSimple<int, int> &grid_src)
{
	ctx_dst.h_strands_size = grid_src.m;
	ctx_dst.v_strands_size = grid_src.n;
	ctx_dst.h_strands = new int[grid_src.m];
	ctx_dst.v_strands = new int[grid_src.n];

	for (int i = 0; i < grid_src.m; ++i)
	{
		ctx_dst.h_strands[i] = grid_src.h_strands[i];
	}
	for (int j = 0; j < grid_src.n; ++j)
	{
		ctx_dst.v_strands[j] = grid_src.v_strands[j];
	}
}

template<int TILE_M, int TILE_N>
void 
copy_strands_deinterleaved(LcsContext &ctx_dst, GridEmbeddingDeinterleaved<int, int, TILE_M, TILE_N> &grid_src)
{
	int m = grid_src.h_desc.Len();
	int n = grid_src.v_desc.Len();
	ctx_dst.h_strands_size = m;
	ctx_dst.v_strands_size = n;
	ctx_dst.h_strands = new int[m];
	ctx_dst.v_strands = new int[n];

	for (int i = 0; i < m; ++i)
	{
		int i_di = grid_src.h_desc.Deinterleave(i);
		ctx_dst.h_strands[i] = grid_src.h_strands[i_di];
	}
	for (int j = 0; j < n; ++j)
	{
		int j_di = grid_src.v_desc.Deinterleave(j);
		ctx_dst.v_strands[j] = grid_src.v_strands[j_di];
	}
}


void make_symbols(int **a_dst, int **b_dst, const LcsInput &input)
{
	int *a = new int[input.a_size];
	int *b = new int[input.b_size];
	for (int i = 0; i < input.a_size; ++i)
	{
		a[i] = input.a_data[input.a_size - i - 1];
	}
	for (int j = 0; j < input.b_size; ++j)
	{
		b[j] = input.b_data[j];
	}

	*a_dst = a;
	*b_dst = b;
}

void make_strands(int **h_strands_dst, int m, int **v_strands_dst, int n)
{
	int *h_strands = new int[m];
	int *v_strands = new int[n];
	for (int i = 0; i < m; ++i)
	{
		h_strands[i] = i;
	}
	for (int j = 0; j < n; ++j)
	{
		v_strands[j] = m + j;
	}

	*h_strands_dst = h_strands;
	*v_strands_dst = v_strands;
}

using int_Buffer = sycl::buffer<int, 1>;

using int_SymbolsAccessor = sycl::accessor<int, 1, sycl::access_mode::read>;
using int_StrandsAccessor = sycl::accessor<int, 1, sycl::access_mode::read_write>;

int_Buffer make_buffer(int *data, int count)
{
	return int_Buffer(data, count);
}

int_SymbolsAccessor make_symbols_accessor(int_Buffer &buf, sycl::handler &cgh)
{
	return int_SymbolsAccessor(buf, cgh);
}

int_StrandsAccessor make_strands_accessor(int_Buffer &buf, sycl::handler &cgh)
{
	return int_StrandsAccessor(buf, cgh);
}