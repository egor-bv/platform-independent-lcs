#pragma once

#include "lcs_types.hpp"
#include "lcs_common.hpp"
#include "lcs_grid.hpp"


#include "permutation.hpp"
#include "braid_multiplication.hpp"

// NOTE: needs to be simple POD to pass to kernel code easily
template<int TILE_M, int TILE_N>
struct GridEmbeddingLayout
{
	int m_given;
	int n_given;

	int num_subproblems_m;
	int num_subproblems_n;

	int max_sub_m;
	int max_sub_n;
	int clipped_sub_m;
	int clipped_sub_n;


	int stride_h;
	int stride_v;

	int h_len_alloc;
	int v_len_alloc;
	int a_len_alloc;
	int b_len_alloc;

	DeinterleavedArrayDescriptor<TILE_M> sub_h_desc;
	DeinterleavedArrayDescriptor<TILE_N> sub_v_desc;

	int num_tiles_m(int size_m) const
	{
		return CeilDiv(size_m, TILE_M);
	}

	int num_tiles_n(int size_n) const
	{
		return CeilDiv(size_n, TILE_N);
	}


	void Init(int a_len, int b_len,
			  int _num_subproblems_m, int _num_subproblems_n)
	{
		m_given = a_len;
		n_given = b_len;

		num_subproblems_m = _num_subproblems_m;
		num_subproblems_n = _num_subproblems_n;

		max_sub_m = CeilDiv(m_given, num_subproblems_m);
		max_sub_n = CeilDiv(n_given, num_subproblems_n);

		clipped_sub_m = m_given - max_sub_m * (num_subproblems_m - 1);
		clipped_sub_n = n_given - max_sub_n * (num_subproblems_n - 1);

		sub_h_desc = DeinterleavedArrayDescriptor<TILE_M>(max_sub_m);
		sub_v_desc = DeinterleavedArrayDescriptor<TILE_N>(max_sub_n);

		// NOTE: size per subproblem is already cacheline-aligned
		stride_h = sub_h_desc.LenAllocated();
		stride_v = sub_v_desc.LenAllocated();

		// NOTE: this is also memory overhead factor
		int replication = num_subproblems_m * num_subproblems_n;
		h_len_alloc = stride_h * replication;
		v_len_alloc = stride_v * replication;

		a_len_alloc = stride_h * num_subproblems_m;
		b_len_alloc = stride_v * num_subproblems_n;
	}

	int h_offset(int i_sub, int j_sub) const
	{
		return i_sub * stride_h + j_sub * num_subproblems_m * stride_h;
	}
	int v_offset(int i_sub, int j_sub) const
	{
		return i_sub * stride_v + j_sub * num_subproblems_m * stride_v;
	}

	int a_offset(int i_sub, int j_sub) const
	{
		return i_sub * stride_h;
	}

	int b_offset(int i_sub, int j_sub) const
	{
		return j_sub * stride_v;
	}
};


template<int TILE_M, int TILE_N, typename Symbol, typename Strand>
struct GridEmbeddingGeneral
{
	Symbol *a_data;
	Symbol *b_data;
	Strand *h_strands;
	Strand *v_strands;

	GridEmbeddingLayout<TILE_M, TILE_N> gl;


	int h_size_combined(int i_sub_first, int i_sub_count)
	{
		Assert(i_sub_first + i_sub_count < gl.num_subproblems_m);
		int end = i_sub_first + i_sub_count;
		if (end == gl.num_subproblems_m)
		{
			return gl.max_sub_m * (i_sub_count - 1) + gl.clipped_sub_m;
		}
		else
		{
			return gl.max_sub_m * (i_sub_count);
		}
	}

	int v_size_combined(int j_sub_first, int j_sub_count)
	{
		Assert(j_sub_first + j_sub_count < gl.num_subproblems_n);
		int end = j_sub_first + j_sub_count;
		if (end == gl.num_subproblems_n)
		{
			return gl.max_sub_n * (j_sub_count - 1) + gl.clipped_sub_n;
		}
		else
		{
			return gl.max_sub_n * (j_sub_count);
		}
	}



	GridEmbeddingGeneral(const Symbol *a_raw, int a_len,
						 const Symbol *b_raw, int b_len,
						 int _num_subproblems_m, int _num_subproblems_n)
	{
		gl.Init(a_len, b_len, _num_subproblems_m, _num_subproblems_n);

		a_data = new Symbol[gl.a_len_alloc];
		b_data = new Symbol[gl.b_len_alloc];

		for (int i = 0; i < gl.m_given; ++i)
		{
			int i_sub = i / gl.max_sub_m;
			int i_local = i % gl.max_sub_m;
			int i_di = gl.sub_h_desc.Deinterleave(i_local);
			a_data[gl.stride_h * i_sub + i_di] = a_raw[gl.m_given - i - 1];
		}
		for (int j = 0; j < gl.n_given; ++j)
		{
			int j_sub = j / gl.max_sub_n;
			int j_local = j % gl.max_sub_n;
			int j_di = gl.sub_v_desc.Deinterleave(j_local);
			b_data[gl.stride_v * j_sub + j_di] = b_raw[j];
		}

		h_strands = new Strand[gl.h_len_alloc];
		v_strands = new Strand[gl.v_len_alloc];

		for (int i_sub = 0; i_sub < gl.num_subproblems_m; ++i_sub)
		{
			for (int j_sub = 0; j_sub < gl.num_subproblems_n; ++j_sub)
			{
				int actual_sub_m = i_sub == gl.num_subproblems_m - 1 ? gl.clipped_sub_m : gl.max_sub_m;
				int actual_sub_n = j_sub == gl.num_subproblems_n - 1 ? gl.clipped_sub_n : gl.max_sub_n;

				for (int i = 0; i < actual_sub_m; ++i)
				{
					int i_di = gl.sub_h_desc.Deinterleave(i);
					h_strands[gl.h_offset(i_sub, j_sub) + i_di] = i;

				}
				for (int j = 0; j < actual_sub_n; ++j)
				{
					int j_di = gl.sub_v_desc.Deinterleave(j);
					v_strands[gl.v_offset(i_sub, j_sub) + j_di] = j + actual_sub_m;
				}
			}
		}

	}

	~GridEmbeddingGeneral()
	{
		delete[] a_data;
		delete[] b_data;
		delete[] h_strands;
		delete[] v_strands;
	}
};

template<int TILE_M, int TILE_N>
GridEmbeddingGeneral<TILE_M, TILE_N, int, int>
make_embedding_general(const LcsInput &input,
					   int subproblem_count_m,
					   int subproblem_count_n)
{
	return
		GridEmbeddingGeneral<TILE_M, TILE_N, int, int>(input.a_data, input.a_size,
													   input.b_data, input.b_size,
													   subproblem_count_m, subproblem_count_n);
}


template<int TILE_M, int TILE_N>
GridBuffers<int, int>
make_buffers(GridEmbeddingGeneral<TILE_M, TILE_N, int, int> &grid)
{
	return GridBuffers<int, int>(grid.a_data, grid.gl.a_len_alloc,
								 grid.b_data, grid.gl.b_len_alloc,
								 grid.h_strands, grid.gl.h_len_alloc,
								 grid.v_strands, grid.gl.v_len_alloc);
}

template<int TILE_M, int TILE_N, typename Symbols, typename Strands>
inline void
load_cache_with_offsets(TiledCache<TILE_M, TILE_N> &cache,
						Symbols a, int a_offset,
						Strands h_strands, int h_offset,
						int i, int i_stride, int ii_limit)
{
	for (int ii = 0; ii < ii_limit; ++ii)
	{
		cache.a[ii] = a[a_offset + i + i_stride * ii];
		cache.h_strands[ii] = h_strands[h_offset + i + i_stride * ii];
	}
}

template<int TILE_M, int TILE_N, typename Strands>
inline void
store_cache_with_offsets(TiledCache<TILE_M, TILE_N> &cache,
						 Strands h_strands, int h_offset,
						 int i, int i_stride, int ii_limit)
{
	for (int ii = 0; ii < ii_limit; ++ii)
	{
		h_strands[h_offset + i + i_stride * ii] = cache.h_strands[ii];
	}
}

void
DivideIntoSubproblems(const LcsInput &input, int max_depth, int *num_subproblems_m, int *num_subproblems_n)
{
	int count_m = 1;
	int count_n = 1;
	int m_remaining = input.a_size;
	int n_remaining = input.b_size;

	constexpr int min_sub_size = 64;
	for (int depth = 0; depth < max_depth; ++depth)
	{
		if (m_remaining <= min_sub_size && n_remaining <= min_sub_size)
		{
			break;
		}
		if (m_remaining >= n_remaining)
		{
			m_remaining /= 2;
			count_m *= 2;
		}
		else
		{
			n_remaining /= 2;
			count_n *= 2;
		}
	}

	*num_subproblems_m = count_m;
	*num_subproblems_n = count_n;
}


template<int TILE_M, int TILE_N>
PermutationMatrix
PermutationFromDeinterleavedStrands(int *di_h_strands, int h_size, DeinterleavedArrayDescriptor<TILE_M> h_desc,
									int *di_v_strands, int v_size, DeinterleavedArrayDescriptor<TILE_N> v_desc)
{
	int size = h_size + v_size;
	int m = h_size;
	int n = v_size;
	int *h_strands = new int[m];
	int *v_strands = new int[n];

	for (int i = 0; i < m; ++i)
	{
		int i_di = h_desc.Deinterleave(i);
		h_strands[i] = di_h_strands[i_di];
	}
	for (int j = 0; j < n; ++j)
	{
		int j_di = v_desc.Deinterleave(j);
		v_strands[j] = di_v_strands[j_di];
	}
	PermutationMatrix result = PermutationMatrix::FromStrands(h_strands, m,
															  v_strands, n);
	delete[] h_strands;
	delete[] v_strands;

	return result;
}

template<int TILE_M, int TILE_N>
PermutationMatrix
PermutationMatrixRecursivelyCombine(int i_sub_first, int i_sub_count,
									int j_sub_first, int j_sub_count,
									GridEmbeddingGeneral<TILE_M, TILE_N, int, int> &grid)
{
	if (i_sub_count == 1 && j_sub_count == 1)
	{
		int h_size = grid.h_size_combined(i_sub_first, 1);
		int v_size = grid.v_size_combined(j_sub_first, 1);

		return PermutationFromDeinterleavedStrands(grid.h_strands + grid.gl.h_offset(i_sub_first, j_sub_first),
												   h_size,
												   grid.gl.sub_h_desc,
												   grid.v_strands + grid.gl.v_offset(i_sub_first, j_sub_first),
												   v_size,
												   grid.gl.sub_v_desc);
		//return PermutationMatrix::FromStrands(grid.h_strands + grid.h_offset(i_sub_first, j_sub_first), h_size,
		//									  grid.v_strands + grid.v_offset(i_sub_first, j_sub_first), v_size);
	}

	if (i_sub_count > j_sub_count)
	{
		int overlap = grid.v_size_combined(j_sub_first, j_sub_count);
		auto p = PermutationMatrixRecursivelyCombine(i_sub_first, i_sub_count / 2,
													 j_sub_first, j_sub_count,
													 grid);
		auto q = PermutationMatrixRecursivelyCombine(i_sub_first + i_sub_count / 2, i_sub_count / 2,
													 j_sub_first, j_sub_count,
													 grid);

		return staggered_multiply<true>(q, p, overlap);
	}
	else
	{
		int overlap = grid.h_size_combined(i_sub_first, i_sub_count);
		auto p = PermutationMatrixRecursivelyCombine(i_sub_first, i_sub_count,
													 j_sub_first, j_sub_count / 2,
													 grid);
		auto q = PermutationMatrixRecursivelyCombine(i_sub_first, i_sub_count,
													 j_sub_first + j_sub_count / 2, j_sub_count / 2,
													 grid);
		return staggered_multiply<false>(p, q, overlap);
	}
}

template<int TILE_M, int TILE_N>
PermutationMatrix
CombineMultiGridGeneral(GridEmbeddingGeneral<TILE_M, TILE_N, int, int> &grid)
{
	return PermutationMatrixRecursivelyCombine(0, grid.gl.num_subproblems_m,
											   0, grid.gl.num_subproblems_n,
											   grid);
}



template<int TILE_M, int TILE_N, typename Symbols, typename Strands>
inline void
update_tile_semilocal_with_offset(TiledCache<TILE_M, TILE_N> &cache,
								  Symbols a, int a_offset,
								  Symbols b, int b_offset,
								  Strands h_strands, int h_offset,
								  Strands v_strands, int v_offset,
								  int i_stride, int j_stride,
								  int i_low, int j_low)
{
	(void)a; (void)a_offset;
	(void)h_strands; (void)h_offset;

	for (int jj = 0; jj < TILE_N; ++jj)
	{
		auto v_strand = v_strands[v_offset + j_low + j_stride * jj];
		auto b_symbol = b[b_offset + j_low + j_stride * jj];

		for (int ii = TILE_M - 1; ii >= 0; --ii)
		{
			auto h_strand = cache.h_strands[ii];
			bool has_match = cache.a[ii] == b_symbol;
			bool has_crossing = h_strand > v_strand;
			bool need_swap = has_match || has_crossing;

			cache.h_strands[ii] = need_swap ? v_strand : h_strand;
			v_strand = need_swap ? h_strand : v_strand;
		}

		v_strands[v_offset + j_low + j_stride * jj] = v_strand;
	}
}

template<int TILE_M, int TILE_N, typename Symbols, typename Strands>
inline void
update_tile_semilocal_limit_with_offset(TiledCache<TILE_M, TILE_N> &cache,
										Symbols a, int a_offset,
										Symbols b, int b_offset,
										Strands h_strands, int h_offset,
										Strands v_strands, int v_offset,
										int i_stride, int j_stride,
										int i_low, int j_low,
										int ii_limit, int jj_limit)
{
	(void)a; (void)a_offset;
	(void)h_strands; (void)h_offset;

	for (int jj = 0; jj < jj_limit; ++jj)
	{
		auto v_strand = v_strands[v_offset + j_low + j_stride * jj];
		auto b_symbol = b[b_offset + j_low + j_stride * jj];

		for (int ii = ii_limit - 1; ii >= 0; --ii)
		{
			auto h_strand = cache.h_strands[ii];
			bool has_match = cache.a[ii] == b_symbol;
			bool has_crossing = h_strand > v_strand;
			bool need_swap = has_match || has_crossing;

			cache.h_strands[ii] = need_swap ? v_strand : h_strand;
			v_strand = need_swap ? h_strand : v_strand;
		}

		v_strands[v_offset + j_low + j_stride * jj] = v_strand;
	}
}


template<int SG_SIZE, int TILE_M, int TILE_N, int DEPTH, int THREAD_SUBDIVISION>
void
Lcs_General(const LcsInput &input, LcsContext &ctx)
{
	int num_subproblems_m = 1;
	int num_subproblems_n = 1;
	DivideIntoSubproblems(input, DEPTH, &num_subproblems_m, &num_subproblems_n);
	auto grid = make_embedding_general<TILE_M, TILE_N>(input, num_subproblems_m, num_subproblems_n);
	auto gl = grid.gl;

	int max_num_tiles_m = gl.num_tiles_m(gl.max_sub_m);
	int max_num_tiles_n = gl.num_tiles_n(gl.max_sub_n);

	int block_height = SG_SIZE;
	int num_blocks_m = CeilDiv(max_num_tiles_m, block_height);
	int block_width = CeilDiv(max_num_tiles_n, THREAD_SUBDIVISION);
	int num_blocks_n = CeilDiv(max_num_tiles_n, block_width);

	int num_subproblems_total = num_subproblems_m * num_subproblems_n;
	int num_passes = num_blocks_m + num_blocks_n - 1;

	int i_stride = gl.sub_h_desc.Stride();
	int j_stride = gl.sub_v_desc.Stride();


	int local_size = SG_SIZE;
	{
		auto buf = make_buffers(grid);

		for (int pass_idx = 0; pass_idx < num_passes; ++pass_idx)
		{
			auto block_diag = antidiag_at(pass_idx, num_blocks_m, num_blocks_n);
			int global_size = local_size * block_diag.diag_len * num_subproblems_total;
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);
				cgh.parallel_for(sycl::nd_range<1>(global_size, local_size),
								 [=](sycl::nd_item<1> item)
								 [[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					auto sg = item.get_sub_group();
					auto local_id = sg.get_local_linear_id();
					auto group_id = item.get_group_linear_id();

					int block_idx = group_id % block_diag.diag_len;
					int i_sub = (group_id / block_diag.diag_len) % gl.num_subproblems_m;
					int j_sub = (group_id / block_diag.diag_len) / gl.num_subproblems_m;

					int actual_sub_m = i_sub + 1 == gl.num_subproblems_m ? gl.clipped_sub_m : gl.max_sub_m;
					int actual_sub_n = j_sub + 1 == gl.num_subproblems_n ? gl.clipped_sub_n : gl.max_sub_n;

					int actual_num_tiles_m = CeilDiv(actual_sub_m, TILE_M);
					int actual_num_tiles_n = CeilDiv(actual_sub_n, TILE_N);

					int a_offset = gl.a_offset(i_sub, j_sub);
					int h_offset = gl.h_offset(i_sub, j_sub);
					int b_offset = gl.b_offset(i_sub, j_sub);
					int v_offset = gl.v_offset(i_sub, j_sub);



					int i0 = (block_diag.i_first + block_idx) * SG_SIZE;
					int left_border = (block_diag.j_first + block_idx) * block_width;
					int right_border = Min(left_border + block_width, actual_num_tiles_n);
					int i = i0 + local_id;

					int ii_incomplete = actual_sub_m % TILE_M;
					int jj_incomplete = actual_sub_n % TILE_N;

					bool incomplete_vertically = i0 + SG_SIZE > actual_num_tiles_m;
					bool tile_incomplete_vertically = (i0 + SG_SIZE >= actual_num_tiles_m) && (ii_incomplete > 0);
					bool tile_incomplete_horizontally = (right_border >= actual_num_tiles_n) && (jj_incomplete > 0);


					bool has_complete_part = !(incomplete_vertically || tile_incomplete_vertically);
					bool has_incomplete_part = !has_complete_part || tile_incomplete_horizontally;


					if (has_complete_part)
					{
						if (has_incomplete_part)
						{
							right_border = actual_num_tiles_n - 1;
						}
						TiledCache<TILE_M, TILE_N> cache;
						load_cache_with_offsets(cache,
												acc.a, a_offset,
												acc.h_strands, h_offset,
												i, i_stride, TILE_M);

						auto update_tile = [&](int i_low, int j_low)
						{

							//update_cell_tile_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
							//						   i_stride, j_stride, i_low, j_low);
							update_tile_semilocal_with_offset(cache,
															  acc.a, a_offset,
															  acc.b, b_offset,
															  acc.h_strands, h_offset,
															  acc.v_strands, v_offset,
															  i_stride, j_stride,
															  i_low, j_low);
						};

						update_stripe<SG_SIZE>(update_tile, sg, i0, left_border, right_border);

						store_cache_with_offsets(cache,
												 acc.h_strands, h_offset,
												 i, i_stride, TILE_M);
					}
					// Remaining incomplete part
					if (has_incomplete_part)
					{
						if (has_complete_part)
						{
							left_border = right_border;
							right_border = actual_num_tiles_n;
						}

						int ii_limit = ii_incomplete ? ii_incomplete : TILE_M;
						int jj_limit = jj_incomplete ? jj_incomplete : TILE_N;

						int ii_limit_here = i == actual_num_tiles_m - 1 ? ii_limit : TILE_M;

						TiledCache<TILE_M, TILE_N> cache;

						load_cache_with_offsets(cache,
												acc.a, a_offset,
												acc.h_strands, h_offset,
												i, i_stride, ii_limit_here);



						auto update_tile = [&](int i_low, int j_low)
						{
							int jj_limit_here = j_low == actual_num_tiles_n - 1 ? jj_limit : TILE_N;

							update_tile_semilocal_limit_with_offset(cache,
															  acc.a, a_offset,
															  acc.b, b_offset,
															  acc.h_strands, h_offset,
															  acc.v_strands, v_offset,
															  i_stride, j_stride,
															  i_low, j_low,
															  ii_limit_here, jj_limit_here);
						};
						auto within_bounds = [&](int i, int j)
						{
							return i < actual_num_tiles_m;
						};

						update_stripe_with_predicate<SG_SIZE>(update_tile, within_bounds, sg, i0,
															  left_border, right_border);
						store_cache_with_offsets(cache,
												 acc.h_strands, h_offset,
												 i, i_stride, ii_limit_here);

					}

				}); // end parallel_for
			}); // end submit
		} // end for 
	} // end buffers lifetime

	ctx.matrix = CombineMultiGridGeneral(grid);
}

