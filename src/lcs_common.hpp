#pragma once
// Common types and functions used by LCS implementations

#include "lcs_types.hpp"

template<typename Symbols, typename Strands>
inline void
update_cell_semilocal(Symbols a, Symbols b, Strands h_strands, Strands v_strands, int i, int j)
{
	auto a_sym = a[i];
	auto b_sym = b[j];
	auto h_strand = h_strands[i];
	auto v_strand = v_strands[j];

	bool has_match = a_sym == b_sym;
	bool has_crossing = h_strand > v_strand;
	bool need_swap = has_match || has_crossing;

	h_strands[i] = need_swap ? v_strand : h_strand;
	v_strands[j] = need_swap ? h_strand : v_strand;
}

template<typename Symbols, typename Strands>
inline void
update_cell_semilocal_separate_indexing(Symbols a, Symbols b, Strands h_strands, Strands v_strands,
										int i_symbols, int j_symbols,
										int i_strands, int j_strands)
{
	auto a_sym = a[i_symbols];
	auto b_sym = b[j_symbols];
	auto h_strand = h_strands[i_strands];
	auto v_strand = v_strands[j_strands];

	bool has_match = a_sym == b_sym;
	bool has_crossing = h_strand > v_strand;
	bool need_swap = has_match || has_crossing;

	h_strands[i_strands] = need_swap ? v_strand : h_strand;
	v_strands[j_strands] = need_swap ? h_strand : v_strand;
}




template<int TILE_M, int TILE_N>
struct TiledCache
{
	int a[TILE_M];
	int h_strands[TILE_M];
};


template<int TILE_M, int TILE_N, typename Symbols, typename Strands>
inline void
update_cell_tile_semilocal(TiledCache<TILE_M, TILE_N> &cache,
						   Symbols a, Symbols b, Strands h_strands, Strands v_strands,
						   int i_stride, int j_stride,
						   int i_low, int j_low)
{
	// a, h_strands are unused
	for (int jj = 0; jj < TILE_N; ++jj)
	{
		auto v_strand = v_strands[j_low + j_stride * jj];
		auto b_sym = b[j_low + j_stride * jj];

		#pragma unroll
		for (int ii = TILE_M - 1; ii >= 0; --ii)
		{
			auto h_strand = cache.h_strands[ii];

			bool has_match = cache.a[ii] == b_sym;
			bool has_crossing = h_strand > v_strand;
			bool need_swap = has_match || has_crossing;

			cache.h_strands[ii] = need_swap ? v_strand : h_strand;
			v_strand = need_swap ? h_strand : v_strand;
		}

		v_strands[j_low + j_stride * jj] = v_strand;
	}
}


template<int TILE_M, int TILE_N, typename Symbols, typename Strands>
inline void
update_cell_tile_limit_semilocal(TiledCache<TILE_M, TILE_N> &cache,
								 Symbols a, Symbols b, Strands h_strands, Strands v_strands,
								 int i_stride, int j_stride,
								 int i_low, int j_low, int ii_limit, int jj_limit)
{
	// a, h_strands are unused
	for (int jj = 0; jj < jj_limit; ++jj)
	{
		auto v_strand = v_strands[j_low + j_stride * jj];
		auto b_sym = b[j_low + j_stride * jj];

		for (int ii = ii_limit - 1; ii >= 0; --ii)
		{
			auto h_strand = cache.h_strands[ii];

			bool has_match = cache.a[ii] == b_sym;
			bool has_crossing = h_strand > v_strand;
			bool need_swap = has_match || has_crossing;

			cache.h_strands[ii] = need_swap ? v_strand : h_strand;
			v_strand = need_swap ? h_strand : v_strand;
		}

		v_strands[j_low + j_stride * jj] = v_strand;
	}
}



// Striped iteration abstraction

template<int SG_SIZE, typename Functor0, typename Functor1>
inline void
stripe_iterate(Functor0 functor_complete, Functor1 functor_incomplete, sycl::sub_group sg,
			   int i0, int left_border, int right_border)
{
	int sg_id = sg.get_local_linear_id();
	int i = i0 + sg_id;

	// Make it so right part is the only part when width is small
	int left_part_first = left_border - SG_SIZE + 1;
	int right_part_first = right_border < SG_SIZE ? left_part_first : Max(0, right_border - SG_SIZE);
	int left_part_end = Min(0, right_part_first);

	for (int j0 = left_part_first; j0 < left_part_end; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border)
		{
			functor_complete(i, j);
		}
		sg.barrier();
	}

	for (int j0 = 0; j0 < right_part_first; ++j0)
	{
		int j = j0 + sg_id;
		{
			functor_complete(i, j);
		}
		sg.barrier();
	}

	for (int j0 = right_part_first; j0 < right_border; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border)
		{
			functor_incomplete(i, j);
		}
		sg.barrier();
	}
	sg.barrier();

}

template<int SG_SIZE, typename Functor, typename Predicate>
inline void
stripe_iterate_with_predicate(Functor functor, Predicate pred, sycl::sub_group sg,
							  int i0, int left_border, int right_border)
{
	int sg_id = sg.get_local_linear_id();
	int i = i0 + sg_id;

	// Make it so right part is the only part when width is small
	int left_part_first = left_border - SG_SIZE + 1;
	int right_part_first = right_border < SG_SIZE ? left_part_first : Max(0, right_border - SG_SIZE);
	int left_part_end = Min(0, right_part_first);

	for (int j0 = left_part_first; j0 < left_part_end; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border && pred(i, j))
		{
			functor(i, j);
		}
		sg.barrier();
	}

	for (int j0 = 0; j0 < right_part_first; ++j0)
	{
		int j = j0 + sg_id;
		if (pred(i, j))
		{
			functor(i, j);
		}
		sg.barrier();
	}

	for (int j0 = right_part_first; j0 < right_border; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border && pred(i, j))
		{
			functor(i, j);
		}
		sg.barrier();
	}
	sg.barrier();

}



template<int SG_SIZE, typename Functor>
inline void
update_stripe(Functor functor, sycl::sub_group sg,
			  int i0, int left_border, int right_border)
{
	int sg_id = sg.get_local_linear_id();
	int i = i0 + sg_id;

	// Make it so right part is the only part when width is small
	int left_part_first = left_border - SG_SIZE + 1;
	int right_part_first = right_border < SG_SIZE ? left_part_first : Max(left_border, right_border - SG_SIZE);
	int left_part_end = Min(left_border, right_part_first);

	for (int j0 = left_part_first; j0 < left_part_end; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border)
		{
			functor(i, j);
		}
		sg.barrier();
	}

	for (int j0 = left_part_end; j0 < right_part_first; ++j0)
	{
		int j = j0 + sg_id;
		{
			functor(i, j);
		}
		sg.barrier();
	}

	for (int j0 = right_part_first; j0 < right_border; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border)
		{
			functor(i, j);
		}
		sg.barrier();
	}

	sg.barrier();
}


template<int SG_SIZE, typename Functor, typename Predicate>
inline void
update_stripe_with_predicate(Functor functor, Predicate pred, sycl::sub_group sg,
							 int i0, int left_border, int right_border)
{
	int sg_id = sg.get_local_linear_id();
	int i = i0 + sg_id;

	// Make it so right part is the only part when width is small
	int left_part_first = left_border - SG_SIZE + 1;
	int right_part_first = right_border < SG_SIZE ? left_part_first : Max(left_border, right_border - SG_SIZE);
	int left_part_end = Min(left_border, right_part_first);

	for (int j0 = left_part_first; j0 < left_part_end; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border && pred(i, j))
		{
			functor(i, j);
		}
		sg.barrier();
	}

	for (int j0 = left_part_end; j0 < right_part_first; ++j0)
	{
		int j = j0 + sg_id;
		if (pred(i, j))
		{
			functor(i, j);
		}
		sg.barrier();
	}

	for (int j0 = right_part_first; j0 < right_border; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border && pred(i, j))
		{
			functor(i, j);
		}
		sg.barrier();
	}

	sg.barrier();

}


template<int TILE_M, int TILE_N, typename Symbols, typename Strands>
inline void
load_cache(TiledCache<TILE_M, TILE_N> &cache, Symbols a, Strands h_strands,
		   int i, int i_stride, int ii_limit)
{
	for (int ii = 0; ii < ii_limit; ++ii)
	{
		cache.a[ii] = a[i + i_stride * ii];
		cache.h_strands[ii] = h_strands[i + i_stride * ii];
	}
}

template<int TILE_M, int TILE_N, typename Strands>
inline void
store_cache(TiledCache<TILE_M, TILE_N> &cache, Strands h_strands,
			int i, int i_stride, int ii_limit)
{
	for (int ii = 0; ii < ii_limit; ++ii)
	{
		h_strands[i + i_stride * ii] = cache.h_strands[ii];
	}
}

