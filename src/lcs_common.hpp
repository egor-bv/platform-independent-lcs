#pragma once

#include "lcs_interface_internal.hpp"
#include "lcs_grid.hpp"
#include "sycl_utility.hpp"

template<typename Symbols, typename Strands>
inline void
update_cell_semilocal(Symbols a, Symbols b,
					  Strands h_strands, Strands v_strands,
					  int i, int j)
{
	auto a_symbol = a[i];
	auto b_symbol = b[j];
	auto h_strand = h_strands[i];
	auto v_strand = v_strands[j];

	bool has_match = a_symbol == b_symbol;
	bool has_crossing = h_strand > v_strand;
	bool need_swap = has_match || has_crossing;

	h_strands[i] = need_swap ? v_strand : h_strand;
	v_strands[j] = need_swap ? h_strand : v_strand;
}

template<int TILE_M>
struct TiledCache
{
	Word a[TILE_M];
	Word h_strands[TILE_M];
};

template<int TILE_M, typename Symbols, typename Strands>
inline void
load_cache(TiledCache<TILE_M> &cache, Symbols a, Strands h_strands,
		   int i, int i_stride, int ii_limit)
{
	for (int ii = 0; ii < ii_limit; ++ii)
	{
		cache.a[ii] = a[i + i_stride * ii];
		cache.h_strands[ii] = h_strands[i + i_stride * ii];
	}
}

template<int TILE_M, typename Strands>
inline void
store_cache(TiledCache<TILE_M> &cache, Strands h_strands,
			int i, int i_stride, int ii_limit)
{
	for (int ii = 0; ii < ii_limit; ++ii)
	{
		h_strands[i + i_stride * ii] = cache.h_strands[ii];
	}
}




template<int TILE_M, int TILE_N, typename Symbols, typename Strands>
inline void
update_tile_semilocal(TiledCache<TILE_M> &cache,
					  Symbols a, Symbols b,
					  Strands h_strands, Strands v_strands,
					  int i_stride, int j_stride,
					  int i_low, int j_low)
{
	UNUSED(h_strands);
	UNUSED(a);

	// #pragma unroll
	for (int jj = 0; jj < TILE_N; ++jj)
	{
		auto v_strand = v_strands[j_low + j_stride * jj];
		auto b_symbol = b[j_low + j_stride * jj];

		#pragma unroll
		for (int ii = TILE_M - 1; ii >= 0; --ii)
		{
			auto h_strand = cache.h_strands[ii];
			auto a_symbol = cache.a[ii];

			bool has_match = a_symbol == b_symbol;
			bool has_crossing = h_strand > v_strand;
			bool need_swap = has_match || has_crossing;

			cache.h_strands[ii] = need_swap ? v_strand : h_strand;
			v_strand = need_swap ? h_strand : v_strand;
		}

		v_strands[j_low + j_stride * jj] = v_strand;
	}
}


template<int TILE_M, int TILE_N, typename SymbolsA, typename SymbolsB, typename StrandsH, typename StrandsV>
inline void
update_tile_semilocal(TiledCache<TILE_M> &cache,
					  SymbolsA a, SymbolsB b,
					  StrandsH h_strands, StrandsV v_strands,
					  int i_stride, int j_stride,
					  int i_low, int j_low)
{
	UNUSED(h_strands);
	UNUSED(a);

	// #pragma unroll
	for (int jj = 0; jj < TILE_N; ++jj)
	{
		auto v_strand = v_strands[j_low + j_stride * jj];
		auto b_symbol = b[j_low + j_stride * jj];

		#pragma unroll
		for (int ii = TILE_M - 1; ii >= 0; --ii)
		{
			auto h_strand = cache.h_strands[ii];
			auto a_symbol = cache.a[ii];

			bool has_match = a_symbol == b_symbol;
			bool has_crossing = h_strand > v_strand;
			bool need_swap = has_match || has_crossing;

			cache.h_strands[ii] = need_swap ? v_strand : h_strand;
			v_strand = need_swap ? h_strand : v_strand;
		}

		v_strands[j_low + j_stride * jj] = v_strand;
	}
}


template<int SG_SIZE, typename Functor>
inline void
update_stripe(Functor functor, sycl::sub_group sg,
			  int i0, int left_border, int right_border)
{
	int sg_id = sg.get_local_linear_id();
	int i = i0 + sg_id;

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


template<int SG_SIZE, int TILE_M, int TILE_N, typename Accessors>
inline void
update_block_tiled_semilocal_with_cache(sycl::sub_group sg, Accessors acc,
										const GridShapeTiled &shape, const BlockShape &block)
{
	auto cache = TiledCache<TILE_M>();

	int i0 = block.i0;
	int i = i0 + sg.get_local_linear_id();


	int i_stride = shape.h_desc.stride;
	int j_stride = shape.v_desc.stride;

	load_cache(cache, acc.a, acc.h_strands, i, i_stride, TILE_M);

	int left_border = block.j0;
	int right_border = left_border + block.jsize;

	auto update_tile = [&](int i_low, int j_low)
	{
		update_tile_semilocal<TILE_M, TILE_N>(cache,
											  acc.a, acc.b,
											  acc.h_strands, acc.v_strands,
											  i_stride, j_stride,
											  i_low, j_low);
	};

	update_stripe<SG_SIZE>(update_tile, sg, i0, left_border, right_border);

	store_cache(cache, acc.h_strands, i, i_stride, TILE_M);
}

#if 1
template<int SG_SIZE, int TILE_M, int TILE_N, typename Accessors, typename LocalSymbols, typename LocalStrands>
inline void
update_block_tiled_semilocal_with_slm(sycl::sub_group sg, Accessors acc, int slm_slot_count, LocalSymbols b_local, LocalStrands v_strands_local,
									  const GridShapeTiled &shape, const BlockShape &block)
{

	int i0 = block.i0;
	int sg_id = sg.get_local_linear_id();
	int i = i0 + sg_id;

	int left_border = block.j0;
	int right_border = left_border + block.jsize;

	int left_part_first = left_border - SG_SIZE + 1;
	int right_part_first = right_border < SG_SIZE ? left_part_first : Max(left_border, right_border - SG_SIZE);
	int left_part_end = Min(left_border, right_part_first);

	int num_slots = block.jsize / SG_SIZE;
	int local_stride = SG_SIZE * slm_slot_count;

	int j_stride = shape.v_desc.stride;
	int i_stride = shape.h_desc.stride;

	if (0 && sg_id == 0)
	{
		K_PRINTF("i0=%d (%d..%d..%d..%d)\n", i0, left_part_first, left_part_end, right_part_first, right_border);
		K_PRINTF("num_slots: %d, locsl_stride: %d\n", num_slots, local_stride);
	}
	K_ASSERT_INT_EQ(block.jsize, SG_SIZE * num_slots); // Must be evenly divisible

	auto load_into_slm = [&](int slot_idx)
	{
		if (0 && sg_id == 0)
		{
			K_PRINTF("i0: %d, load slot: %d\n", i0, slot_idx);
		}
		for (int jj = 0; jj < TILE_N; ++jj)
		{
			int local_idx = (slot_idx % slm_slot_count) * SG_SIZE + sg_id + jj * local_stride;
			int global_idx = left_border + slot_idx * SG_SIZE + sg_id + jj * j_stride;
			if (0 && sg_id == 0 && jj == 0)
			{
				K_PRINTF("i0=%d: LOAD[%d]: local_idx: %d, global_idx: %d\n", i0, slot_idx, local_idx, global_idx);
			}
			b_local[local_idx] = acc.b[global_idx];
			v_strands_local[local_idx] = acc.v_strands[global_idx];
		}
	};

	auto store_from_slm = [&](int slot_idx)
	{
		if (0 && sg_id == 0)
		{
			K_PRINTF("i0: %d, store slot:%d\n", i0, slot_idx);
		}
		for (int jj = 0; jj < TILE_N; ++jj)
		{
			int local_idx = (slot_idx % slm_slot_count) * SG_SIZE + sg_id + jj * local_stride;
			int global_idx = left_border + slot_idx * SG_SIZE + sg_id + jj * j_stride;
			if (0 && sg_id == 0 && jj == 0)
			{
				K_PRINTF("i0=%d: STORE[%d]: local_idx: %d, global_idx: %d\n", i0, slot_idx, local_idx, global_idx);
			}
			acc.v_strands[global_idx] = v_strands_local[local_idx];
		}
	};

	auto cache = TiledCache<TILE_M>();
	load_cache(cache, acc.a, acc.h_strands, i, i_stride, TILE_M);

	auto functor = [&](int i_low, int j_low)
	{
		update_tile_semilocal<TILE_M, TILE_N>(cache,
											  acc.a, b_local,
											  acc.h_strands, v_strands_local,
											  i_stride, local_stride,
											  i_low, j_low % local_stride);
	};

	K_ASSERT(num_slots >= 4);

	load_into_slm(0);
	load_into_slm(1);
	// Do left part...


	for (int j0 = left_part_first; j0 < left_part_end; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border)
		{
			functor(i, j - left_part_first);
		}
		sg.barrier();
	}

	for (int step = 0; step < num_slots - 1; ++step)
	{
		if (step + 2 < num_slots) load_into_slm(step + 2);
		sg.barrier();
		if (0 && sg_id == 0) { K_PRINTF("i0: %d, j0: %d..%d\n", i0, left_part_end + SG_SIZE * step, left_part_end + SG_SIZE * step + SG_SIZE); }
		for (int j0 = 0; j0 < SG_SIZE; ++j0)
		{

			int j = left_part_end - left_part_first + j0 + SG_SIZE * step + sg_id;
			{
				functor(i, j);
			}
			sg.barrier();
		}
		store_from_slm(step);
	}
	sg.barrier();

	for (int j0 = right_part_first; j0 < right_border; ++j0)
	{
		int j = j0 + sg_id;
		if (j >= left_border && j < right_border)
		{
			functor(i, j - left_part_first);
		}
		sg.barrier();
	}

	sg.barrier();

	// store_from_slm(num_slots - 2);
	store_from_slm(num_slots - 1);

	sg.barrier();

	store_cache(cache, acc.h_strands, i, i_stride, TILE_M);
}
#endif

template<int SG_SIZE, int TILE_M, int TILE_N, typename Accessors, typename LocalSymbols, typename LocalStrands>
inline void
update_block_tiled_semilocal_with_slm_at_once(sycl::sub_group sg,
												  Accessors acc,
												  int slm_slot_count, LocalSymbols b_local, LocalStrands v_strands_local,
												  const GridShapeTiled &shape, const BlockShape &block)
{

	int i0 = block.i0;
	int sg_id = sg.get_local_linear_id();
	int i = i0 + sg_id;

	int left_border = block.j0;
	int right_border = left_border + block.jsize;

	int left_part_first = left_border - SG_SIZE + 1;
	int right_part_first = right_border < SG_SIZE ? left_part_first : Max(left_border, right_border - SG_SIZE);
	int left_part_end = Min(left_border, right_part_first);

	int i_stride = shape.h_desc.stride;
	int j_stride = shape.v_desc.stride;

	int local_stride = slm_slot_count * SG_SIZE;


	// Load everything in cache...
	for (int slot_idx = 0; slot_idx < slm_slot_count; ++slot_idx)
	{
		for (int jj = 0; jj < TILE_N; ++jj)
		{
			int local_idx = slot_idx * SG_SIZE + jj * local_stride + sg_id;
			int global_idx = left_border + slot_idx * SG_SIZE + jj * j_stride + sg_id;

			b_local[local_idx] = acc.b[global_idx];
			v_strands_local[local_idx] = acc.v_strands[global_idx];
		}
	}

	auto cache = TiledCache<TILE_M>();
	load_cache(cache, acc.a, acc.h_strands, i, i_stride, TILE_M);

	auto functor = [&](int i_low, int j_low)
	{
		update_tile_semilocal<TILE_M, TILE_N>(cache,
											  acc.a, b_local,
											  acc.h_strands, v_strands_local,
											  i_stride, local_stride,
											  i_low, j_low % local_stride);
	};

	// Actual combing
	{
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
	}



	store_cache(cache, acc.h_strands, i, i_stride, TILE_M);

	for (int slot_idx = 0; slot_idx < slm_slot_count; ++slot_idx)
	{
		for (int jj = 0; jj < TILE_N; ++jj)
		{
			int local_idx = slot_idx * SG_SIZE + jj * local_stride + sg_id;
			int global_idx = left_border + slot_idx * SG_SIZE + jj * j_stride + sg_id;

			acc.v_strands[global_idx] = v_strands_local[local_idx];
		}
	}


}