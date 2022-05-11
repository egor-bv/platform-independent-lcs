#pragma once

#include "lcs_types.hpp"
#include "lcs_common.hpp"
#include "lcs_grid.hpp"





// Minimally factored implementation to ensure correctness / reference
template<int SG_SIZE, int TILE_M, int TILE_N>
void Lcs_Semi_Tiled_ST_Reference(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	{
		auto buf = make_buffers(grid);
		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);

			int stripe_size = SG_SIZE;
			int stripe_count = region_m / stripe_size;

			int left_border = 0;
			int right_border = region_n;

			cgh.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				int sg_id = sg.get_local_linear_id();

				// NOTE: pulling these lambdas out is not easy -- almost 2x decrease in performance
				auto process_cell_tile = [=](TiledCache<TILE_M, TILE_N> &cache,
					int i_low, int j_low)
				{
					for (int jj = 0; jj < TILE_N; ++jj)
					{
						auto v_strand = acc.v_strands[j_low + j_stride * jj];
						auto b_sym = acc.b[j_low + j_stride * jj];

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

						acc.v_strands[j_low + j_stride * jj] = v_strand;
					}
				};

				auto process_cell_tile_limit = [=](TiledCache<TILE_M, TILE_N> &cache,
					int i_low, int j_low, int region_m, int region_n, int ii_limit, int jj_limit)
				{
					ii_limit = i_low == region_m - 1 ? ii_limit : TILE_M;
					jj_limit = j_low == region_n - 1 ? jj_limit : TILE_N;
					for (int jj = 0; jj < jj_limit; ++jj)
					{
						auto v_strand = acc.v_strands[j_low + j_stride * jj];
						auto b_sym = acc.b[j_low + j_stride * jj];

						for (int ii = ii_limit - 1; ii >= 0; --ii)
						{
							auto h_strand = cache.h_strands[ii];

							bool has_match = cache.a[ii] == b_sym;
							bool has_crossing = h_strand > v_strand;
							bool need_swap = has_match || has_crossing;

							cache.h_strands[ii] = need_swap ? v_strand : h_strand;
							v_strand = need_swap ? h_strand : v_strand;
						}

						acc.v_strands[j_low + j_stride * jj] = v_strand;
					}
				};

				int top_stripe = stripe_count * stripe_size == region_m ? stripe_count - 1 : stripe_count;
				// top stripe
				// run special code that assumes both incomplete stripe & incomplete tile
				{
					int i = top_stripe * SG_SIZE + sg_id;
					int top_border = region_m;
					TiledCache<TILE_M, TILE_N> cache;


					int ii_limit_here = i == region_m - 1 ? ii_limit : TILE_M;
					// load tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						int i_global = i + i_stride * ii;
						if (ii < ii_limit_here) cache.a[ii] = acc.a[i_global];
						if (ii < ii_limit_here) cache.h_strands[ii] = acc.h_strands[i_global];
					}

					// left part
					for (int j0 = left_border - 1 - SG_SIZE; j0 < left_border; ++j0)
					{
						int j = j0 + sg_id;
						if (j >= left_border && j < right_border && i < top_border)
						{
							process_cell_tile_limit(cache, i, j, region_m, region_n, ii_limit, jj_limit);
						}
						sg.barrier();
					}

					// middle part
					int right_part_first = Max(left_border, right_border - SG_SIZE);
					for (int j0 = 0; j0 < right_part_first; ++j0)
					{
						int j = j0 + sg_id;
						if (i < top_border)
						{
							process_cell_tile_limit(cache, i, j, region_m, region_n, ii_limit, jj_limit);
						}
						sg.barrier();
					}

					// right part & assume incomplete tile
					for (int j0 = right_part_first; j0 < right_border; ++j0)
					{
						int j = j0 + sg_id;
						if (j < right_border && i < top_border)
						{
							process_cell_tile_limit(cache, i, j, region_m, region_n, ii_limit, jj_limit);
						}
						sg.barrier();
					}

					// store back tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						int i_global = i + i_stride * ii;
						if (ii < ii_limit_here) acc.h_strands[i_global] = cache.h_strands[ii];
					}

					sg.barrier();
				}

				// complete stripes
				for (int stripe = top_stripe - 1; stripe >= 0; --stripe)
				{
					int i = stripe * SG_SIZE + sg_id;
					TiledCache<TILE_M, TILE_N> cache;

					// load tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						cache.a[ii] = acc.a[i + i_stride * ii];
						cache.h_strands[ii] = acc.h_strands[i + i_stride * ii];
					}

					int left_part_first = left_border - SG_SIZE + 1;
					int right_part_first = right_border < SG_SIZE ? left_part_first : Max(0, right_border - SG_SIZE);
					int left_part_end = Min(0, right_part_first);

					// left part
					for (int j0 = left_part_first; j0 < left_part_end; ++j0)
					{
						int j = j0 + sg_id;
						if (j >= left_border && j < right_border)
						{
							process_cell_tile(cache, i, j);
						}
						sg.barrier();
					}

					// middle part

					for (int j0 = 0; j0 < right_part_first; ++j0)
					{
						int j = j0 + sg_id;
						process_cell_tile(cache, i, j);
						sg.barrier();
					}

					// right part & assume incomplete tile
					for (int j0 = right_part_first; j0 < right_border; ++j0)
					{
						int j = j0 + sg_id;
						if (j >= left_border && j < right_border)
						{
							process_cell_tile_limit(cache, i, j, region_m, region_n, ii_limit, jj_limit);
						}
						sg.barrier();
					}

					// store back tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						acc.h_strands[i + i_stride * ii] = cache.h_strands[ii];
					}

					sg.barrier();
				}
			});
		});
	}

	copy_strands_deinterleaved(ctx, grid);
}


// A more factored version
template<int SG_SIZE, int TILE_M, int TILE_N>
void Lcs_Semi_Tiled_ST_Factored(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	{
		auto buf = make_buffers(grid);
		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);

			int stripe_size = SG_SIZE;
			int stripe_count = region_m / stripe_size;

			int left_border = 0;
			int right_border = region_n;

			cgh.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				int sg_id = sg.get_local_linear_id();

				int top_stripe = stripe_count * stripe_size == region_m ? stripe_count - 1 : stripe_count;
				int top_border = region_m;

				// NOTE: it's important that we have different caches & there's no dependency between them
				TiledCache<TILE_M, TILE_N> cache;
				TiledCache<TILE_M, TILE_N> cache1;

				auto process_cell_tile_limit = [&](int i_low, int j_low)
				{
					int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
					int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

					update_cell_tile_limit_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
						i_stride, j_stride, i_low, j_low, ii_limit_here, jj_limit_here);
				};

				auto out_of_bounds_predicate = [&](int i, int j)
				{
					return i < top_border;
				};

				auto process_cell_tile = [&](int i_low, int j_low)
				{
					update_cell_tile_semilocal(cache1, acc.a, acc.b, acc.h_strands, acc.v_strands,
						i_stride, j_stride, i_low, j_low);
				};

				auto process_cell_tile_limit1 = [&](int i_low, int j_low)
				{
					int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
					int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

					update_cell_tile_limit_semilocal(cache1, acc.a, acc.b, acc.h_strands, acc.v_strands,
						i_stride, j_stride, i_low, j_low, ii_limit_here, jj_limit_here);
				};


				// top stripe
				// run special code that assumes both incomplete stripe & incomplete tile
				{
					int i = top_stripe * SG_SIZE + sg_id;

					auto out_of_bounds_predicate = [&](int i, int j)
					{
						return i < top_border;
					};

					int ii_limit_here = i == region_m - 1 ? ii_limit : TILE_M;
					// load tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						int i_global = i + i_stride * ii;
						if (ii < ii_limit_here) cache.a[ii] = acc.a[i_global];
						if (ii < ii_limit_here) cache.h_strands[ii] = acc.h_strands[i_global];
					}

					stripe_iterate_with_predicate<SG_SIZE>(process_cell_tile_limit, out_of_bounds_predicate,
						sg, top_stripe * SG_SIZE, left_border, right_border);

					// store back tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						int i_global = i + i_stride * ii;
						if (ii < ii_limit_here) acc.h_strands[i_global] = cache.h_strands[ii];
					}

					sg.barrier();
				}

				// complete stripes
				for (int stripe = top_stripe - 1; stripe >= 0; --stripe)
				{
					int i = stripe * SG_SIZE + sg_id;

					// load tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						cache1.a[ii] = acc.a[i + i_stride * ii];
						cache1.h_strands[ii] = acc.h_strands[i + i_stride * ii];
					}

					stripe_iterate<SG_SIZE>(process_cell_tile, process_cell_tile_limit1,
						sg, stripe * SG_SIZE, left_border, right_border);

					// store back tiled cache
					for (int ii = 0; ii < TILE_M; ++ii)
					{
						acc.h_strands[i + i_stride * ii] = cache1.h_strands[ii];
					}

					sg.barrier();
				}
			});
		});
	}

	copy_strands_deinterleaved(ctx, grid);
}



// only supports exact subdivisions
template<int SG_SIZE, int TILE_M, int TILE_N>
void Lcs_Semi_Tiled_ST(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int stripe_size = SG_SIZE;
	int stripe_count = region_m / stripe_size;

	//int block_width = region_n;
	//int num_blocks_n = region_n / block_width + (region_m % block_width != 0);
	//int num_blocks_m = stripe_count;

	//int pass_count = num_blocks_n + num_blocks_m - 1;

	{
		auto buf = make_buffers(grid);

		// only one pass
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

				int left_border = 0;
				int right_border = region_n;

				int local_size = SG_SIZE;
				int global_size = SG_SIZE;


				cgh.parallel_for(
					sycl::nd_range<1>(local_size, global_size),
					[=](sycl::nd_item<1> item)
					[[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					auto sg = item.get_sub_group();
					int sg_id = sg.get_local_linear_id();
					//for (int stripe = stripe_count - 1; stripe >= 0; --stripe)
					//{
					//	int i0 = stripe * SG_SIZE;
					//	int left_border = 0;
					//	int right_border = region_n;


					//	TiledCache<TILE_M, TILE_N> cache;
					//	load_cache(cache, acc.a, acc.h_strands, i0, i_stride, TILE_M);
					//	sg.barrier();

					//	auto update_cell = [&](int i_low, int j_low)
					//	{
					//		update_cell_tile_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
					//			i_stride, j_stride, i_low, j_low);
					//	};

					//	update_stripe<SG_SIZE>(update_cell, sg, i0, left_border, right_border);

					//	store_cache(cache, acc.h_strands, i0, i_stride, TILE_M);
					//	sg.barrier();
					//}



					// complete stripes
					for (int stripe = stripe_count - 1; stripe >= 0; --stripe)
					{
						TiledCache<TILE_M, TILE_N> cache1;

						auto process_cell_tile = [&](int i_low, int j_low)
						{
							update_cell_tile_semilocal(cache1, acc.a, acc.b, acc.h_strands, acc.v_strands,
								i_stride, j_stride, i_low, j_low);
						};

						//auto process_cell_tile_limit1 = [&](int i_low, int j_low)
						//{
						//	int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
						//	int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

						//	update_cell_tile_limit_semilocal(cache1, acc.a, acc.b, acc.h_strands, acc.v_strands,
						//		i_stride, j_stride, i_low, j_low, ii_limit_here, jj_limit_here);
						//};


						int i = stripe * SG_SIZE + sg_id;

						// load tiled cache
						load_cache(cache1, acc.a, acc.h_strands, i, i_stride, TILE_M);
						//for (int ii = 0; ii < TILE_M; ++ii)
						//{
						//	cache1.a[ii] = acc.a[i + i_stride * ii];
						//	cache1.h_strands[ii] = acc.h_strands[i + i_stride * ii];
						//}

						update_stripe<SG_SIZE>(process_cell_tile,
							sg, stripe * SG_SIZE, left_border, right_border);

						// store back tiled cache
						//for (int ii = 0; ii < TILE_M; ++ii)
						//{
						//	acc.h_strands[i + i_stride * ii] = cache1.h_strands[ii];
						//}

						store_cache(cache1, acc.h_strands, i, i_stride, TILE_M);

						sg.barrier();
					}


				});
			});
		}
	}

	copy_strands_deinterleaved(ctx, grid);
}




#include <sycl/ext/oneapi/experimental/builtins.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

static const CONSTANT char FMT_GROUP[] = "group_id: %d, left_border: %d, right_border: %d\n";
// static const CONSTANT char FMT_GROUP[]

#define K_PRINTF(FMT, ...) const CONSTANT char FMT_[] = FMT; sycl::ext::oneapi::experimental::printf(FMT_, __VA_ARGS__)


template<int SG_SIZE, int TILE_M, int TILE_N>
void Lcs_Semi_Tiled_MT(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int substripe_count = 4;
	int stripe_size = SG_SIZE * substripe_count;
	int stripe_count = region_m / stripe_size;
	int num_blocks_m = stripe_count;

	int block_width = region_n / 16;
	int num_blocks_n = region_n / block_width + (region_n % block_width != 0);

	int pass_count = num_blocks_m + num_blocks_n - 1;

	{
		auto buf = make_buffers(grid);

		for (int pass = 0; pass < pass_count; ++pass)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

				// int left_border = 0;
				// int right_border = region_n;

				// compute shape of the pass...
				auto block_diag = antidiag_at(pass, num_blocks_m, num_blocks_n);

				int local_size = SG_SIZE;
				int global_size = SG_SIZE * block_diag.diag_len;

				cgh.parallel_for(
					sycl::nd_range<1>(global_size, local_size),
					[=](sycl::nd_item<1> item)
					[[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					auto sg = item.get_sub_group();
					int sg_id = sg.get_local_linear_id();
					int group_id = item.get_group_linear_id();


					int i0_stripe = (block_diag.i_first + group_id) * stripe_size;
					int left_border = (block_diag.j_first + group_id) * block_width;
					int right_border = left_border + block_width;

					for(int substripe = 0; substripe < substripe_count; ++substripe)
					{
						int i0 = i0_stripe + substripe * SG_SIZE;
						int i = i0 + sg_id;
						TiledCache<TILE_M, TILE_N> cache;
						load_cache(cache, acc.a, acc.h_strands, i, i_stride, TILE_M);

						auto process_cell_tile = [&](int i_low, int j_low)
						{
							update_cell_tile_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
								i_stride, j_stride, i_low, j_low);
						};

						update_stripe<SG_SIZE>(process_cell_tile,
							sg, i0, left_border, right_border);


						store_cache(cache, acc.h_strands, i, i_stride, TILE_M);
						sg.barrier();
					}

				});
			});
		}
	}

	copy_strands_deinterleaved(ctx, grid);
}



template<int SG_SIZE, int TILE_M, int TILE_N>
void Lcs_Semi_Tiled_MT_good(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int stripe_size = SG_SIZE;
	int stripe_count = region_m / stripe_size;
	int num_blocks_m = stripe_count;

	int block_width =  region_n / 32;
	int num_blocks_n = region_n / block_width + (region_n % block_width != 0);

	int pass_count = num_blocks_m + num_blocks_n - 1;

	{
		auto buf = make_buffers(grid);

		for (int pass = 0; pass < pass_count; ++pass)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

				// int left_border = 0;
				// int right_border = region_n;

				// compute shape of the pass...
				auto block_diag = antidiag_at(pass, num_blocks_m, num_blocks_n);

				int local_size = SG_SIZE;
				int global_size = SG_SIZE * block_diag.diag_len;

				cgh.parallel_for(
					sycl::nd_range<1>(global_size, local_size),
					[=](sycl::nd_item<1> item)
					[[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					auto sg = item.get_sub_group();
					int sg_id = sg.get_local_linear_id();
					int group_id = item.get_group_linear_id();

					// do a single block
					{
						int i0 = (block_diag.i_first + group_id) * SG_SIZE;
						int left_border = (block_diag.j_first + group_id) * block_width;
						int right_border = left_border + block_width;

						int i = i0 + sg_id;
						TiledCache<TILE_M, TILE_N> cache;
						load_cache(cache, acc.a, acc.h_strands, i, i_stride, TILE_M);

						auto process_cell_tile = [&](int i_low, int j_low)
						{
							update_cell_tile_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
								i_stride, j_stride, i_low, j_low);
						};

						update_stripe<SG_SIZE>(process_cell_tile,
							sg, i0, left_border, right_border);


						store_cache(cache, acc.h_strands, i, i_stride, TILE_M);

					}

				});
			});
		}
	}

	copy_strands_deinterleaved(ctx, grid);
}





template<int SG_SIZE, int TILE_M, int TILE_N>
void Lcs_Semi_Tiled_MT_bad(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int stripe_size = SG_SIZE;
	int stripe_count = region_m / stripe_size;
	int num_blocks_m = stripe_count;

	int block_width = 64;
	int num_blocks_n = region_n / block_width + (region_m % block_width != 0);

	int pass_count = num_blocks_m + num_blocks_n - 1;

	{
		auto buf = make_buffers(grid);

		for (int pass = 0; pass < pass_count; ++pass)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

				// int left_border = 0;
				// int right_border = region_n;

				// compute shape of the pass...
				auto block_diag = antidiag_at(pass, num_blocks_m, num_blocks_n);

				int local_size = SG_SIZE;
				int global_size = SG_SIZE * block_diag.diag_len;


				cgh.parallel_for(
					sycl::nd_range<1>(local_size, global_size),
					[=](sycl::nd_item<1> item)
					[[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					auto sg = item.get_sub_group();
					int sg_id = sg.get_local_linear_id();
					int group_id = item.get_group_linear_id();

					int block_i = block_diag.i_first + group_id;
					int block_j = block_diag.j_first + group_id;

					int i0 = block_i * SG_SIZE;
					int left_border = block_j * block_width;
					int right_border = left_border + block_width;

					TiledCache<TILE_M, TILE_N> cache;
					load_cache(cache, acc.a, acc.h_strands, i0 + sg_id, i_stride, TILE_M);


					auto update_cell = [&](int i_low, int j_low)
					{
						update_cell_tile_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
							i_stride, j_stride, i_low, j_low);
					};

					update_stripe<SG_SIZE>(update_cell, sg, i0, left_border, right_border);

					store_cache(cache, acc.h_strands, i0 + sg_id, i_stride, TILE_M);
				});
			});
		}
	}

	copy_strands_deinterleaved(ctx, grid);
}


