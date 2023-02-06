#pragma once

#include "lcs_types.hpp"
#include "lcs_grid.hpp"
#include "lcs_common.hpp"

template<int SG_SIZE, int TILE_M, int TILE_N>
void
Lcs_Semi_Tiled_ST_OldScoping(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int stripe_size = SG_SIZE;
	int complete_stripe_count = region_m / stripe_size;
	bool has_incomplete_stripe = complete_stripe_count * stripe_size != region_m;
	{
		auto buf = make_buffers(grid);

		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);
			int left_border = 0;
			int right_border = region_n;

			int local_size = SG_SIZE;
			int global_size = SG_SIZE;

			cgh.parallel_for(
				sycl::nd_range<1>(global_size, local_size),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				int sg_id = sg.get_local_linear_id();

				int top_stripe = has_incomplete_stripe ? complete_stripe_count : complete_stripe_count - 1;
				int top_border = region_m;

				TiledCache<TILE_M, TILE_N> cache_incomplete;
				TiledCache<TILE_M, TILE_N> cache_complete;

				// Special case for incomplete stripe: some entries in cache are masked out
				auto update_cell_tile_limit_incomplete = [&](int i_low, int j_low)
				{
					int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
					int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

					update_cell_tile_limit_semilocal(cache_incomplete,
													 acc.a, acc.b, acc.h_strands, acc.v_strands,
													 i_stride, j_stride, i_low, j_low,
													 ii_limit_here, jj_limit_here);
				};

				auto within_bounds = [&](int i, int j)
				{
					return i < top_border;
				};

				auto update_cell_tile_complete = [&](int i_low, int j_low)
				{
					update_cell_tile_semilocal(cache_complete,
											   acc.a, acc.b, acc.h_strands, acc.v_strands,
											   i_stride, j_stride, i_low, j_low);
				};

				auto update_cell_tile_limit_complete = [&](int i_low, int j_low)
				{
					int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
					int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

					update_cell_tile_limit_semilocal(cache_complete,
													 acc.a, acc.b, acc.h_strands, acc.v_strands,
													 i_stride, j_stride, i_low, j_low,
													 ii_limit_here, jj_limit_here);
				};

				// Run special code path for top stripe that assumes both incomplete stripe
				// and incomplete tile
				{
					int i0 = top_stripe * SG_SIZE;
					int i = i0 + sg_id;

					int ii_limit_here = i == region_m - 1 ? ii_limit : TILE_M;
					// Load cache
					load_cache(cache_incomplete, acc.a, acc.h_strands, i, i_stride, ii_limit_here);

					stripe_iterate_with_predicate<SG_SIZE>(update_cell_tile_limit_incomplete, within_bounds,
														   sg, i0, left_border, right_border);

					store_cache(cache_incomplete, acc.h_strands, i, i_stride, ii_limit_here);
					sg.barrier();
				}

				for (int stripe = top_stripe - 1; stripe >= 0; --stripe)
				{
					int i0 = stripe * SG_SIZE;
					int i = i0 + sg_id;
					load_cache(cache_complete, acc.a, acc.h_strands, i, i_stride, TILE_M);
					stripe_iterate<SG_SIZE>(update_cell_tile_complete, update_cell_tile_limit_complete,
											sg, i0, left_border, right_border);
					store_cache(cache_complete, acc.h_strands, i, i_stride, TILE_M);

					sg.barrier();
				}

			}); // end parallel_for
		}); // end submit
	} // end buffer lifetime

	copy_strands_deinterleaved(ctx, grid);
}


template<int SG_SIZE, int TILE_M, int TILE_N>
void
Lcs_Semi_Tiled_ST_NewScoping(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int stripe_size = SG_SIZE;
	int complete_stripe_count = region_m / stripe_size;
	bool has_incomplete_stripe = complete_stripe_count * stripe_size != region_m;
	{
		auto buf = make_buffers(grid);

		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);
			int left_border = 0;
			int right_border = region_n;

			int local_size = SG_SIZE;
			int global_size = SG_SIZE;

			cgh.parallel_for(
				sycl::nd_range<1>(global_size, local_size),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				int sg_id = sg.get_local_linear_id();

				int top_stripe = has_incomplete_stripe ? complete_stripe_count : complete_stripe_count - 1;
				int top_border = region_m;



				// Run special code path for top stripe that assumes both incomplete stripe
				// and incomplete tile
				{
					TiledCache<TILE_M, TILE_N> cache_incomplete;

					// Special case for incomplete stripe: some entries in cache are masked out
					auto update_cell_tile_limit_incomplete = [&](int i_low, int j_low)
					{
						int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
						int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

						update_cell_tile_limit_semilocal(cache_incomplete,
														 acc.a, acc.b, acc.h_strands, acc.v_strands,
														 i_stride, j_stride, i_low, j_low,
														 ii_limit_here, jj_limit_here);
					};

					auto within_bounds = [&](int i, int j)
					{
						return i < top_border;
					};


					int i0 = top_stripe * SG_SIZE;
					int i = i0 + sg_id;

					int ii_limit_here = i == region_m - 1 ? ii_limit : TILE_M;

					load_cache(cache_incomplete, acc.a, acc.h_strands, i, i_stride, ii_limit_here);

					stripe_iterate_with_predicate<SG_SIZE>(update_cell_tile_limit_incomplete, within_bounds,
														   sg, i0, left_border, right_border);

					store_cache(cache_incomplete, acc.h_strands, i, i_stride, ii_limit_here);
					sg.barrier();
				}

				for (int stripe = top_stripe - 1; stripe >= 0; --stripe)
				{
					TiledCache<TILE_M, TILE_N> cache_complete;
					auto update_cell_tile_complete = [&](int i_low, int j_low)
					{
						update_cell_tile_semilocal(cache_complete,
												   acc.a, acc.b, acc.h_strands, acc.v_strands,
												   i_stride, j_stride, i_low, j_low);
					};

					auto update_cell_tile_limit_complete = [&](int i_low, int j_low)
					{
						int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
						int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

						update_cell_tile_limit_semilocal(cache_complete,
														 acc.a, acc.b, acc.h_strands, acc.v_strands,
														 i_stride, j_stride, i_low, j_low,
														 ii_limit_here, jj_limit_here);
					};

					int i0 = stripe * SG_SIZE;
					int i = i0 + sg_id;
					load_cache(cache_complete, acc.a, acc.h_strands, i, i_stride, TILE_M);
					stripe_iterate<SG_SIZE>(update_cell_tile_complete, update_cell_tile_limit_complete,
											sg, i0, left_border, right_border);
					store_cache(cache_complete, acc.h_strands, i, i_stride, TILE_M);

					sg.barrier();
				}

			}); // end parallel_for
		}); // end submit
	} // end buffer lifetime

	copy_strands_deinterleaved(ctx, grid);
}



// Works for exact multiples only!
template<int SG_SIZE, int TILE_M, int TILE_N>
void
Lcs_Semi_Tiled_ST_Exact(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	// NOTE: this code does not handle tile limits at all
	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();
	UNUSED(ii_limit);
	UNUSED(jj_limit);

	int stripe_size = SG_SIZE;
	int complete_stripe_count = region_m / stripe_size;

	{
		auto buf = make_buffers(grid);

		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);
			int left_border = 0;
			int right_border = region_n;

			int local_size = SG_SIZE;
			int global_size = SG_SIZE;

			cgh.parallel_for(
				sycl::nd_range<1>(global_size, local_size),
				[=](sycl::nd_item<1> item)
				[[intel::reqd_sub_group_size(SG_SIZE)]]
			{
				auto sg = item.get_sub_group();
				int sg_id = sg.get_local_linear_id();

				// Complete stripes
				for (int stripe = complete_stripe_count - 1; stripe >= 0; --stripe)
				{
					TiledCache<TILE_M, TILE_N> cache1;
					auto update_cell_tile = [&](int i_low, int j_low)
					{
						update_cell_tile_semilocal(cache1, acc.a, acc.b, acc.h_strands, acc.v_strands,
												   i_stride, j_stride, i_low, j_low);
					};

					int i0 = stripe * stripe_size;
					int i = i0 + sg_id;
					load_cache(cache1, acc.a, acc.h_strands, i, i_stride, TILE_M);
					update_stripe<SG_SIZE>(update_cell_tile, sg, i0, left_border, right_border);
					store_cache(cache1, acc.h_strands, i, i_stride, TILE_M);
				}
			}); // end parallel_for
		}); // end submit
	} // end buffer lifetime

	copy_strands_deinterleaved(ctx, grid);
}



template<int SG_SIZE, int TILE_M, int TILE_N>
void
Lcs_Semi_Tiled_MT_Verbose(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int stripe_size = SG_SIZE;
	int complete_stripe_count = region_m / stripe_size;
	bool has_incomplete_stripe = complete_stripe_count * stripe_size != region_m;

	int num_blocks_m = complete_stripe_count + int(has_incomplete_stripe);
	int num_subdivisions = 32;
	int block_width = region_n / num_subdivisions;
	int num_blocks_n = CeilDiv(region_n, block_width);
	int pass_count = num_blocks_m + num_blocks_n - 1;

	{
		auto buf = make_buffers(grid);

		for (int pass = 0; pass < pass_count; ++pass)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);
				
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

					int i0 = (block_diag.i_first + group_id) * SG_SIZE;
					int left_border = (block_diag.j_first + group_id) * block_width;
					int right_border = left_border + block_width;
					right_border = Min(right_border, region_n);
					int top_border = region_m;

					int i = i0 + sg_id;

					//// Update a single stripe from left_border to right_border
					//if (i0 == num_blocks_m - 1) // If this is topmost stripe
					//{
					//	TiledCache<TILE_M, TILE_N> cache_incomplete;

					//	// Special case for incomplete stripe: some entries in cache are masked out
					//	auto update_cell_tile_limit_incomplete = [&](int i_low, int j_low)
					//	{
					//		int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
					//		int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

					//		update_cell_tile_limit_semilocal(cache_incomplete,
					//										 acc.a, acc.b, acc.h_strands, acc.v_strands,
					//										 i_stride, j_stride, i_low, j_low,
					//										 ii_limit_here, jj_limit_here);
					//	};

					//	auto within_bounds = [&](int i, int j)
					//	{
					//		return i < top_border;
					//	};


					//	int ii_limit_here = i == region_m - 1 ? ii_limit : TILE_M;

					//	load_cache(cache_incomplete, acc.a, acc.h_strands, i, i_stride, ii_limit_here);

					//	stripe_iterate_with_predicate<SG_SIZE>(update_cell_tile_limit_incomplete, within_bounds,
					//										   sg, i0, left_border, right_border);

					//	store_cache(cache_incomplete, acc.h_strands, i, i_stride, ii_limit_here);
					//	sg.barrier();
					//}
					// else // If this is a complete stripe
					{
						TiledCache<TILE_M, TILE_N> cache_complete;
						auto update_cell_tile_complete = [&](int i_low, int j_low)
						{
							update_cell_tile_semilocal(cache_complete,
													   acc.a, acc.b, acc.h_strands, acc.v_strands,
													   i_stride, j_stride, i_low, j_low);
						};

						//auto update_cell_tile_limit_complete = [&](int i_low, int j_low)
						//{
						//	int ii_limit_here = i_low == region_m - 1 ? ii_limit : TILE_M;
						//	int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;

						//	update_cell_tile_limit_semilocal(cache_complete,
						//									 acc.a, acc.b, acc.h_strands, acc.v_strands,
						//									 i_stride, j_stride, i_low, j_low,
						//									 ii_limit_here, jj_limit_here);
						//};

						load_cache(cache_complete, acc.a, acc.h_strands, i, i_stride, TILE_M);
						stripe_iterate<SG_SIZE>(update_cell_tile_complete, update_cell_tile_complete,
												sg, i0, left_border, right_border);
						store_cache(cache_complete, acc.h_strands, i, i_stride, TILE_M);

						sg.barrier();
					}


				}); // end parallel_for
			}); // end submit
		}// end for
	} // end buffer lifetime

	copy_strands_deinterleaved(ctx, grid);
}


template<int SG_SIZE, int TILE_M, int TILE_N>
void Lcs_Semi_Tiled_MT_better(const LcsInput &input, LcsContext &ctx)
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

	int block_width = region_n / 32;
	int num_blocks_n = region_n / block_width + (region_n % block_width != 0);

	int pass_count = num_blocks_m + num_blocks_n - 1;

	{
		auto buf = make_buffers(grid);

		for (int pass = 0; pass < pass_count; ++pass)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

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
						


						if ((group_id == block_diag.diag_len - 1) || (right_border > region_n))
						{
							TiledCache<TILE_M, TILE_N> cache;
							

							right_border = Min(right_border, region_n);
							int ii_limit_here = group_id == block_diag.diag_len - 1 ? ii_limit : TILE_M;
							int jj_limit_here = right_border == region_n ? jj_limit : TILE_N;
							
							auto process_cell_tile_limit = [&](int i_low, int j_low)
							{
								update_cell_tile_limit_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
																 i_stride, j_stride, i_low, j_low,
																 ii_limit_here, jj_limit_here);
							};

							load_cache(cache, acc.a, acc.h_strands, i, i_stride, ii_limit_here);

							update_stripe<SG_SIZE>(process_cell_tile_limit,
												   sg, i0, left_border, right_border);
							store_cache(cache, acc.h_strands, i, i_stride, ii_limit_here);

						} 
						else
						{
							TiledCache<TILE_M, TILE_N> cache;
							
							auto process_cell_tile = [&](int i_low, int j_low)
							{
								update_cell_tile_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
														   i_stride, j_stride, i_low, j_low);
							};

							load_cache(cache, acc.a, acc.h_strands, i, i_stride, TILE_M);

							update_stripe<SG_SIZE>(process_cell_tile,
												   sg, i0, left_border, right_border);
							store_cache(cache, acc.h_strands, i, i_stride, TILE_M);
						}

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

	int block_width = region_n / 32;
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

template<int SG_SIZE, int TILE_M, int TILE_N, int SUBDIVISIONS>
void Lcs_Semi_Tiled_MT_good_threads(const LcsInput &input, LcsContext &ctx)
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

	int block_width = region_n / SUBDIVISIONS;
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

#include <sycl/ext/oneapi/experimental/builtins.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

static const CONSTANT char FMT_GROUP[] = "group_id: %d, left_border: %d, right_border: %d\n";
// static const CONSTANT char FMT_GROUP[]

#define K_PRINTF(FMT, ...) const CONSTANT char FMT_[] = FMT; sycl::ext::oneapi::experimental::printf(FMT_, __VA_ARGS__)

template<int SG_SIZE, int TILE_M, int TILE_N, int SUBDIVISIONS>
void Lcs_Semi_Tiled_MT_Correct(const LcsInput &input, LcsContext &ctx)
{
	auto grid = make_embedding_deinterleaved<TILE_M, TILE_N>(input);

	int region_m = grid.h_desc.RegionUsefulSize();
	int region_n = grid.v_desc.RegionUsefulSize();

	int i_stride = grid.h_desc.Stride();
	int j_stride = grid.v_desc.Stride();

	int ii_limit = grid.h_desc.IncompleteTileLimit();
	int jj_limit = grid.v_desc.IncompleteTileLimit();

	int stripe_size = SG_SIZE;
	int stripe_count = CeilDiv(region_m, stripe_size);
	int complete_stripe_count = region_m / stripe_size;
	int num_blocks_m = stripe_count;

	int block_width = CeilDiv(region_n, SUBDIVISIONS);
	int num_blocks_n = CeilDiv(region_n, block_width);

	int pass_count = num_blocks_m + num_blocks_n - 1;
	{
		auto buf = make_buffers(grid);

		for (int pass = 0; pass < pass_count; ++pass)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

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

					int i0 = (block_diag.i_first + group_id) * SG_SIZE;
					int left_border = (block_diag.j_first + group_id) * block_width;
					int right_border = Min(left_border + block_width, region_n);
					
					int left_border_original = left_border;
					int right_border_original = right_border;

					int i = i0 + sg_id;

					bool incomplete_stripe =
						i0 == complete_stripe_count * SG_SIZE
						&& complete_stripe_count < stripe_count;

					bool incomplete_tile = 
						((i0 == (stripe_count - 1) * SG_SIZE) && (ii_limit != TILE_M))
						|| (right_border >= region_n && jj_limit != TILE_N);

					bool incomplete = incomplete_stripe || incomplete_tile;
					
					
					if (!incomplete_stripe)
					{
						if (right_border >= region_n && jj_limit != TILE_N)
						{
							right_border = region_n - 1;
						}

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

					if(incomplete_stripe || right_border < right_border_original)
					{
						if (sg_id == 0)
						{
							// K_PRINTF("i0 = %d, left_border = %d, right_border = %d\n", i0, left_border, right_border);
						}

						if (right_border < right_border_original)
						{
							left_border = right_border;
							right_border = right_border_original;
						}

						int ii_limit_here = i == region_m - 1 ? ii_limit : TILE_M;

						TiledCache<TILE_M, TILE_N> cache;
						load_cache(cache, acc.a, acc.h_strands, i, i_stride, ii_limit_here);

						auto process_cell_tile = [&](int i_low, int j_low)
						{
							int jj_limit_here = j_low == region_n - 1 ? jj_limit : TILE_N;
							update_cell_tile_limit_semilocal(cache, acc.a, acc.b, acc.h_strands, acc.v_strands,
															 i_stride, j_stride, i_low, j_low,
															 ii_limit_here, jj_limit_here);
						};
						auto within_bounds = [&](int i, int j)
						{
							return i < region_m;
						};

						/*update_stripe<SG_SIZE>(process_cell_tile,
											   sg, i0, left_border, right_border);*/

						if (1)
						{
							//if (sg_id == 0)
							//{
							//	K_PRINTF("%dx%d --- i0=%d --- %d-%d\n", region_m, region_n, i0, left_border, right_border);
							//}
							update_stripe_with_predicate<SG_SIZE>(process_cell_tile, within_bounds, sg, i0,
																  left_border, right_border);
						}

						store_cache(cache, acc.h_strands, i, i_stride, ii_limit_here);
					}


				});
			});
		}
	}

	copy_strands_deinterleaved(ctx, grid);
}

