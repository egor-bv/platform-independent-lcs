#pragma once

#include "lcs_interface_internal.hpp"
#include "lcs_residual_fixup.hpp"
#include "lcs_grid.hpp"
#include "lcs_common.hpp"
#include "sycl_utility.hpp"


template <int SG_SIZE, int TILE_M, int TILE_N>
void
Lcs_Semi_Tiled_ST_(LcsProblemContext &ctx)
{
	auto grid = make_grid_embedding_tiled(ctx, TILE_M, SG_SIZE, TILE_N, SG_SIZE * 2);
	auto shape = grid.shape;

	int stripe_count = shape.h_desc.tile_count / SG_SIZE;
	{
		auto buf = make_buffers(grid);
		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);

			int local_size = SG_SIZE;
			int global_size = SG_SIZE;

			cgh.parallel_for(
				sycl::nd_range<1>(global_size, local_size),
				[=](sycl::nd_item<1> item) REQD_SUB_GROUP_SIZE(SG_SIZE)
			{
				auto sg = item.get_sub_group();

				for (int stripe_idx = 0; stripe_idx < stripe_count; ++stripe_idx)
				{
					int i0 = (stripe_count - stripe_idx - 1) * SG_SIZE;

					BlockShape block = {};
					block.i0 = i0;
					block.j0 = 0;
					block.jsize = shape.v_desc.tile_count;

					update_block_tiled_semilocal_with_cache<SG_SIZE, TILE_M, TILE_N>(sg, acc, shape, block);
				}


			}); // end parallel_for
		}); // end submit
	} // end buffers lifetime

	copy_strands_and_fixup_tiled(ctx, grid);
	// fixup_residuals(ctx);

}

template <int SG_SIZE, int TILE_M, int TILE_N>
void
Lcs_Semi_Tiled_ST(LcsProblemContext &ctx)
{
	auto grid = make_grid_embedding_tiled(ctx, TILE_M, SG_SIZE, TILE_N, SG_SIZE*2);
	auto shape = grid.shape;
	auto stripes = divide_tiled_into_stripes(shape, SG_SIZE);

	{
		auto buf = make_buffers(grid);
		ctx.queue->submit([&](sycl::handler &cgh)
		{
			auto acc = make_accessors(buf, cgh);

			int local_size = SG_SIZE;
			int global_size = SG_SIZE;

			cgh.parallel_for(
				sycl::nd_range<1>(global_size, local_size),
				[=](sycl::nd_item<1> item) REQD_SUB_GROUP_SIZE(SG_SIZE)
			{
				auto sg = item.get_sub_group();

				for (int block_idx = 0; block_idx < stripes.block_count; ++block_idx)
				{
					auto block = stripes.block_at(block_idx);

					update_block_tiled_semilocal_with_cache<SG_SIZE, TILE_M, TILE_N>(sg, acc, shape, block);
				}


			}); // end parallel_for
		}); // end submit
	} // end buffers lifetime

	copy_strands_and_fixup_tiled(ctx, grid);
	// fixup_residuals(ctx);

}

template <int SG_SIZE, int TILE_M, int TILE_N, int SECTIONS>
void
Lcs_Semi_Tiled_MT(LcsProblemContext &ctx)
{
	auto grid = make_grid_embedding_tiled(ctx, TILE_M, SG_SIZE, TILE_N, SG_SIZE * SECTIONS);
	auto shape = grid.shape;

	int block_width = grid.shape.v_desc.tile_count / SECTIONS;
	auto blocks = divide_tiled_into_blocks(shape, SG_SIZE, block_width);

	{
		auto buf = make_buffers(grid);

		for (int pass_idx = 0; pass_idx < blocks.pass_count(); ++pass_idx)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

				auto diag = blocks.diagonal_at(pass_idx);

				int local_size = SG_SIZE;
				int global_size = local_size * diag.len;

				cgh.parallel_for(
					sycl::nd_range<1>(global_size, local_size),
					[=](sycl::nd_item<1> item) REQD_SUB_GROUP_SIZE(SG_SIZE)
				{
					auto sg = item.get_sub_group();
					int group_id = item.get_group_linear_id();

					auto block = blocks.block_at(diag, group_id);
					update_block_tiled_semilocal_with_cache<SG_SIZE, TILE_M, TILE_N>(sg, acc, shape, block);

				}); // end parallel_for
			}); // end submit
		}// end for
	} // end buffers lifetime

	copy_strands_and_fixup_tiled(ctx, grid);
	// fixup_residuals(ctx);

}



template <int SG_SIZE, int TILE_M, int TILE_N, int SECTIONS>
void
Lcs_Semi_Tiled_MT_SLM(LcsProblemContext &ctx)
{
	auto grid = make_grid_embedding_tiled(ctx, TILE_M, SG_SIZE, TILE_N, SG_SIZE * SECTIONS);
	auto shape = grid.shape;

	int slm_max_bytes = 32 * 1024;
	int slm_slot_size = SG_SIZE * TILE_N;
	int slm_bytes_per_slot = slm_slot_size * sizeof(Word) * 2;

	int block_width = grid.shape.v_desc.tile_count / SECTIONS;
	auto blocks = divide_tiled_into_blocks(shape, SG_SIZE, block_width);

	int slm_slot_count = block_width / SG_SIZE;
	int slm_size = slm_slot_size * slm_slot_count;

	printf("slm allocation: %.1fkB\n", (double)slm_slot_count * slm_bytes_per_slot / 1024.0);

	{
		auto buf = make_buffers(grid);

		for (int pass_idx = 0; pass_idx < blocks.pass_count(); ++pass_idx)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

				auto b_local = sycl::accessor<Word, 1, sycl::access_mode::read_write, sycl::access::target::local>(slm_size, cgh);
				auto v_strands_local = sycl::accessor<Word, 1, sycl::access_mode::read_write, sycl::access::target::local>(slm_size, cgh);


				auto diag = blocks.diagonal_at(pass_idx);

				int local_size = SG_SIZE;
				int global_size = local_size * diag.len;

				cgh.parallel_for(
					sycl::nd_range<1>(global_size, local_size),
					[=](sycl::nd_item<1> item) REQD_SUB_GROUP_SIZE(SG_SIZE)
				{
					auto sg = item.get_sub_group();
					int group_id = item.get_group_linear_id();

					auto block = blocks.block_at(diag, group_id);
					update_block_tiled_semilocal_with_slm_at_once<SG_SIZE, TILE_M, TILE_N>(sg, acc, slm_slot_count, b_local, v_strands_local, shape, block);

				}); // end parallel_for
			}); // end submit
		}// end for
	} // end buffers lifetime

	copy_strands_and_fixup_tiled(ctx, grid);
	// fixup_residuals(ctx);

}

template <int SG_SIZE, int TILE_M, int TILE_N>
void
Lcs_Semi_Tiled_MT_SLM_incremental(LcsProblemContext &ctx)
{
	auto grid = make_grid_embedding_tiled(ctx, TILE_M, SG_SIZE, TILE_N, SG_SIZE * TILE_N * 8);
	auto shape = grid.shape;

	int block_width = grid.shape.v_desc.tile_count / 4;
	auto blocks = divide_tiled_into_blocks(shape, SG_SIZE, block_width);


	int slm_slot_size = SG_SIZE * TILE_N;
	int slm_slot_count = 8;
	int slm_size = slm_slot_size * slm_slot_count;


	{
		auto buf = make_buffers(grid);

		for (int pass_idx = 0; pass_idx < blocks.pass_count(); ++pass_idx)
		{
			ctx.queue->submit([&](sycl::handler &cgh)
			{
				auto acc = make_accessors(buf, cgh);

				auto b_local = sycl::accessor<Word, 1, sycl::access_mode::read_write, sycl::access::target::local>(slm_size, cgh);
				auto v_strands_local = sycl::accessor<Word, 1, sycl::access_mode::read_write, sycl::access::target::local>(slm_size, cgh);


				auto diag = blocks.diagonal_at(pass_idx);

				int local_size = SG_SIZE;
				int global_size = local_size * diag.len;

				cgh.parallel_for(
					sycl::nd_range<1>(global_size, local_size),
					[=](sycl::nd_item<1> item) REQD_SUB_GROUP_SIZE(SG_SIZE)
				{
					auto sg = item.get_sub_group();
					int group_id = item.get_group_linear_id();

					auto block = blocks.block_at(diag, group_id);
					// K_PRINTF("slm_size: %d\n", slm_size);
					update_block_tiled_semilocal_with_slm<SG_SIZE, TILE_M, TILE_N>(sg, acc, slm_slot_count, b_local, v_strands_local, shape, block);

				}); // end parallel_for
			}); // end submit
		}// end for
	} // end buffers lifetime

	copy_strands_and_fixup_tiled(ctx, grid);
	// fixup_residuals(ctx);

}