#pragma once

#include "utility.hpp"
#include <CL/sycl.hpp>

/// <summary>
/// Comb sticky braids in a staircase pattern, using global memory to exchange data between consequtive iterations
/// (Incomplete, only works for m -- multiple of SG_SIZE)
/// * Template parameter SG_POW2 must be 2 or 3
/// </summary>
/// <param name="q">command queue</param>
/// <param name="_a_rev">first sequence, reversed</param>
/// <param name="_b">second sequence, in original order</param>
/// <param name="m">length of first sequence</param>
/// <param name="n">length of second sequence</param>
/// <param name="_h_strands">horizontal strands, initialized in reverse order</param>
/// <param name="_v_strands">vertical strands, initialized in original order</param>
template <int SG_POW2>
void StickyBraidComb_StaircaseGlobal(sycl::queue q, const int *_a_rev, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a_rev(_a_rev, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	constexpr size_t SG_SIZE = 1 << SG_POW2;
	const int num_rows = m / SG_SIZE;
	q.submit([&](auto &h)
		{
			auto a_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			h.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					auto sg = item.get_sub_group();
					int sg_id = sg.get_local_linear_id();

					// topmost row

					for (int row = 0; row < num_rows; ++row)
					{
						int i = (num_rows - row - 1) * SG_SIZE + sg_id;
						int a_sym = a_rev[i];
						int h = h_strands[i];

						// first columns

						for (int horz_step = 1 - SG_SIZE; horz_step < 0; ++horz_step)
						{
							int j = horz_step + sg_id;
							if (j >= 0)
							{
								int b_sym = b[j];
								int v = v_strands[j];

								bool need_swap = a_sym == b_sym || h > v;
								int new_h = need_swap ? v : h;
								int new_v = need_swap ? h : v;
								h = new_h;
								v = new_v;
								v_strands[j] = v;
							}
							sg.barrier();
						}

						// middle columns

						for (int horz_step = 0; horz_step < n - SG_SIZE; ++horz_step)
						{
							int j = horz_step + sg_id;
							int b_sym = b[j];
							int v = v_strands[j];

							bool need_swap = a_sym == b_sym || h > v;
							int new_h = need_swap ? v : h;
							int new_v = need_swap ? h : v;
							h = new_h;
							v = new_v;
							v_strands[j] = v;
							sg.barrier();
						}

						// last columns
						for (int horz_step = n - SG_SIZE; horz_step < n; ++horz_step)
						{
							int j = horz_step + sg_id;
							if (j < n)
							{
								int b_sym = b[j];
								int v = v_strands[j];

								bool need_swap = a_sym == b_sym || h > v;
								int new_h = need_swap ? v : h;
								int new_v = need_swap ? h : v;
								h = new_h;
								v = new_v;
								v_strands[j] = v;
							}
							sg.barrier();
						}

						h_strands[i] = h;
						sg.barrier();
					}
				}
			);

		}
	);
}


/// <summary>
/// Comb sticky braids in a staircase pattern, using shuffle operations to exchange data between consequtive iterations
/// (Incomplete, doesn't work at all at the moment)
/// </summary>
/// <param name="q">command queue</param>
/// <param name="_a_rev">first sequence, reversed</param>
/// <param name="_b">second sequence, in original order</param>
/// <param name="m">length of first sequence</param>
/// <param name="n">length of second sequence</param>
/// <param name="_h_strands">horizontal strands, initialized in reverse order</param>
/// <param name="_v_strands">vertical strands, initialized in original order</param>
void StickyBraidComb_StaircaseCrosslane(sycl::queue q, const int *_a_rev, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a_rev(_a_rev, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	constexpr size_t SG_POW2 = 3;
	constexpr size_t SG_SIZE = 1 << SG_POW2;
	constexpr size_t SG_ID_LAST = SG_SIZE - 1;
	const int num_rows = m / SG_SIZE;
	const int num_full_horz_steps = n / SG_SIZE;

	q.submit([&](auto &h)
		{
			auto a_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			h.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					// TODO: do everything *very* carefully
					auto sg = item.get_sub_group();
					int sg_id = sg.get_local_linear_id();

					// starting with aligned rows...
					for (int row = 0; row < num_rows; ++row)
					{
						int i = (num_rows - row - 1) * SG_SIZE + sg_id;
						int a_sym = a_rev[i];
						int h = h_strands[i];

						int v0 = 0;
						int v1 = 0;
						int sym_eq0 = 0;
						int sym_eq1 = 0;
						int b_sym0 = 0;
						int b_sym1 = 0;

						int j0 = sg_id;
						v0 = v_strands[j0];
						b_sym0 = b[j0];
						sym_eq0 = b_sym0 == a_sym;

						int sym_eq = 0;
						int v = 0;

						// main iteration part
						for (int horz_step = 0; horz_step < num_full_horz_steps; ++horz_step)
						{
							int j1 = j0 + SG_SIZE;
							v1 = sg.shuffle_xor(v_strands[j1], 7); // reverse
							// v1 = sg.shuffle_xor(v1, 7); // reverse
							b_sym1 = v_strands[j1];
							sym_eq1 = b_sym1 == a_sym;
							sym_eq1 = sg.shuffle_xor(sym_eq1, 7);

							// first iteration is directly on v0
							v = v0;
							sym_eq = sym_eq0;
							// this loop will be inlined
							for (int s0 = 0; s0 < SG_SIZE; ++s0)
							{
								int has_crossing = h > v;
								int need_swap = has_crossing || sym_eq;
								int new_h = need_swap ? v : h;
								int new_v = need_swap ? h : v;

								// 1. stash first element... (into v0?)
								v0 = sg.shuffle_up(v0, 1);
								if (sg_id == 0) v0 = new_v;

								// 2. shift left by one
								v = sg.shuffle_down(new_v, 1);

								// 3. load last one in
								if (sg_id == SG_ID_LAST) v = v1;
								v1 = sg.shuffle_up(v1, 1);

								sym_eq = sg.shuffle_down(sym_eq, 1);
								if (sg_id == SG_ID_LAST) sym_eq = sym_eq1;
								sym_eq1 = sg.shuffle_up(sym_eq1, 1);

								h = new_h;
							}
							v_strands[j0] = sg.shuffle_xor(v0, 7); // reverse
							v0 = v; // ???
							sym_eq0 = sym_eq1;

						}

						// at last
						h_strands[i] = h;
					}

				}
			);

		}
	);
}

template <int SG_POW2>
void StickyBraidComb_StaircaseLocal(sycl::queue q, const int *_a_rev, const int *_b, int m, int n, int *_h_strands, int *_v_strands)
{
	sycl::buffer<int, 1> buf_a_rev(_a_rev, m);
	sycl::buffer<int, 1> buf_b(_b, n);
	sycl::buffer<int, 1> buf_h_strands(_h_strands, m);
	sycl::buffer<int, 1> buf_v_strands(_v_strands, n);

	constexpr size_t SG_SIZE = 1 << SG_POW2;
	const int num_rows = m / SG_SIZE;
	q.submit([&](auto &h)
		{
			auto a_rev = buf_a_rev.get_access<sycl::access::mode::read>(h);
			auto b = buf_b.get_access<sycl::access::mode::read>(h);
			auto h_strands = buf_h_strands.get_access<sycl::access::mode::read_write>(h);
			auto v_strands = buf_v_strands.get_access<sycl::access::mode::read_write>(h);

			h.parallel_for(
				sycl::nd_range<1>(SG_SIZE, SG_SIZE),
				[=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					auto sg = item.get_sub_group();
					int sg_id = sg.get_local_linear_id();

					// topmost row

					for (int row = 0; row < num_rows; ++row)
					{
						int i = (num_rows - row - 1) * SG_SIZE + sg_id;
						int a_sym = a_rev[i];
						int h = h_strands[i];


						h_strands[i] = h;
						sg.barrier();
					}
				}
			);

		}
	);
}