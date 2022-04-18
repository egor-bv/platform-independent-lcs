#pragma once

#include <CL/sycl.hpp>
#include "lcs_common.hpp"

template<typename Symbol, typename Strand>
class GridBuffers
{
public:
	sycl::buffer<Symbol, 1> a;
	sycl::buffer<Symbol, 1> b;
	sycl::buffer<Strand, 1> h_strands;
	sycl::buffer<Strand, 1> v_strands;

	GridBuffers(LcsContext &ctx)
		: a(ctx.a, ctx.m)
		, b(ctx.b, ctx.n)
		, h_strands(ctx.h_strands, ctx.m)
		, v_strands(ctx.v_strands, ctx.n)
	{
	}
};



template<typename Symbol, typename Strand, typename CGH>
class GridAccessors
{
public:
	sycl::accessor<Symbol, 1, sycl::access_mode::read> a;
	sycl::accessor<Symbol, 1, sycl::access_mode::read> b;
	sycl::accessor<Strand, 1, sycl::access_mode::read_write> h_strands;
	sycl::accessor<Strand, 1, sycl::access_mode::read_write> v_strands;

	GridAccessors(GridBuffers &g, CGH &cgh)
		: a(g.a)
		, b(g.b)
		, h_strands(g.h_strands)
		, v_strands(g.v_strands)
	{
	}
};